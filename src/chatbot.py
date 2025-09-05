from document_processor import DocumentProcessor
from vector_store import FAISSVectorStore
from langchain_openai import ChatOpenAI
from pathlib import Path
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from langchain.callbacks.base import BaseCallbackHandler

class StreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.current_text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="", flush=True)   # stream to console
        self.current_text += token

    def get_text(self):
        return self.current_text


class BioethicsChatbot:
    def __init__(self, data_dir: str="data/sample_papers"):
        self.processor = DocumentProcessor()
        self.vector_store = FAISSVectorStore()
        self.history = []
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.65,
            'low': 0.5}

        if not self.vector_store.load_index():
            print("No existing vector store, creating one...")
            pdf_files = list(Path(data_dir).glob("*.pdf"))
            if not pdf_files:
                raise ValueError(f"No PDFs found in {data_dir}")

            chunks = self.processor.process_documents([str(p) for p in pdf_files])
            self.vector_store.add_documents(chunks)
            logger.info("Indexed %d documents.", len(chunks))

        else:
            logger.info("Index loaded from disk")

        self.stream_handler = StreamHandler()
        self.llm = ChatOpenAI(model="gpt-4o-mini", streaming=True,
                              callbacks=[self.stream_handler])

    def add_new_document(self, pdf_path: str):
        filename = Path(pdf_path).name

        # Check if already in the index
        existing_files = {doc["metadata"].get("filename") for doc in self.vector_store.documents}
        if filename in existing_files:
            print(f"Skipping {filename}: already indexed.")
            return

        # Otherwise process & add
        chunks = self.processor.process_document(pdf_path)
        self.vector_store.add_documents(chunks)
        print(f"Added {len(chunks)} chunks from {pdf_path}")

    def get_citation_confidence(self, similarity_score: float) -> str:
        """Determine citation confidence level based on similarity score"""
        if similarity_score >= self.confidence_thresholds['high']:
            return "high_confidence"
        elif similarity_score >= self.confidence_thresholds['medium']:
            return "medium_confidence"
        elif similarity_score >= self.confidence_thresholds['low']:
            return "low_confidence"
        return "context_only"

    def ask(self, question: str, k: int = 10) -> str:
        # Step 1: Retrieve relevant chunks
        results = self.vector_store.search(question, k=k)

        # DEBUG: Print what we found
        print(f"Found {len(results)} results for query: '{question}'")
        for i, r in enumerate(results[:3]):  # Show top 3
            print(f"Result {i + 1} (score: {r.get('similarity_score', 'N/A'):.3f}): {r['content'][:200]}...")

        if not results:
            return "I couldn't find relevant information in the documents."

        # Step 2: Build context from retrieved chunks
        context_blocks = []
        citation_groups = {
            'high_confidence': [],
            'medium_confidence': [],
            'low_confidence': [],
        }
        for r in results:
            title = r["metadata"].get("title", None)
            authors = r["metadata"].get("authors", None)
            year = r["metadata"].get("year", "n.d.")

            confidence = self.get_citation_confidence(r["similarity_score"])

            block = (
                f"Source: {authors} ({year}). *{title}* "
                f"[chunk {r['metadata'].get('chunk_id', '?')}, confidence: {confidence}]\n"
                f"{r['content']}\n"
            )

            context_blocks.append(block)
            if authors is not None and authors != "Unknown Author(s)":
                citation_groups[confidence].append(block)

        history_text = "\n".join(
            [f"User: {u}\nBot: {b}" for u, b in self.history[-4:]]
        ) or "No previous conversation."
        context = f"""
        Conversation so far:
        {history_text}

        Relevant sources (use them to guide your answer, but cite only the ones in citation groups):
        {"\n\n".join(context_blocks)}
        
        Do not cite if the author is "Unknown Author(s)".
        CITATION GUIDELINES:        
        - HIGH CONFIDENCE sources: Use direct citations "(Author, Year)"
        - MEDIUM CONFIDENCE sources: Use "According to Author (Year)..." 
        - LOW CONFIDENCE sources: Use "(see Author, Year)"
        
        High confidence sources:
        {"\n\n".join(citation_groups['high_confidence']) or "None"}
        
        Medium confidence sources:  
        {"\n\n".join(citation_groups['medium_confidence']) or "None"}
        
        Low confidence sources:
        {"\n\n".join(citation_groups['low_confidence']) or "None"}

        """

        # Step 3: Construct prompt
        prompt = f"""
        You are a bioethics expert assistant. 
        Answer the user's question using the context provided below. 
        Draw justified connections between concepts even if not explicitly stated.
        If you need to make reasonable inferences based on the context, do so.
        If the context doesn't contain enough information, say what you do know from the context and indicate what information is missing.
        If the question doesn't concern neither bioethics nor previous questions, inform the user about it and don't answer it.
        Context:
        {context}

        Question: {question}
        Answer:
        """

        self.stream_handler.current_text = ""

        _ = self.llm.invoke(prompt)  # streaming happens here
        print()  # newline after streaming

        answer = self.stream_handler.get_text()
        self.history.append((question, answer))

        return answer



