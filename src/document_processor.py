import fitz
import re
from typing import List, Dict
from pathlib import Path
import logging
import PyPDF2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text_from_pdf(self,pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            doc = fitz.open(pdf_path)
            text = ""

            for page in doc:
                text += page.get_text()
                text += f"\n--- Page {page.number + 1} ---\n"  # page.number is 0-indexed

            logger.info(f"Extracted text from {pdf_path}: {len(text)} characters, {len(doc)} pages")
            doc.close()
            return text

        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""

    def clean_text(self,text: str) -> str:
        """Clean text from PDF"""
        text = re.sub(r'\n{2,}', '\n', text)  # keep single newlines
        text = re.sub(r'[ \t]+', ' ', text)  # collapse spaces/tabs

        # Remove page headers/footers
        text = re.sub(r'Page \d+.*?\n', '', text)

        # Remove references to figures/tables
        text = re.sub(r'\[Figure \d+\]|\[Table \d+\]', '', text)

        return text.strip()

    def chunk_text(self,text: str, metadata: Dict = None) -> List[Dict]:
        """Split text into chunks with metadata"""
        if not text:
            return []

        sentences = text.split('. ')
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size:
                if current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "metadata": metadata or {},
                        "chunk_id": len(chunks)
                    })

                    # Start new chunk with overlap
                    overlap_text = current_chunk[-self.chunk_overlap:] if len(
                        current_chunk) > self.chunk_overlap else current_chunk
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += ". " + sentence if current_chunk else sentence

            # Add final chunk
        if current_chunk:
            chunks.append({
                "text": current_chunk.strip(),
                "metadata": metadata or {},
                "chunk_id": len(chunks)
            })

        logger.info(f"Created {len(chunks)} chunks")
        return chunks

    def extract_metadata(self, pdf_path: str) -> dict:
        """Extract metadata (title, authors, year, filename, file_size) from a PDF."""

        metadata = {
            "filename": Path(pdf_path).name,
            "file_size": Path(pdf_path).stat().st_size,
            "title": None,
            "authors": None,
            "year": None
        }

        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)

            # 1. Try embedded PDF metadata
            pdf_meta = reader.metadata
            if pdf_meta:
                title = pdf_meta.get("/Title", "").strip()
                author = pdf_meta.get("/Author", "").strip()

                if title and title.lower() not in ["", "untitled", "unknown"]:
                    metadata["title"] = title

                if author and author.lower() not in ["", "anonymous", "unknown"]:
                    metadata["authors"] = author

            # 2. Fallback: look at first page
            if not metadata["title"] or not metadata["authors"]:
                try:
                    first_page = reader.pages[0].extract_text() or ""
                    lines = [line.strip() for line in first_page.split("\n") if line.strip()]

                    # crude heuristic: first line = title
                    if not metadata["title"] and lines:
                        metadata["title"] = lines[0]

                    # crude heuristic: authors in line(s) after title
                    if not metadata["authors"] and len(lines) > 1:
                        possible_authors = lines[1]
                        if re.search(r"[A-Z][a-z]+(?: [A-Z][a-z]+)*", possible_authors):
                            metadata["authors"] = possible_authors

                    # crude heuristic: find year (e.g., 2023, 2024)
                    year_match = re.search(r"\b(19|20)\d{2}\b", first_page)
                    if year_match:
                        metadata["year"] = year_match.group(0)

                except Exception:
                    pass

        # Defaults if missing
        metadata["title"] = metadata["title"] or "Unknown Title"
        metadata["authors"] = metadata["authors"] if metadata["authors"] else None
        metadata["year"] = metadata["year"] or "n.d."

        return metadata

    def process_document(self,pdf_path: str) -> List[Dict]:
        """Complete document processing"""
        try:
            file_path = Path(pdf_path)

        except TypeError as e:  # Catches specifically if pdf_path is the wrong type
            logger.error(f"Invalid path type: {pdf_path}: {e}")
            raise
        except OSError as e:  # Catches other filesystem-related errors
            logger.error(f"OS error with path: {pdf_path}: {e}")
            raise

        metadata=self.extract_metadata(pdf_path)

        raw_text = self.extract_text_from_pdf(pdf_path)
        clean_text = self.clean_text(raw_text)
        chunks = self.chunk_text(clean_text, metadata)
        logger.info(f"Processed {pdf_path}: {len(chunks)} chunks created")
        return chunks

    def process_documents(self, pdf_paths: List[str]) -> List[Dict]:
        documents = []
        for path in pdf_paths:
            documents.extend(self.process_document(path))
        return documents