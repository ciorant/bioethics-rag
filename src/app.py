# app.py
import streamlit as st
from chatbot import BioethicsChatbot
import time
import io
import sys
from contextlib import redirect_stdout, redirect_stderr

st.set_page_config(
    page_title="Bioethics AI Assistant",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Bioethics AI Assistant")
st.markdown("*Ask questions about medical ethics, informed consent, research ethics, and more*")

# Custom CSS to hide debug output
st.markdown("""
<style>
    .debug-output {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
        font-family: monospace;
        font-size: 12px;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar info
with st.sidebar:
    st.markdown("### About")
    st.write("This demo uses Retrieval-Augmented Generation (RAG) with open-access bioethics papers.")

    st.markdown("### Sample Questions")
    sample_questions = [
        "What is informed consent in medical research?",
        "What are the ethical issues with genetic testing?",
        "How should AI bias in healthcare be addressed?",
        "What is the principle of beneficence?",
        "What are the ethics of end-of-life care?"
    ]

    for q in sample_questions:
        if st.button(q, key=q, use_container_width=True):
            st.session_state.current_question = q

    st.markdown("---")
    st.markdown("### Demo Info")
    st.info("💡 This demo shows sources found and similarity scores for transparency")

# Rate limiting
if 'query_count' not in st.session_state:
    st.session_state.query_count = 0


# Initialize chatbot (only once)
@st.cache_resource
def load_chatbot():
    """Load chatbot once and cache it"""
    return BioethicsChatbot("data/")


# Main interface
col1, col2 = st.columns([4, 1])

with col1:
    question = st.text_input(
        "Your question:",
        value=st.session_state.get('current_question', ''),
        placeholder="e.g., What are the ethical considerations in clinical trials?",
        key="question_input"
    )

with col2:
    st.metric("Queries Used", f"{st.session_state.query_count}/50")

# Clear the current_question after it's been used
if 'current_question' in st.session_state:
    del st.session_state.current_question

if question and st.session_state.query_count < 50:

    # Load chatbot
    try:
        if 'bot' not in st.session_state:
            with st.spinner("🔄 Loading bioethics knowledge base..."):
                st.session_state.bot = load_chatbot()
                st.success("✅ Knowledge base loaded!")

        st.session_state.query_count += 1

        # Create columns for response
        response_col, debug_col = st.columns([2, 1])

        with response_col:
            st.markdown("### 🤖 Assistant Response")

            # Capture the streaming output and debug info
            start_time = time.time()

            # Capture stdout to get debug prints
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()

            try:
                # Redirect prints to capture debug info
                sys.stdout = stdout_capture
                sys.stderr = stderr_capture

                # Get the answer (this will stream to captured stdout)
                answer = st.session_state.bot.ask(question)

            finally:
                # Restore stdout/stderr
                sys.stdout = old_stdout
                sys.stderr = old_stderr

            response_time = time.time() - start_time

            # Display the final answer
            st.write(answer)

        with debug_col:
            st.markdown("### 🔍 Debug Info")

            # Show captured debug output
            debug_output = stdout_capture.getvalue()
            if debug_output:
                with st.expander("📊 Search Results", expanded=True):
                    st.markdown(f'<div class="debug-output">{debug_output}</div>',
                                unsafe_allow_html=True)

            # Show response metadata
            st.metric("Response Time", f"{response_time:.2f}s")
            st.metric("Model", "GPT-4o-mini")

            # Show conversation history count
            if hasattr(st.session_state.bot, 'history'):
                st.metric("Conversation Turn", len(st.session_state.bot.history))

        # Show source information
        with st.expander("📚 About the Sources"):
            st.markdown("""
            This assistant searches through open-access bioethics papers to find relevant information.

            **Search Process:**
            1. Your question is converted to embeddings
            2. Similar text chunks are found using FAISS vector search
            3. Only chunks with similarity score ≥ 0.65 are used for citations
            4. The language model synthesizes an answer from these sources
            """)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("Please try refreshing the page or try a different question.")

elif st.session_state.query_count >= 50:
    st.error("📈 Demo limit reached for today. This prevents API abuse.")
    st.info("💡 For unlimited use, clone the repository and use your own API key.")

    with st.expander("🚀 How to run locally"):
        st.code("""
# Clone the repository
git clone your-repo-url
cd bioethics-chatbot

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="your-key-here"

# Run locally
streamlit run app.py
        """, language="bash")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🔗 Links**")
    st.markdown("- [GitHub Repository](your-repo-link)")
    st.markdown("- [Open Source Papers Used](./data/LICENSE_INFO.md)")

with col2:
    st.markdown("**🛠️ Tech Stack**")
    st.markdown("- Python & Streamlit")
    st.markdown("- OpenAI GPT-4o-mini")
    st.markdown("- FAISS Vector Search")
    st.markdown("- LangChain")

with col3:
    st.markdown("**📊 Demo Stats**")
    if 'bot' in st.session_state and hasattr(st.session_state.bot, 'vector_store'):
        doc_count = len(st.session_state.bot.vector_store.documents)
        st.markdown(f"- {doc_count} text chunks indexed")
        st.markdown(f"- Vector dimension: {st.session_state.bot.vector_store.dimension}")
    st.markdown(f"- Queries today: {st.session_state.query_count}")

# Add some spacing
st.markdown("<br>", unsafe_allow_html=True)