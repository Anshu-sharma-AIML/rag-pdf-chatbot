import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64
import os
import asyncio
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime

# Fix for asyncio event loop conflict in some environments
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


# ─────────────────────────────────────────────
# Core RAG Functions
# ─────────────────────────────────────────────

def get_pdf_text(pdf_docs):
    """Extract raw text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text


def get_text_chunks(text):
    """Split text into overlapping chunks for better context retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    """Create FAISS vector store using HuggingFace embeddings and save locally."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store


def get_conversational_chain(api_key):
    """Build a LangChain QA chain using Gemini 2.0 Flash."""
    prompt_template = """Answer the question as detailed as possible from the provided context.
If the answer is not in the provided context, say: 'The answer is not available in the uploaded documents.'
Do NOT make up an answer.

Context:
{context}

Question:
{question}

Answer:"""
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        google_api_key=api_key
    )
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | model | StrOutputParser()
    return chain


# ─────────────────────────────────────────────
# Chat UI Helper
# ─────────────────────────────────────────────

CHAT_CSS = """
<style>
    .chat-message {
        padding: 1.5rem; border-radius: 0.5rem;
        margin-bottom: 1rem; display: flex;
    }
    .chat-message.user { background-color: #2b313e; }
    .chat-message.bot  { background-color: #475063; }
    .chat-message .avatar { width: 20%; }
    .chat-message .avatar img {
        max-width: 78px; max-height: 78px;
        border-radius: 50%; object-fit: cover;
    }
    .chat-message .message {
        width: 80%; padding: 0 1.5rem; color: #fff;
    }
</style>
"""

USER_AVATAR = "https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png"
BOT_AVATAR  = "https://i.ibb.co/wNmYHsx/langchain-logo.webp"


def render_message(question, answer):
    """Render a single Q&A pair as styled chat bubbles."""
    st.markdown(
        f"""
        {CHAT_CSS}
        <div class="chat-message user">
            <div class="avatar"><img src="{USER_AVATAR}"></div>
            <div class="message">{question}</div>
        </div>
        <div class="chat-message bot">
            <div class="avatar"><img src="{BOT_AVATAR}"></div>
            <div class="message">{answer}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# Main Handler
# ─────────────────────────────────────────────

def handle_user_input(user_question, api_key, pdf_docs):
    """Process user question against uploaded PDFs and display the response."""
    if not api_key or not pdf_docs:
        st.warning("Please upload PDF files and provide your Google API Key.")
        return

    # Build / refresh the vector store on every query
    raw_text = get_pdf_text(pdf_docs)
    chunks = get_text_chunks(raw_text)
    get_vector_store(chunks)

    # Retrieve relevant chunks
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    docs = db.similarity_search(user_question)

    # Get answer from Gemini
    chain = get_conversational_chain(api_key)
    context_text = "\n\n".join([doc.page_content for doc in docs])
    answer = chain.invoke({"context": context_text, "question": user_question})

    # Save to conversation history
    pdf_names = ", ".join([pdf.name for pdf in pdf_docs])
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.conversation_history.append(
        (user_question, answer, timestamp, pdf_names)
    )

    # Render latest message first
    render_message(user_question, answer)

    # Render older messages below
    history = st.session_state.conversation_history[:-1]
    for q, a, _, _ in reversed(history):
        render_message(q, a)

    # Download history as CSV
    if st.session_state.conversation_history:
        df = pd.DataFrame(
            st.session_state.conversation_history,
            columns=["Question", "Answer", "Timestamp", "PDF Name"],
        )
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = (
            f'<a href="data:file/csv;base64,{b64}" download="chat_history.csv">'
            f"<button>⬇️ Download Chat History</button></a>"
        )
        st.sidebar.markdown(href, unsafe_allow_html=True)

    st.snow()


# ─────────────────────────────────────────────
# Streamlit App Entry Point
# ─────────────────────────────────────────────

def main():
    st.set_page_config(page_title="PDF ChatBot", page_icon="📚")
    st.header("📚 Chat with your PDF Documents")
    st.caption("Powered by Google Gemini · LangChain · FAISS · HuggingFace")

    # Initialise session state
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    # ── Sidebar ──
    with st.sidebar:
        st.title("⚙️ Settings")

        # ── Social links ──
        YOUR_LINKEDIN = "https://www.linkedin.com/in/anshu-sharma13/"
        YOUR_GITHUB   = "https://github.com/Anshu-sharma-AIML"
        YOUR_EMAIL    = "Anshusharma6117@gmail.com"

        st.markdown(
            f"[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)]({YOUR_LINKEDIN}) "
            f"[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)]({YOUR_GITHUB}) "
            f"[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:{YOUR_EMAIL})"
        )

        st.divider()

        # API Key
        api_key = st.text_input("🔑 Google Gemini API Key", type="password")
        st.markdown("Get your key → [Google AI Studio](https://ai.google.dev/)")
        if not api_key:
            st.warning("Enter your API key to get started.")

        st.divider()

        # PDF Upload
        pdf_docs = st.file_uploader(
            "📄 Upload PDF Files",
            accept_multiple_files=True,
            type=["pdf"],
        )
        if st.button("✅ Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing PDFs…"):
                    raw_text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(raw_text)
                    get_vector_store(chunks)
                st.success(f"Processed {len(pdf_docs)} file(s) — {len(chunks)} chunks created!")
            else:
                st.warning("Please upload at least one PDF.")

        st.divider()

        # Reset
        if st.button("🔄 Reset Conversation"):
            st.session_state.conversation_history = []
            st.rerun()

    # ── Main chat input ──
    user_question = st.text_input("💬 Ask a question about your PDFs…")
    if user_question:
        handle_user_input(user_question, api_key, pdf_docs)


if __name__ == "__main__":
    main()
