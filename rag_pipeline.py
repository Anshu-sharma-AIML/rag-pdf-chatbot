"""
rag_pipeline.py
───────────────
Wikipedia-based RAG (Retrieval-Augmented Generation) pipeline.

Users enter any topic → the app fetches the Wikipedia article,
splits it into chunks, embeds them with SentenceTransformers,
stores them in a FAISS index, and answers questions with
a RoBERTa-based extractive QA model.
"""

import numpy as np
import faiss
import wikipedia
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# ── Page config (must be the very first Streamlit call) ──────────────────────
st.set_page_config(page_title="Wikipedia RAG Q&A", page_icon="🧠")


# ── Model loading (cached so they load only once) ────────────────────────────

@st.cache_resource
def load_embedding_model():
    """Load SentenceTransformer for creating dense embeddings."""
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


@st.cache_resource
def load_qa_pipeline():
    """Load an extractive QA pipeline (RoBERTa fine-tuned on SQuAD 2)."""
    model_name = "deepset/roberta-base-squad2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return pipeline("question-answering", model=model, tokenizer=tokenizer)


embedding_model = load_embedding_model()
qa_pipe = load_qa_pipeline()


# ── Helper functions ─────────────────────────────────────────────────────────

def fetch_wikipedia(topic: str) -> str | None:
    """Fetch the full text of a Wikipedia article for the given topic."""
    try:
        page = wikipedia.page(topic)
        return page.content
    except wikipedia.exceptions.PageError:
        st.warning(f"No Wikipedia page found for **{topic}**. Try a different topic.")
        return None
    except wikipedia.exceptions.DisambiguationError as e:
        st.warning(
            f"**{topic}** is ambiguous. Please be more specific.\n\n"
            f"Suggestions: {', '.join(e.options[:6])}"
        )
        return None


def split_into_chunks(text: str, tokenizer, chunk_size: int = 256, overlap: int = 20) -> list[str]:
    """
    Tokenise the text and split into overlapping chunks.
    Overlap helps preserve context across chunk boundaries.
    """
    tokens = tokenizer.tokenize(text)
    chunks, start = [], 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(tokenizer.convert_tokens_to_string(tokens[start:end]))
        if end == len(tokens):
            break
        start = end - overlap
    return chunks


def build_faiss_index(chunks: list[str]):
    """Embed all chunks and store them in a FAISS L2 index."""
    embeddings = embedding_model.encode(chunks, show_progress_bar=False)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, embeddings


def retrieve_top_k(query: str, index, chunks: list[str], k: int = 3) -> list[str]:
    """Return the top-k most relevant chunks for a query."""
    query_vec = embedding_model.encode([query])
    _, indices = index.search(np.array(query_vec), k)
    return [chunks[i] for i in indices[0]]


# ── Streamlit UI ─────────────────────────────────────────────────────────────

def main():
    st.title("🧠 Wikipedia RAG Q&A")
    st.write(
        "Enter any topic and ask questions. "
        "The app fetches knowledge from Wikipedia in real time and answers using a local QA model."
    )

    topic = st.text_input("🔍 Topic to retrieve from Wikipedia:", placeholder="e.g. Artificial Intelligence")

    if not topic:
        st.info("Enter a topic above to get started.")
        return

    with st.spinner(f"Fetching Wikipedia article for **{topic}**…"):
        document = fetch_wikipedia(topic)

    if not document:
        return

    # Build index once per topic (cache in session state)
    cache_key = f"index_{topic}"
    if cache_key not in st.session_state:
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        with st.spinner("Splitting text and building FAISS index…"):
            chunks = split_into_chunks(document, tokenizer)
            index, _ = build_faiss_index(chunks)
        st.session_state[cache_key] = (chunks, index)
        st.success(f"✅ Article split into **{len(chunks)} chunks** and indexed.")
    else:
        chunks, index = st.session_state[cache_key]
        st.success(f"✅ Using cached index — **{len(chunks)} chunks** available.")

    query = st.text_input("💬 Ask a question about this topic:", placeholder="e.g. Who invented it?")

    if not query:
        return

    # Retrieve relevant chunks
    top_chunks = retrieve_top_k(query, index, chunks)

    with st.expander("📄 Retrieved Chunks (used as context)"):
        for i, chunk in enumerate(top_chunks, 1):
            st.markdown(f"**Chunk {i}:** {chunk[:300]}…")

    # Generate answer
    with st.spinner("Generating answer…"):
        context = " ".join(top_chunks)
        result = qa_pipe(question=query, context=context)

    st.subheader("✅ Answer")
    st.markdown(f"### {result['answer']}")
    st.caption(f"Confidence score: `{result['score']:.2%}`")


if __name__ == "__main__":
    main()
