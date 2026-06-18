# 📚 RAG PDF Chatbot

> **Chat with your PDF documents using Google Gemini AI — powered by LangChain, FAISS & HuggingFace.**

[![🚀 Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://rag-pdf-chatbot-ni3cbeypwvwx5wrf5xut5q.streamlit.app/)

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square&logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-green?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 🚀 What This Project Does

This project provides **two AI-powered applications**:

| App | File | Description |
|-----|------|-------------|
| 📄 PDF Chatbot | `app.py` | Upload multiple PDFs and ask questions — Gemini answers from document context |
| 🧠 Wikipedia RAG | `rag_pipeline.py` | Enter any topic, fetch Wikipedia knowledge, and ask questions using a local QA model |

---

## 🏗️ Architecture

```
User Question
      │
      ▼
  PDF Upload  ──► Text Extraction (PyPDF2)
                          │
                          ▼
               Text Chunking (LangChain)
                          │
                          ▼
         HuggingFace Embeddings (all-MiniLM-L6-v2)
                          │
                          ▼
              FAISS Vector Store (local)
                          │
                  Similarity Search
                          │
                          ▼
             Google Gemini 2.0 Flash (LLM)
                          │
                          ▼
                     Answer to User
```

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM**: Google Gemini 2.0 Flash (via LangChain)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace)
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **PDF Parsing**: PyPDF2
- **Wikipedia RAG**: `deepset/roberta-base-squad2` + FAISS

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/rag-pdf-chatbot.git
cd rag-pdf-chatbot
```

### 2. Create a virtual environment
```bash
python -m venv myenv

# Windows
myenv\Scripts\activate

# macOS / Linux
source myenv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Get a Google Gemini API Key
- Visit [Google AI Studio](https://ai.google.dev/)
- Create a free API key
- You'll enter it in the app sidebar (no `.env` file needed)

---

## ▶️ Running the Apps

### PDF Chatbot
```bash
streamlit run app.py
```

### Wikipedia RAG Pipeline
```bash
streamlit run rag_pipeline.py
```

---

## 📸 Features

- ✅ Upload **multiple PDFs** at once
- ✅ Intelligent chunking with overlap for better context
- ✅ **Local embeddings** — no extra API cost for vectorisation
- ✅ Powered by **Gemini 2.0 Flash** for fast, accurate answers
- ✅ Chat history displayed in the UI
- ✅ **Download chat history** as a CSV file
- ✅ Wikipedia-based RAG with confidence scores
- ✅ Clean, responsive Streamlit UI

---

## 📁 Project Structure

```
rag-pdf-chatbot/
├── app.py                  # Main PDF chatbot app
├── rag_pipeline.py         # Wikipedia RAG pipeline
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project metadata
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

---

## 💡 How It Works (Simple Explanation)

1. **You upload a PDF** → the app reads all the text from it
2. **Text is split** into small overlapping chunks (so no context is lost)
3. **Each chunk is converted** into a numerical vector (embedding)
4. **Vectors are stored** in a FAISS index (a fast similarity search database)
5. **When you ask a question**, your question is also converted to a vector
6. **FAISS finds** the most similar chunks to your question
7. **Those chunks + your question** are sent to Gemini AI
8. **Gemini generates** a precise answer based only on your document

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 📄 License

This project is licensed under the MIT License.

---

*Built with ❤️ using LangChain, Streamlit, and Google Gemini*
