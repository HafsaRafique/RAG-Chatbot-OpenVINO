#  RAG-Chatbot-OpenVINO

A lightweight, built for speed **Retrieval-Augmented Generation (RAG) chatbot** that lets you upload a PDF and ask natural-language questions about its contents with inference accelerated by **Intel OpenVINO**.

Built with Flask, FAISS, Sentence-Transformers, and `optimum-intel`, the app retrieves the most relevant chunks of your document and grounds the LLM's answer strictly in that context, refusing to answer when the document doesn't contain the information. (During the initial stages of chatbot development boom in early 2024)

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-web%20app-000000?logo=flask&logoColor=white)
![OpenVINO](https://img.shields.io/badge/OpenVINO-inference-00A3E0?logo=intel&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-vector%20search-4B8BBE)
![License](https://img.shields.io/badge/license-MIT-green)

---

##  Features

- ** PDF upload & parsing**,  extracts text from any PDF via `PyPDF2`
- ** Semantic chunking**, splits document text into overlapping, sentence-aware chunks (NLTK) instead of naive fixed-length splitting
- ** Vector search**,  embeds chunks with `sentence-transformers/all-MiniLM-L6-v2` and indexes them with `faiss.IndexFlatL2` for fast nearest-neighbor retrieval
- ** Grounded answers**,  computes cosine similarity between the query and retrieved chunks; if nothing is similar enough, the bot honestly replies that the answer isn't in the document instead of hallucinating
- ** OpenVINO-accelerated inference**,  the LLM runs through `optimum-intel`'s `OVModelForCausalLM`, taking advantage of Intel hardware acceleration instead of a plain PyTorch/Transformers pipeline
- ** Simple web UI**,  a minimal Flask front end for uploading a PDF and chatting with it in the browser
- ** Session reset**,  a `/restart` endpoint to clear the in-memory index and start over with a new document

---

##  How it works

```
 PDF Upload
     │
     ▼
 Text Extraction (PyPDF2)
     │
     ▼
 Semantic Chunking (NLTK sentence tokenizer, chunk_size=1000, overlap=200)
     │
     ▼
 Embedding (all-MiniLM-L6-v2) ──► FAISS Index (IndexFlatL2)
     │
     ▼
 User Question
     │
     ▼
 Top-k Retrieval + Cosine Similarity Check
     │
     ▼
 Prompt Construction (context + question)
     │
     ▼
 LLM Generation (OpenVINO IR model via optimum-intel)
     │
     ▼
 Answer
```

---

##  Tech Stack

| Layer            | Tool / Library                          |
|-------------------|------------------------------------------|
| Web framework     | Flask                                     |
| PDF parsing       | PyPDF2                                    |
| Chunking          | NLTK (`punkt`)                            |
| Embeddings        | Sentence-Transformers (`all-MiniLM-L6-v2`)|
| Vector store       | FAISS (`IndexFlatL2`)                     |
| Similarity scoring | scikit-learn (cosine similarity)          |
| LLM inference     | Intel OpenVINO via `optimum-intel`        |
| Tokenization      | Hugging Face `transformers`               |

---

##  Project Structure

```
RAG-Chatbot-OpenVINO/
├── main.py              # Flask app, RAG pipeline, and API routes
├── chatbot_state.py      # In-memory state container (index, chunks, models)
├── templates/
│   └── index.html        # Front-end chat UI
├── requirements.txt
└── LICENSE
```

---

##  Getting Started

### Prerequisites

- Python 3.9+
- An LLM converted to **OpenVINO IR format** (e.g. via `optimum-cli export openvino`), or a model already compatible with `optimum-intel`
- An Intel CPU/GPU/NPU for OpenVINO acceleration (falls back to CPU if no discrete GPU is available)

### 1. Clone the repository

```bash
git clone https://github.com/HafsaRafique/RAG-Chatbot-OpenVINO.git
cd RAG-Chatbot-OpenVINO
```

### 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Point the app at your OpenVINO model

In `main.py`, set `model_id` to the local path (or model repo) of your OpenVINO IR model:

```python
model_id = "path/to/your/openvino-model"
```

>  You can convert most Hugging Face causal LM checkpoints to OpenVINO IR format using [`optimum-intel`](https://github.com/huggingface/optimum-intel):
> ```bash
> optimum-cli export openvino --model <hf-model-id> <output-dir>
> ```

### 4. Run the app

```bash
python main.py
```

Then open **http://127.0.0.1:5000** in your browser, upload a PDF, and start asking questions.

---

##  API Reference

| Endpoint         | Method | Description                                      |
|-------------------|--------|---------------------------------------------------|
| `/`               | GET    | Serves the chat UI                                 |
| `/upload_pdf`     | POST   | Accepts a `.pdf` file, builds the embedding index  |
| `/ask_question`   | POST   | Accepts `{ "question": "..." }`, returns `{ "answer": "..." }` |
| `/restart`        | POST   | Clears the current session/index                   |

**Example — asking a question:**

```bash
curl -X POST http://127.0.0.1:5000/ask_question \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main conclusion of this document?"}'
```

