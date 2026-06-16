# Flex Policies RAG Chatbot

An intelligent, interactive chat application built using a Retrieval-Augmented Generation (RAG) pipeline to query and answer questions about internal corporate policies (e.g., Anti-Corruption, Human Rights, Global Tax Strategy, and Supplier Codes of Conduct).

The application is styled to match the corporate branding theme of **flex.com** (Flex Blue & Green) and features a ChatGPT-like interface with source attribution.

---

## Key Features

*   **Corporate Branding**: Styled in a clean, professional light-theme matching `flex.com` with a stylized Flex logo representation.
*   **Knowledge Base Overview**: Real-time stats on the sidebar showing the active vector database collection name, total indexed chunks, and unique policy filenames.
*   **Source Attribution**: A toggleable accordion under each response reveals the exact document names, page numbers, and similarity scores of the chunks used to synthesize the answer.
*   **Preset Prompt Chips**: Clickable cards representing frequently asked policy questions to test the pipeline immediately.
*   **Developer-Free Chat**: The developer-oriented LLM parameters (model names, temperature, similarity thresholds) are hidden from the sidebar but persist as default settings in the background for a streamlined user experience.
*   **Fast API Backend**: Handled using FastAPI with automatic watch-reloading and static file serving.

---

## Project Structure

```text
├── data/
│   └── vector_store/           # Persistent ChromaDB collection files
├── rag_pipeline/               # Core modular RAG package
│   ├── __init__.py             # Exports core classes and methods
│   ├── embeddings.py           # EmbeddingManager (BAAI/bge-small-en-v1.5)
│   ├── vector_store.py         # VectorStore wrapper connecting to ChromaDB
│   ├── retriever.py            # RAGRetriever wrapper for similarity search
│   └── llm.py                  # get_llm (ChatGroq) and RAG response generator
├── static/                     # Web frontend files
│   ├── index.html              # Main chat layout
│   ├── style.css               # Corporate branding light CSS
│   └── app.js                  # Javascript chat logic & sources formatter
├── main.py                     # Entry point (FastAPI server + Uvicorn)
├── test_pipeline.py            # Command line test script to check the RAG flow
├── requirements.txt            # Python package dependencies
├── pyproject.toml              # Project dependencies configuration
└── .env                        # Environment credentials (API Keys)
```

---

## RAG Architecture

The pipeline follows these steps to answer questions:
1.  **Status Check**: On page load, the frontend hits `/api/status` to list current collection counts and unique source files in Chroma.
2.  **User Question**: The user types a query (e.g., *"What is the limit of working hours?"*) and presses Enter.
3.  **Embeddings**: The `EmbeddingManager` uses the local SentenceTransformer model `BAAI/bge-small-en-v1.5` to generate a 384-dimensional query embedding vector.
4.  **Retrieval**: `RAGRetriever` performs a cosine-similarity search against the persistent ChromaDB collection. It fetches the top $K$ relevant document snippets (default: 3).
5.  **Synthesis**: The retrieved chunks are formatted into a prompt template and passed to `ChatGroq` using the `llama-3.1-8b-instant` model.
6.  **Response**: The REST endpoint `/api/chat` returns a JSON object containing the markdown-generated answer and the array of retrieved sources (including filename, page number, similarity score, and excerpt).

---

## Setup & Running Guide

### 1. Prerequisites
- Python 3.10+
- An API Key from [Groq Console](https://console.groq.com/)

### 2. Install Dependencies
You can install the required packages using standard `pip`:
```bash
pip install -r requirements.txt
```
Or, if you use the modern `uv` tool:
```bash
uv sync
```

### 3. Environment Configuration
Verify that you have a `.env` file in the root of the project containing your Groq API key:
```env
GROQ_API_KEY="your-groq-api-key-here"
```

### 4. Running a Test Check (CLI)
To check if the python environment is configured correctly and your API key works:
```bash
python test_pipeline.py
```
This should output the initialized collection statistics and answer a default query via your terminal.

### 5. Launching the Web App
Run the main FastAPI server:
```bash
python main.py
```
Uvicorn will start and listen on port 8000:
`INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)`

Open **[http://localhost:8000](http://localhost:8000)** in your browser to start chatting with your corporate documents!
