# Technical Documentation — Flex Policies RAG Chatbot

> **Version:** 0.1.0  
> **Last Updated:** June 2026  
> **Python:** ≥ 3.14  
> **Package Manager:** [uv](https://docs.astral.sh/uv/)

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Project Structure](#2-project-structure)
3. [Data Flow — End-to-End](#3-data-flow--end-to-end)
4. [Backend Modules (`rag_pipeline/`)](#4-backend-modules-rag_pipeline)
   - 4.1 [`embeddings.py` — EmbeddingManager](#41-embeddingspy--embeddingmanager)
   - 4.2 [`vector_store.py` — VectorStore](#42-vector_storepy--vectorstore)
   - 4.3 [`retriever.py` — RAGRetriever](#43-retrieverpy--ragretriever)
   - 4.4 [`llm.py` — LLM + `rag_simple()`](#44-llmpy--llm--rag_simple)
   - 4.5 [`__init__.py` — Package Exports](#45-__init__py--package-exports)
5. [FastAPI Server (`main.py`)](#5-fastapi-server-mainpy)
   - 5.1 [Application Bootstrap](#51-application-bootstrap)
   - 5.2 [API Endpoints](#52-api-endpoints)
   - 5.3 [Request / Response Schemas](#53-request--response-schemas)
6. [Frontend (`static/`)](#6-frontend-static)
   - 6.1 [HTML Layout (`index.html`)](#61-html-layout-indexhtml)
   - 6.2 [CSS Design System (`style.css`)](#62-css-design-system-stylecss)
   - 6.3 [JavaScript Controller (`app.js`)](#63-javascript-controller-appjs)
7. [Data & Ingestion Pipeline](#7-data--ingestion-pipeline)
8. [Environment Variables & Secrets](#8-environment-variables--secrets)
9. [Dependency Stack](#9-dependency-stack)
10. [Development & Debugging Guide](#10-development--debugging-guide)
11. [Known Gotchas & Troubleshooting](#11-known-gotchas--troubleshooting)

---

## 1. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                           BROWSER (Client)                          │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────────────────┐ │
│  │ index.html │  │  style.css   │  │          app.js              │ │
│  │ (layout)   │  │ (flex theme) │  │ (chat logic, API calls,      │ │
│  │            │  │              │  │  markdown rendering)          │ │
│  └────────────┘  └──────────────┘  └──────────────────────────────┘ │
│                        │ fetch('/api/chat')                          │
│                        │ fetch('/api/status')                        │
└────────────────────────┼────────────────────────────────────────────┘
                         │  HTTP (JSON)
                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     FASTAPI SERVER (main.py)                        │
│                                                                      │
│  POST /api/chat    → ChatRequest → rag_simple() → ChatResponse      │
│  GET  /api/status  → VectorStore.get_collection_info()               │
│  GET  /            → Serves static/index.html                        │
│  /*   /static/*    → StaticFiles mount                               │
└─────────────┬────────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      rag_pipeline (Python Package)                   │
│                                                                      │
│  ┌─────────────────┐  ┌──────────────┐  ┌──────────────────────────┐│
│  │ EmbeddingManager│  │  VectorStore │  │     RAGRetriever         ││
│  │ (SentenceTransf)│  │  (ChromaDB)  │  │  (query → ranked docs)  ││
│  └────────┬────────┘  └──────┬───────┘  └────────────┬─────────────┘│
│           │                  │                       │              │
│           │         ┌────────┴────────┐              │              │
│           │         │  ChromaDB Disk  │              │              │
│           │         │  data/vector_   │◄─────────────┘              │
│           │         │  store/         │                              │
│           │         └─────────────────┘                              │
│           │                                                          │
│  ┌────────┴──────────────────────────────────────────────────┐      │
│  │  get_llm() → ChatGroq (Groq Cloud)                        │      │
│  │  rag_simple(query, retriever, llm) → { answer, sources }  │      │
│  └────────────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────────┘
```

**Summary:** The frontend sends JSON requests to FastAPI. The server uses the modular `rag_pipeline` package to embed the query, search the persistent ChromaDB vector store, retrieve top-K chunks, inject them into a prompt, and call the Groq-hosted LLM to synthesize a natural-language answer — all returned along with source attribution metadata.

---

## 2. Project Structure

```text
RAG/
├── .env                        # API keys (GROQ, OPENAI, GOOGLE, NVIDIA)
├── .gitignore                  # Ignores .env, .venv, vector_store, caches
├── .python-version             # Python version pin (3.14)
├── pyproject.toml              # uv/pip project metadata and dependencies
├── requirements.txt            # Flat pip-compatible dependency list
├── uv.lock                     # Deterministic lock file (uv tool)
│
├── main.py                     # FastAPI entry point — Uvicorn server
├── test_pipeline.py            # CLI smoke test for the RAG pipeline
│
├── rag_pipeline/               # Core modular Python package
│   ├── __init__.py             # Re-exports: EmbeddingManager, VectorStore,
│   │                           #   RAGRetriever, get_llm, rag_simple
│   ├── embeddings.py           # EmbeddingManager class
│   ├── vector_store.py         # VectorStore class (ChromaDB wrapper)
│   ├── retriever.py            # RAGRetriever class (query → ranked docs)
│   └── llm.py                  # get_llm(), rag_simple() function
│
├── static/                     # Frontend web assets served by FastAPI
│   ├── index.html              # Main HTML layout (header, sidebar, chat)
│   ├── style.css               # Full CSS design system (Flex brand theme)
│   ├── app.js                  # JavaScript chat controller
│   ├── logo.png                # Flex company logo image
│   └── assets/                 # Additional static assets (images, etc.)
│
├── data/
│   ├── pdf_files/              # Source PDF policy documents (7 files)
│   │   ├── anti-corruption-policy.pdf
│   │   ├── EHS_Policy_Rev_I.pdf
│   │   ├── flex-culture-statement.pdf
│   │   ├── flex-foundation-india-csr.pdf
│   │   ├── Global_Tax_Strategy_Flex.pdf
│   │   ├── human-rights-policy.pdf
│   │   └── Supplier-Code-of-Conduct-English.pdf
│   ├── text_files/             # (Reserved for plaintext document ingestion)
│   └── vector_store/           # ChromaDB persistent storage (.gitignored)
│
└── notebook/
    └── document.ipynb          # Original Jupyter notebook (exploration/ingestion)
```

---

## 3. Data Flow — End-to-End

Below is the complete lifecycle of a single user query:

### 3.1 Page Load

```
Browser                    Server
   │  GET /                   │
   │ ◄────────────────────── HTML (index.html)
   │  GET /static/style.css   │
   │  GET /static/app.js      │
   │  GET /api/status          │
   │ ◄────────────────────── { status, document_count, sources[], embedding_model }
   │                           │
   │  (Sidebar populated       │
   │   with KB stats)          │
```

### 3.2 Query Processing

```
Step  Component           Action
────  ──────────────────  ───────────────────────────────────────────────
  1   Browser (app.js)    User types query → POST /api/chat with JSON body
  2   FastAPI (main.py)   Deserializes ChatRequest, calls get_llm() and rag_simple()
  3   EmbeddingManager    Encodes query string → 384-dim float vector
  4   RAGRetriever        Passes query vector to ChromaDB's .query() method
  5   ChromaDB            Performs cosine similarity search, returns top-K results
  6   RAGRetriever        Converts distances → similarity scores, filters by threshold
  7   rag_simple()        Joins retrieved chunk texts into context string
  8   ChatGroq (Groq)     Sends prompt (context + question) to Groq cloud API
  9   rag_simple()        Returns { answer: str, sources: List[Dict] }
 10   FastAPI             Serializes ChatResponse → JSON
 11   Browser (app.js)    Renders markdown answer + collapsible source cards
```

### 3.3 Similarity Score Calculation

ChromaDB returns raw **cosine distances** (lower is better). The retriever converts these:

```
similarity_score = 1.0 - distance
```

- **Score ≥ 0.6** → "High match" (green badge in UI)
- **Score < 0.6** → "Medium match" (orange badge in UI)
- Documents below `score_threshold` are filtered out entirely.

---

## 4. Backend Modules (`rag_pipeline/`)

### 4.1 `embeddings.py` — EmbeddingManager

| Property        | Details                                                |
|-----------------|--------------------------------------------------------|
| **Model**       | `BAAI/bge-small-en-v1.5` (HuggingFace, 33M params)    |
| **Dimension**   | 384-dimensional float vectors                          |
| **Library**     | `sentence-transformers`                                |
| **Loading**     | Eager-loaded in `__init__()` via `_load_model()`       |

**Key Method:**

```python
def generate_embeddings(self, texts: List[str]) -> np.ndarray:
    """
    Encodes a batch of text strings into embedding vectors.
    Progress bars are disabled for server-side usage.
    Returns: numpy array of shape (len(texts), 384)
    """
```

**Design Decision — Why `bge-small-en-v1.5`?**
- Lightweight (~130MB) — runs on CPU with low latency
- Strong retrieval performance for its size on MTEB benchmarks
- English-optimized — suitable for corporate policy documents

---

### 4.2 `vector_store.py` — VectorStore

| Property           | Details                                                  |
|--------------------|----------------------------------------------------------|
| **Database**       | ChromaDB (PersistentClient)                              |
| **Collection**     | `pdf_documents`                                          |
| **Persist Dir**    | `<project_root>/data/vector_store/`                      |
| **ID Format**      | `doc_<uuid8>_<index>` (e.g., `doc_a3f2b1c4_12`)         |

**Key Methods:**

| Method                | Description                                                           |
|-----------------------|-----------------------------------------------------------------------|
| `_initialize_store()` | Creates ChromaDB client, gets/creates collection, logs document count |
| `add_documents()`     | Batch-inserts LangChain `Document` objects with their embeddings      |
| `get_collection_info()` | Returns collection name, total count, and unique source filenames   |

**Metadata stored per chunk:**

```python
{
    "doc_index": int,           # Sequential index within the batch
    "content_length": int,      # Character count of chunk text
    "source": str,              # Original file path (from LangChain loader)
    "page": int,                # Page number (0-indexed, from PyMuPDF/PyPDF)
    # ... any additional metadata from the document loader
}
```

**Path Resolution:**
The default persist directory is computed relative to the `vector_store.py` file location, not the working directory:
```python
current_dir = os.path.dirname(os.path.abspath(__file__))  # rag_pipeline/
project_root = os.path.dirname(current_dir)               # RAG/
DEFAULT_PERSIST_DIR = os.path.join(project_root, "data", "vector_store")
```

---

### 4.3 `retriever.py` — RAGRetriever

**Constructor Dependencies:**
```python
RAGRetriever(vector_store: VectorStore, embedding_manager: EmbeddingManager)
```

**Key Method:**

```python
def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
```

**Return Schema (per retrieved document):**
```python
{
    "id": str,                  # ChromaDB document ID
    "content": str,             # Full text content of the chunk
    "metadata": dict,           # Original metadata (source, page, etc.)
    "similarity_score": float,  # 1.0 - cosine_distance
    "distance": float,          # Raw cosine distance from ChromaDB
    "rank": int                 # 1-indexed rank position
}
```

**Retrieval Process:**
1. Embed the query string using `EmbeddingManager`
2. Call `ChromaDB collection.query()` with the embedding vector
3. Convert distances to similarity scores
4. Filter by `score_threshold`
5. Return ranked list of document dictionaries

---

### 4.4 `llm.py` — LLM + `rag_simple()`

#### `get_llm()`

| Parameter     | Default                 | Description                              |
|---------------|-------------------------|------------------------------------------|
| `model`       | `llama-3.1-8b-instant`  | Groq-hosted model identifier             |
| `temperature` | `0.1`                   | Sampling temperature (lower = factual)   |

Returns a `ChatGroq` instance from `langchain_groq`. The API key is loaded from `.env` via `python-dotenv`.

**Supported Groq Models (available in dropdown):**
- `llama-3.1-8b-instant` (default — fastest)
- `llama-3.3-70b-versatile` (most capable)
- `llama3-8b-8192` (longer context)
- `gemma2-9b-it` (Google's Gemma)

#### `rag_simple()`

The core orchestration function that ties retrieval and generation together:

```python
def rag_simple(query, retriever, llm, top_k=3, score_threshold=0.0) -> Dict[str, Any]:
```

**Behavior:**
1. Calls `retriever.retrieve(query, top_k, score_threshold)`
2. Joins all retrieved chunk contents with `\n\n` separator
3. If no context is found, returns a static "no relevant context" message
4. Constructs a grounded prompt:
   ```
   Use the following context to answer the question.
   If the context does not contain relevant information,
   state that you cannot find the answer in the provided documents.
   Do not make up facts.

   Context:
   {retrieved chunks}

   Question: {user query}

   Answer:
   ```
5. Calls `llm.invoke(prompt)` → extracts `.content`
6. Returns `{ "answer": str, "sources": List[Dict] }`

---

### 4.5 `__init__.py` — Package Exports

Clean public API surface:

```python
from .embeddings import EmbeddingManager
from .vector_store import VectorStore
from .retriever import RAGRetriever
from .llm import get_llm, rag_simple

__all__ = ['EmbeddingManager', 'VectorStore', 'RAGRetriever', 'get_llm', 'rag_simple']
```

**Usage from any file in the project:**
```python
from rag_pipeline import EmbeddingManager, VectorStore, RAGRetriever, get_llm, rag_simple
```

---

## 5. FastAPI Server (`main.py`)

### 5.1 Application Bootstrap

On server startup (`python main.py`), the following happens **once** at module level:

```python
# These are initialized ONCE and shared across all requests
embedding_manager = EmbeddingManager()     # Loads SentenceTransformer model (~2-3s)
vector_store = VectorStore()               # Connects to ChromaDB persistent store
retriever = RAGRetriever(vector_store, embedding_manager)
```

The FastAPI app is created with CORS middleware allowing all origins (for local development):

```python
app = FastAPI(title="Flex RAG Chatbot Service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
```

Static files are mounted at `/static` and the root `/` serves `index.html`.

### 5.2 API Endpoints

| Method | Path           | Description                                    |
|--------|----------------|------------------------------------------------|
| `POST` | `/api/chat`    | Accepts a query, returns AI answer + sources   |
| `GET`  | `/api/status`  | Returns vector store stats (count, sources)    |
| `GET`  | `/`            | Serves the main `index.html` frontend          |
| `GET`  | `/static/*`    | Serves static assets (CSS, JS, images)         |

### 5.3 Request / Response Schemas

#### `POST /api/chat`

**Request Body (`ChatRequest`):**
```json
{
    "message": "What is the limit of working hours?",
    "temperature": 0.1,
    "top_k": 3,
    "score_threshold": 0.0,
    "model": "llama-3.1-8b-instant"
}
```

| Field             | Type   | Default                | Constraints      |
|-------------------|--------|------------------------|------------------|
| `message`         | string | (required)             | —                |
| `temperature`     | float  | `0.1`                  | 0.0 – 1.0       |
| `top_k`           | int    | `3`                    | 1 – 10           |
| `score_threshold` | float  | `0.0`                  | 0.0 – 1.0       |
| `model`           | string | `llama-3.1-8b-instant` | Valid Groq model |

**Response Body (`ChatResponse`):**
```json
{
    "answer": "According to the Supplier Code of Conduct...",
    "sources": [
        {
            "id": "doc_a3f2b1c4_5",
            "content": "Working hours shall not exceed 60 hours...",
            "metadata": {
                "source": "/path/to/Supplier-Code-of-Conduct-English.pdf",
                "page": 3,
                "doc_index": 5,
                "content_length": 412
            },
            "similarity_score": 0.7823,
            "distance": 0.2177,
            "rank": 1
        }
    ]
}
```

#### `GET /api/status`

**Response:**
```json
{
    "status": "ready",
    "collection_name": "pdf_documents",
    "document_count": 282,
    "sources": [
        "anti-corruption-policy.pdf",
        "EHS_Policy_Rev_I.pdf",
        "Supplier-Code-of-Conduct-English.pdf"
    ],
    "embedding_model": "BAAI/bge-small-en-v1.5"
}
```

---

## 6. Frontend (`static/`)

### 6.1 HTML Layout (`index.html`)

The page is structured into four main visual sections:

```
┌─────────────────────────────────────────────────────────────┐
│  SITE HEADER  (#009add blue)  — Flex logo, nav links        │
├──────────┬──────────────────────────────────────────────────┤
│ SIDEBAR  │            MAIN CHAT AREA                        │
│          │  ┌─────────────────────────────────────────────┐ │
│ • KB     │  │  Chat Header (title, model badge, refresh)  │ │
│   Stats  │  ├─────────────────────────────────────────────┤ │
│ • Config │  │  Chat Messages (scrollable)                 │ │
│   (hidden│  │   ├── Welcome Screen (FAQ chips)            │ │
│ • Actions│  │   ├── User messages                         │ │
│          │  │   └── Assistant messages (markdown + srcs)  │ │
│          │  ├─────────────────────────────────────────────┤ │
│          │  │  Input Area (textarea + send button)         │ │
│          │  └─────────────────────────────────────────────┘ │
├──────────┴──────────────────────────────────────────────────┤
│  SITE FOOTER  (#009add blue)  — © Flex, links              │
└─────────────────────────────────────────────────────────────┘
```

**External Dependencies (CDN):**
- **Google Fonts** — `Inter` (body) + `Outfit` (headings)
- **Font Awesome 6.4.0** — Icon set
- **Marked.js** — Markdown → HTML rendering for assistant responses

**Key Element IDs:**

| ID                   | Element           | Purpose                                    |
|----------------------|-------------------|--------------------------------------------|
| `chat-form`          | `<form>`          | Chat input form submission                 |
| `user-input`         | `<textarea>`      | User query input field                     |
| `send-btn`           | `<button>`        | Submit button (disabled when input empty)  |
| `chat-messages`      | `<div>`           | Scrollable message container               |
| `welcome-screen`     | `<div>`           | Initial welcome with FAQ suggestion chips  |
| `clear-chat`         | `<button>`        | Clears chat history, resets welcome screen |
| `refresh-db`         | `<button>`        | Re-fetches `/api/status`                   |
| `db-chunks`          | `<span>`          | Displays total indexed chunk count         |
| `db-files`           | `<ul>`            | Lists unique indexed policy filenames      |
| `model-select`       | `<select>`        | LLM model dropdown (hidden section)       |
| `temp-slider`        | `<input[range]>`  | Temperature control (hidden section)       |
| `topk-slider`        | `<input[range]>`  | Top-K control (hidden section)             |
| `threshold-slider`   | `<input[range]>`  | Score threshold control (hidden section)   |
| `active-model-badge` | `<span>`          | Shows currently selected model name        |

---

### 6.2 CSS Design System (`style.css`)

**Design Language:** Flex.com corporate light theme with `#009add` sky-blue brand accent.

#### CSS Custom Properties (Design Tokens)

```css
:root {
    /* Background Hierarchy */
    --bg-darkest:   #f3f4f6;    /* Page canvas */
    --bg-darker:    #ffffff;    /* Sidebar background */
    --bg-card:      #f9fafb;    /* Card/panel backgrounds */
    --border-color: #e5e7eb;    /* Divider lines */

    /* Text Hierarchy */
    --text-primary:   #111827;  /* Headings, primary content */
    --text-secondary: #4b5563;  /* Labels, secondary text */
    --text-muted:     #9ca3af;  /* Placeholder, disabled text */

    /* Brand Colors */
    --accent-color-one:          #009add;  /* Primary accent */
    --accent-color-two:          #009add;  /* Secondary accent */
    --footer-accent-text-color:  #009add;  /* Footer links */
    --footer-btn-fill-color:     #009add;  /* Footer button fill */
    --accent-neon-blue:          #00b0f0;  /* Gradient endpoint */

    /* Status Colors */
    --success-color: #2ed573;   /* High similarity scores */
    --warning-color: #ffa502;   /* Medium similarity scores */
    --error-color:   #ff4757;   /* Errors and failures */

    /* Typography */
    --font-heading: 'Outfit', sans-serif;
    --font-body:    'Inter', sans-serif;

    /* Shadows */
    --shadow-sm:  0 1px 2px rgba(0,0,0,0.05);
    --shadow-md:  0 4px 6px rgba(0,0,0,0.05), 0 2px 4px rgba(0,0,0,0.03);
    --shadow-lg:  0 10px 15px rgba(0,0,0,0.05), 0 4px 6px rgba(0,0,0,0.02);
}
```

#### Key CSS Components

| Component           | Selector                  | Notes                                           |
|---------------------|---------------------------|-------------------------------------------------|
| Site Header         | `.site-header`            | `#009add` blue bar, Flex logo inverted to white |
| Site Footer         | `.site-footer`            | `#009add` blue bar, legal links                 |
| Sidebar             | `.sidebar`                | White bg, flex-column layout, fixed width       |
| Chat Container      | `.chat-container`         | Flex-grow main content area                     |
| Message Bubbles     | `.message.user`           | Right-aligned, blue gradient background         |
|                     | `.message.assistant`      | Left-aligned, white background                  |
| Source Cards        | `.source-card`            | Collapsible accordion below assistant messages  |
| Welcome Screen      | `.welcome-screen`         | Centered with FAQ suggestion chips grid         |
| Suggestion Chips    | `.suggestion-chip`        | Clickable cards with hover micro-animations     |
| Typing Indicator    | `.typing-dot`             | Animated bounce dots (CSS keyframes)            |
| Custom Scrollbar    | `::-webkit-scrollbar`     | Thin 6px track with rounded thumb               |

---

### 6.3 JavaScript Controller (`app.js`)

**Entry Point:** `DOMContentLoaded` event listener wraps all initialization.

#### Initialization Flow

```
DOMContentLoaded
├── Cache all DOM element references
├── Attach event listeners
│   ├── Slider input → update display values
│   ├── Model select → update badge
│   ├── Textarea input → auto-resize + toggle send button
│   ├── Enter key (no Shift) → submit form
│   ├── Suggestion chips → inject text + submit
│   ├── Clear chat button → reset to welcome screen
│   └── Refresh DB button → re-fetch /api/status
└── loadDatabaseStatus()  ← Initial API call
```

#### Key Functions

| Function                   | Purpose                                                      |
|----------------------------|--------------------------------------------------------------|
| `loadDatabaseStatus()`     | `GET /api/status` → populates sidebar KB stats               |
| `submitMessage(queryText)` | Main flow: add user bubble → show typing → POST `/api/chat` → render response |
| `appendUserMessage(text)`  | Creates and appends a user message bubble (XSS-safe)         |
| `appendTypingIndicator()`  | Shows animated dots while waiting for API response           |
| `appendAssistantMessage()` | Renders markdown answer + collapsible source cards           |
| `appendErrorMessage(err)`  | Renders styled error bubble on API failure                   |
| `escapeHtml(text)`         | XSS protection — escapes `& < > " '` characters             |
| `toggleSources(button)`    | Toggles source card accordion visibility (global function)   |
| `scrollToBottom()`         | Smooth-scrolls chat container to latest message              |

#### Markdown Rendering

Assistant responses are parsed via `marked.parse(answer)` before injection. This supports:
- Headings, bold, italic
- Bullet and numbered lists
- Code blocks and inline code
- Tables

**XSS Note:** User input is always escaped via `escapeHtml()`. Assistant responses are rendered as HTML via `marked.js` — the LLM output is trusted because it originates from the server.

---

## 7. Data & Ingestion Pipeline

### 7.1 Source Documents

Seven PDF policy documents are stored in `data/pdf_files/`:

| Document                              | Size    | Topic                               |
|---------------------------------------|---------|--------------------------------------|
| `anti-corruption-policy.pdf`          | 128 KB  | Anti-corruption & bribery rules      |
| `EHS_Policy_Rev_I.pdf`               | 78 KB   | Environment, Health & Safety policy  |
| `flex-culture-statement.pdf`         | 42 KB   | Company culture and values           |
| `flex-foundation-india-csr.pdf`      | 84 KB   | India CSR foundation activities      |
| `Global_Tax_Strategy_Flex.pdf`       | 286 KB  | Global tax management strategy       |
| `human-rights-policy.pdf`            | 110 KB  | Human rights due diligence           |
| `Supplier-Code-of-Conduct-English.pdf` | 223 KB | Supplier standards & working hours   |

### 7.2 Ingestion Workflow (via Jupyter Notebook)

The ingestion process is performed using `notebook/document.ipynb`:

```
PDF Files (data/pdf_files/)
    │
    ▼  PyMuPDF / PyPDF Loaders
LangChain Document Objects
    │  (page_content + metadata{source, page})
    ▼  RecursiveCharacterTextSplitter
Chunked Documents (e.g., 282 chunks)
    │
    ▼  EmbeddingManager.generate_embeddings()
Embedding Vectors (N × 384 float arrays)
    │
    ▼  VectorStore.add_documents()
ChromaDB PersistentClient (data/vector_store/)
```

**Important:** The ingestion step is **not** part of the web server. It must be run separately via the Jupyter notebook to populate the vector store before the chatbot can answer questions.

### 7.3 Vector Store Persistence

ChromaDB uses SQLite under the hood. The persistent store at `data/vector_store/` contains:
- `chroma.sqlite3` — Main database with embeddings, documents, and metadata
- UUID-named directories — Segment storage files

This directory is `.gitignored` because it can be regenerated from the source PDFs.

---

## 8. Environment Variables & Secrets

All secrets are stored in `.env` at the project root (`.gitignored`):

| Variable          | Required | Used By              | Purpose                              |
|-------------------|----------|----------------------|--------------------------------------|
| `GROQ_API_KEY`    | **Yes**  | `llm.py → get_llm()` | Authentication for Groq Cloud API    |
| `OPENAI_API_KEY`  | No       | (reserved)            | Potential future OpenAI integration  |
| `GOOGLE_API_KEY`  | No       | (reserved)            | Potential future Google AI integration|
| `NVIDIA_API_KEY`  | No       | (reserved)            | Potential future NVIDIA NIM integration|

**Loading Mechanism:**
```python
# In llm.py
load_dotenv(os.path.join(project_root, ".env"))
api_key = os.getenv("GROQ_API_KEY")
```

---

## 9. Dependency Stack

### Core Dependencies (`pyproject.toml`)

| Package                   | Version  | Role                                                  |
|---------------------------|----------|-------------------------------------------------------|
| `fastapi`                 | ≥0.137.1 | Web framework — REST API endpoints                   |
| `uvicorn`                 | ≥0.49.0  | ASGI server — runs FastAPI app                       |
| `chromadb`                | ≥1.5.9   | Vector database — embedding storage & similarity search |
| `sentence-transformers`   | ≥5.5.1   | Embedding model loader & encoder                     |
| `langchain`               | ≥1.3.9   | Document loading, text splitting orchestration       |
| `langchain-groq`          | ≥1.1.3   | Groq LLM integration via LangChain                   |
| `langchain-text-splitters` | ≥1.1.2  | RecursiveCharacterTextSplitter                       |
| `langchain-community`     | ≥0.4.2   | Community document loaders                           |
| `langchain-core`          | ≥1.4.7   | Core LangChain abstractions                          |
| `langchain-openai`        | ≥1.3.2   | (Reserved) OpenAI LLM integration                    |
| `langgraph`               | ≥1.2.5   | (Reserved) Graph-based agent orchestration           |
| `pymupdf`                 | ≥1.27.2  | PDF parsing (PyMuPDF / fitz)                         |
| `pypdf`                   | ≥6.13.2  | Alternative PDF parsing library                      |
| `python-dotenv`           | ≥1.2.2   | .env file loading                                    |
| `tqdm`                    | ≥4.68.2  | Progress bars for batch operations                   |
| `faiss-cpu`               | ≥1.14.3  | (Reserved) Alternative vector similarity search      |
| `unstructured`            | ≥0.23.1  | Document parsing for various file formats            |
| `selenium`                | ≥4.44.0  | (Reserved) Web scraping capabilities                 |
| `undetected-chromedriver` | ≥3.5.5   | (Reserved) Anti-detection web scraping               |
| `curl-cffi`               | ≥0.15.0  | HTTP client with browser impersonation               |
| `typesense`               | ≥2.0.0   | (Reserved) Full-text search engine integration       |
| `ipykernel`               | ≥7.3.0   | Jupyter kernel for notebook execution                |

### Frontend Dependencies (CDN — No Build Step)

| Library          | Version | CDN Source                            | Purpose              |
|------------------|---------|---------------------------------------|----------------------|
| Google Fonts     | —       | `fonts.googleapis.com`                | Inter + Outfit fonts |
| Font Awesome     | 6.4.0   | `cdnjs.cloudflare.com`               | UI icons             |
| Marked.js        | latest  | `cdn.jsdelivr.net/npm/marked`        | Markdown rendering   |

---

## 10. Development & Debugging Guide

### 10.1 Quick Start

```bash
# 1. Clone and enter the project
cd RAG/

# 2. Install dependencies (using uv)
uv sync

# 3. Create .env file with your API key
echo 'GROQ_API_KEY="your-key-here"' > .env

# 4. (If vector store is empty) Run the ingestion notebook
#    Open notebook/document.ipynb in Jupyter and execute all cells

# 5. Verify pipeline works via CLI
python test_pipeline.py

# 6. Launch the web application
python main.py
# → Open http://localhost:8000
```

### 10.2 Development Server

```bash
python main.py
# Uvicorn starts with reload=True
# Any changes to .py files will auto-restart the server
# Frontend changes (HTML/CSS/JS) take effect on browser refresh
```

### 10.3 Testing the API Directly

```bash
# Check server status
curl http://localhost:8000/api/status | python -m json.tool

# Send a chat query
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the limit of working hours?",
    "temperature": 0.1,
    "top_k": 3,
    "score_threshold": 0.0,
    "model": "llama-3.1-8b-instant"
  }' | python -m json.tool
```

### 10.4 Interactive API Docs

FastAPI auto-generates interactive documentation:
- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

### 10.5 Adding New Documents

1. Place new PDF files in `data/pdf_files/`
2. Open `notebook/document.ipynb`
3. Run the ingestion cells to process and embed the new documents
4. Restart the server (`python main.py`) to pick up the updated vector store
5. Verify via `/api/status` that the document count has increased

---

## 11. Known Gotchas & Troubleshooting

### Vector Store Shows 0 Documents

**Symptom:** `/api/status` returns `document_count: 0` even after running ingestion.

**Possible Causes:**
- The notebook ingestion cells were not fully executed
- A different `persist_directory` was used during ingestion vs. server startup
- The `data/vector_store/` directory was deleted or corrupted

**Fix:** Re-run the full ingestion notebook and ensure the `persist_directory` matches `data/vector_store/` (the default).

---

### `GROQ_API_KEY is not set` Error

**Symptom:** Server crashes on startup with a `ValueError`.

**Fix:**
1. Ensure `.env` exists in the project root
2. Verify the key format: `GROQ_API_KEY="gsk_xxxxx"`
3. Check that `python-dotenv` is installed (`uv sync`)

---

### Slow First Response After Server Start

**Symptom:** The first query takes 5-10 seconds, subsequent queries are faster.

**Cause:** The SentenceTransformer model is loaded eagerly on startup, but the first `.encode()` call triggers JIT compilation/warmup.

**Mitigation:** This is expected behavior. The model stays in memory after the first call.

---

### Frontend Chat Not Submitting

**Symptom:** Pressing Enter or clicking Send does nothing.

**Possible Causes:**
- JavaScript error in the console (check browser DevTools → Console)
- The `send-btn` is still `disabled` (textarea validation)
- The form's `submit` event is not being captured

**Fix:** Open browser DevTools, check for errors, and verify that `app.js` loaded successfully (Network tab).

---

### CORS Errors in Browser Console

**Symptom:** `Access-Control-Allow-Origin` errors when fetching from the API.

**Current Config:** CORS is configured to allow all origins (`allow_origins=["*"]`). This is suitable for local development. For production, restrict to specific domains.

---

### ChromaDB SQLite Locking

**Symptom:** `OperationalError: database is locked` when running the notebook while the server is active.

**Fix:** Stop the FastAPI server before running ingestion in the notebook. ChromaDB's SQLite backend doesn't support concurrent write access from multiple processes.

---

> **Contributing:** When modifying the `rag_pipeline` package, ensure all new classes/functions are re-exported through `__init__.py` to maintain the clean import interface.
