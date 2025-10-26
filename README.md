QuestDB RAG System

A Retrieval-Augmented Generation (RAG) system for QuestDB documentation using Qwen3 embeddings and Qdrant vector database.

The system allows you to:

Scrape QuestDB documentation.

Chunk and embed the documentation.

Store embeddings in Qdrant.

Ask questions grounded in the documentation using Qwen3 LLM.

##Folder Structure
questdb_rag/
│
├─ scraper/
│   └─ scrape.py          # Scrape QuestDB docs to JSONL
│
├─ ingest/
│   └─ ingest.py          # Embed and upload docs to Qdrant
│
├─ rag/
│   └─ rag.py             # RAG pipeline (retrieve + LLM answer)
│
├─ embedding.py           # Embedding loader (Qwen3 / SentenceTransformers)
├─ demo.py                # Demo script with sample table + question
├─ app.py                 # Streamlit UI for production
├─ requirements.txt       # Python dependencies
└─ README.md              # This file
Setup
1. Clone repository
git clone <your-repo-url>
cd questdb_rag
2. Create virtual environment (recommended)
python -m venv venv
# Activate:
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

Start Qdrant
Run Qdrant via Docker:
docker run -p 6333:6333 qdrant/qdrant
Access the Qdrant UI at http://localhost:6333
 to monitor collections.
 Scrape QuestDB Documentation
python scraper/scrape.py --out data/questdb_docs.jsonl
Scrape QuestDB Documentation
python scraper/scrape.py --out data/questdb_docs.jsonl


Output: data/questdb_docs.jsonl

Each entry contains:

{
  "source_url": "...",
  "title": "...",
  "section": "...",
  "text": "..."
}

Ingest into Qdrant
python ingest/ingest.py --jsonl data/questdb_docs.jsonl --host http://localhost:6333 --collection questdb_docs --recreate


Creates questdb_docs collection in Qdrant.

Embeds documentation using Qwen3-embedding-0.6B.

Upserts points in batches for efficiency.

Run Demo Script
python demo.py


Uses a sample table schema and question.

Outputs:

Top retrieved documentation chunks.

Final LLM-generated answer.

Run Streamlit UI (Production)
streamlit run app.py


Provides interactive UI:

Enter table metadata (JSON).

Enter a natural language question.

Retrieve docs and generate answer.

Expand retrieved chunks and view grounded answer.

RAG Pipeline

Embedding: Qwen3-embedding-0.6B converts docs and questions to vector embeddings.

Retrieval: Qdrant retrieves top-k similar documentation chunks.

LLM Answering: Qwen3-0.6B generates answer using retrieved docs and table metadata.

Configuration Options

Collection Name: Change --collection in ingest script.

Embedding Model: --model in ingest script, supports fallback to sentence-transformers/all-MiniLM-L12-v2.

Batch Size: --batch-size in ingest script.

Top-k Docs: Adjust top_k in rag.rag.retrieve_docs().

Requirements

Python 3.10+

Docker (for Qdrant)

Packages: See requirements.txt

Quick troubleshooting & environment variables
------------------------------------------

This project can run in three modes for generation:

- GPU local model: when a CUDA-enabled GPU is available the requested LLM (for example Qwen) will be loaded locally and used for generation. This provides the best latency and avoids API costs.
- HF Inference API: if you set an HF API token the app will prefer calling Hugging Face Inference for generation when no GPU is present. This avoids loading large local models but may incur API costs and rate limits.
- CPU fallback: when no GPU and no HF token are present the code will automatically fall back to a small CPU-friendly model (default `distilgpt2`) so you can still test the pipeline without extra configuration. If you explicitly want to attempt loading the requested (potentially large) model on CPU, set `FORCE_LLM_ON_CPU=1`.

Important environment variables

- `QDRANT_URL` — URL for Qdrant (default: `http://localhost:6333`). Example (PowerShell):

  $env:QDRANT_URL = "http://localhost:6333"

- `EMBED_MODEL` — Embedding model id (default in code: `Qwen/Qwen3-Embedding-0.6B`). For local CPU runs prefer a small model such as `sentence-transformers/all-MiniLM-L12-v2`.

- `LLM_MODEL` — Default LLM model id used for generation in the UI (empty = disabled).

- `HF_API_TOKEN` or `HUGGINGFACE_API_TOKEN` — Hugging Face Inference API token. If set and no CUDA is available the code will call the HF Inference API for completions instead of trying to load large local models.

  PowerShell example (temporary for current shell):

  $env:HF_API_TOKEN = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

- `FORCE_LLM_ON_CPU` — If set to `1` (or `true`), the code will attempt to load the requested LLM locally on CPU even when it looks large (danger: may OOM or be extremely slow). Example:

  $env:FORCE_LLM_ON_CPU = "1"

- `FORCE_QWEN_ON_CPU` — Similar to above but used by the Qwen embedder path where applicable.

- `CPU_FRIENDLY_LLM` — (Optional) Override the default small CPU model used as a fallback (default `distilgpt2`). Example:

  $env:CPU_FRIENDLY_LLM = "gpt2"

Starting Qdrant
----------------

If you see connection errors (connection refused, getaddrinfo failed, ResponseHandlingException) the app cannot reach Qdrant. To start a local Qdrant using Docker:

```powershell
docker run -p 6333:6333 qdrant/qdrant
```

Then either restart the app or set `QDRANT_URL` to the reachable address.

Demo fallback (no Qdrant)
--------------------------

The Streamlit UI includes a lightweight demo fallback when Qdrant is unreachable: the app will show an error/hint and then use a small built-in set of documentation chunks and perform in-memory retrieval using the selected embedder. This makes it easy to experiment with the UI and generation without an external vector DB. Note:

- The demo docs are minimal and intended for experimentation only.
- The embedder still runs locally to encode the question and demo docs — pick a small `EMBED_MODEL` on CPU to avoid heavy loads.

Generation behavior notes
------------------------

- The generator now returns only the newly generated text (not the original prompt) to avoid the model echoing the question back in the response.
- If you want more/less creative outputs you can adjust generation parameters in `rag.py` (temperature, top_p, max_new_tokens). Defaults are conservative.

Quick run examples (PowerShell)
------------------------------

- Run Streamlit UI:

  ```powershell
  streamlit run app.py
  ```

- Run the CLI RAG script (retrieval + generation) against the demo table:

  ```powershell
  python rag.py --table data/demo_table.json --question "How can I efficiently query the average temperature per device over the last hour?" --llm-model Qwen/Qwen3-0.6B
  ```

If you have no GPU and want to use HF Inference, set HF_API_TOKEN first. If you prefer to test locally on CPU with a small model, either set `CPU_FRIENDLY_LLM` or leave token unset and the code will fall back automatically.

Troubleshooting tips
--------------------

- I get connection refused / getaddrinfo failed: start Qdrant (see above) or set the correct `QDRANT_URL`.
- I see warnings about large models on CPU: either set `HF_API_TOKEN` to use the Inference API, set `FORCE_LLM_ON_CPU=1` to attempt local CPU load (risky), or pick a smaller model for `EMBED_MODEL` / `LLM_MODEL`.
- The model repeats the question or returns the prompt + answer: the code now slices generated tokens to return only the completion. If you still observe echoing, try increasing sampling temperature slightly (in `rag.py`) or reworking the prompt.

Next steps
----------

- If you want, I can add a visible banner in the Streamlit UI that shows whether the app will use Qdrant or demo mode, and whether generation will use HF Inference, a local GPU model, or the CPU fallback.
- I can also add explicit CLI options to force demo mode or select the CPU-friendly fallback model without env vars.
