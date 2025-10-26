# app.py
import json
import os
import streamlit as st
from qdrant_client import QdrantClient
from embedding import load_text_embedder
from rag import retrieve_docs, generate_answer, ensure_collection

# ============================
# CONFIG
# ============================
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "questdb_docs")
EMBED_MODEL = os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
LLM_MODEL = os.getenv("LLM_MODEL", "")

_embedder_cache = {}

def get_embedder(model_name: str = None):
    if model_name is None:
        model_name = EMBED_MODEL
    if model_name in _embedder_cache:
        return _embedder_cache[model_name]
    try:
        emb = load_text_embedder(model_name)
        _embedder_cache[model_name] = emb
        return emb
    except Exception as e:
        fallback = "sentence-transformers/all-MiniLM-L12-v2"
        emb = load_text_embedder(fallback)
        _embedder_cache[fallback] = emb
        st.warning(f"Failed to load '{model_name}': {e}. Falling back to '{fallback}'")
        return emb

client = QdrantClient(url=QDRANT_URL)

# ============================
# UI
# ============================
st.set_page_config(page_title="QuestDB RAG Assistant", layout="wide")
st.title("QuestDB RAG Assistant")
st.markdown("Enter table schema JSON and your question. Retrieves docs and optionally generates an answer.")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Table Metadata")
    default_schema = {
        "table_name": "sensor_data",
        "columns": [
            {"name": "device_id", "type": "STRING"},
            {"name": "temperature", "type": "DOUBLE"},
            {"name": "ts", "type": "TIMESTAMP"}
        ],
        "partition_by": "DAY",
        "wal_enabled": True
    }
    table_metadata_input = st.text_area("Enter JSON", value=json.dumps(default_schema, indent=2), height=300)

    st.subheader("Model selection")
    embed_options = [EMBED_MODEL, "sentence-transformers/all-MiniLM-L12-v2", "Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-1.7B"]
    embed_options = list(dict.fromkeys(embed_options))
    selected_embed_model = st.selectbox("Embedding model", embed_options, index=0)

    llm_options = ["(disabled)", "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"]
    default_llm_idx = 0 if not LLM_MODEL else llm_options.index(LLM_MODEL) if LLM_MODEL in llm_options else 0
    selected_llm = st.selectbox("LLM model (optional)", llm_options, index=default_llm_idx)

    st.subheader("Your Question")
    user_question = st.text_area("Enter your question here", height=150)

    if st.button("Get Answer"):
        try:
            table_metadata = json.loads(table_metadata_input)
        except Exception as e:
            st.error(f"Invalid JSON: {e}")
            st.stop()

        embedder = get_embedder(selected_embed_model)
        ensure_collection(client, COLLECTION_NAME, embedder)

        # Ingest demo docs if collection empty
        points, _ = client.scroll(collection_name=COLLECTION_NAME)
        if len(points) == 0:
            st.info("Ingesting demo documents into Qdrant...")
            demo_docs = [
                "QuestDB stores time-series data efficiently and supports SQL queries with time-based filtering.",
                "Use the LATEST BY clause to get the most recent value per key.",
                "Aggregations over time windows are supported via SQL functions."
            ]
            vectors = [embedder.encode([d])[0].tolist() for d in demo_docs]
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=[{"id": i, "vector": v, "payload": {"chunk_text": d}} for i, (d, v) in enumerate(zip(demo_docs, vectors))]
            )
            st.success("Demo documents ingested.")

        # Retrieve docs
        try:
            docs = retrieve_docs(user_question, table_metadata, client, embedder, collection=COLLECTION_NAME)
        except Exception as e:
            st.error(f"Failed to retrieve documents: {e}")
            st.stop()

        # LLM generation
        answer = None
        llm_to_use = "" if selected_llm == "(disabled)" else selected_llm
        if llm_to_use:
            force_cpu = st.checkbox("Force LLM on CPU (very slow & memory heavy)")
            if force_cpu:
                os.environ["FORCE_LLM_ON_CPU"] = "1"
            with st.spinner("Generating answer..."):
                try:
                    answer = generate_answer(user_question, table_metadata, docs, model_name=llm_to_use)
                except RuntimeError as e:
                    st.error(str(e))
                    answer = None
        else:
            st.info("LLM generation disabled. Only retrieval performed.")

        st.session_state["docs"] = docs
        st.session_state["answer"] = answer

with col2:
    if "docs" in st.session_state:
        st.subheader("Retrieved Documentation Chunks")
        for i, doc in enumerate(st.session_state["docs"], start=1):
            with st.expander(f"Chunk {i}"):
                st.write(doc)
    if "answer" in st.session_state and st.session_state["answer"]:
        st.subheader("Answer")
        st.markdown(f"**{st.session_state['answer']}**")
