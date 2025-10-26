echo "Installing dependencies..."
pip install -r requirements.txt
echo "Starting Qdrant..."
docker run -d -p 6333:6333 qdrant/qdrant
echo "Scraping QuestDB documentation..."
echo "Ingesting data into Qdrant..."
echo "Running demo query..."
#!/usr/bin/env bash
set -euo pipefail

# Create necessary folders
mkdir -p data scraper

# Install dependencies into the active Python environment
echo "Installing dependencies... (will install into the active Python environment)"
pip install -r requirements.txt

# Choose port: prefer 6333, fallback 6334 if in use
QDRANT_HOST_PORT=6333
if ss -ltn | grep -q ":6333" ; then
    echo "Port 6333 is in use; falling back to 6334"
    QDRANT_HOST_PORT=6334
fi

echo "Starting Qdrant on host port ${QDRANT_HOST_PORT}..."
docker run -d -p ${QDRANT_HOST_PORT}:6333 qdrant/qdrant
export QDRANT_URL="http://localhost:${QDRANT_HOST_PORT}"
echo "QDRANT_URL=${QDRANT_URL}"

echo "Scraping QuestDB documentation (limited pages for quick run)..."
python3 scraper/scrape.py --out data/questdb_docs.jsonl --max-pages 500 --delay 0.4

echo "Ingesting data into Qdrant..."
if [ -z "${EMBED_MODEL:-}" ]; then
    EMBED_MODEL="sentence-transformers/all-MiniLM-L12-v2"
fi
echo "Using embedding model: ${EMBED_MODEL}"
QDRANT_HEALTH="${QDRANT_URL}/api/version"
echo "Waiting for Qdrant at ${QDRANT_HEALTH} ..."
for i in {1..30}; do
    if curl -s --fail "${QDRANT_HEALTH}" >/dev/null 2>&1; then
        echo "Qdrant is ready."
        break
    fi
    echo "Waiting for Qdrant... $i"
    sleep 2
done

python3 ingest.py --host "${QDRANT_URL}" --jsonl data/questdb_docs.jsonl --collection questdb_docs --model "${EMBED_MODEL}" --recreate

echo "Running demo query (calling rag.py directly against ${QDRANT_URL})..."
python3 rag.py --host "${QDRANT_URL}" --table data/demo_table.json --question "How can I efficiently query the average temperature per device over the last hour?"

echo "All steps completed successfully!"

# Optionally start Streamlit UI unless STREAMLIT_NO_START is set
if [ -z "${STREAMLIT_NO_START:-}" ]; then
    echo "Starting Streamlit UI in background..."
    # try nohup to keep it running; output is redirected to streamlit_ui.log
    nohup streamlit run streamlit_ui.py > streamlit_ui.log 2>&1 &
    echo "Streamlit started (logs: streamlit_ui.log)"
else
    echo "Skipping Streamlit startup because STREAMLIT_NO_START is set."
fi