echo Installing dependencies...
pip install -r requirements.txt
echo Starting Qdrant...
docker run -d -p 6333:6333 qdrant/qdrant
echo Scraping QuestDB documentation...
echo Ingesting data into Qdrant...
echo Running demo query...
@echo off

REM --- prepare folders ---
if not exist data mkdir data
if not exist scraper mkdir scraper

REM --- install dependencies ---
echo Installing dependencies... (will install into the active Python environment)
pip install -r requirements.txt

REM --- choose host port for Qdrant (prefer 6333, fallback to 6334) ---
set "QDRANT_HOST_PORT=6333"
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":6333"') do set PORT6333_PID=%%a
if defined PORT6333_PID (
    echo Port 6333 is in use by PID %PORT6333_PID%. Falling back to 6334.
    set "QDRANT_HOST_PORT=6334"
)

echo Starting Qdrant on host port %QDRANT_HOST_PORT% ...
docker run -d -p %QDRANT_HOST_PORT%:6333 qdrant/qdrant
set "QDRANT_URL=http://localhost:%QDRANT_HOST_PORT%"
echo QDRANT_URL=%QDRANT_URL%

REM --- scrape docs ---
echo Scraping QuestDB documentation (limited pages for quick run)...
python scraper\scrape.py --out data/questdb_docs.jsonl --max-pages 500 --delay 0.4

REM --- ingest into Qdrant ---
echo Ingesting data into Qdrant...
if "%EMBED_MODEL%"=="" (
    REM default to a small, fast sentence-transformer to avoid large downloads; override by setting EMBED_MODEL env var
    set "EMBED_MODEL=sentence-transformers/all-MiniLM-L12-v2"
)
echo Using embedding model: %EMBED_MODEL%
REM Wait for Qdrant to be ready (poll /api/version)
set "QDRANT_HEALTH=%QDRANT_URL%/api/version"
echo Waiting for Qdrant at %QDRANT_HEALTH% ...
for /l %%i in (1,1,30) do (
    powershell -Command "try { iwr -UseBasicParsing -TimeoutSec 2 '%QDRANT_HEALTH%' | Out-Null; exit 0 } catch { exit 1 }"
    if not errorlevel 1 goto QDRANT_READY
    echo Waiting for Qdrant... %%i
    timeout /t 2 >nul
)
echo Failed to connect to Qdrant at %QDRANT_HEALTH% after waiting; continuing anyway.
:QDRANT_READY
echo Qdrant reported healthy.

python ingest.py --host %QDRANT_URL% --jsonl data/questdb_docs.jsonl --collection questdb_docs --model %EMBED_MODEL% --recreate

REM --- run demo query ---
echo Running demo query (calling rag.py directly against %QDRANT_URL%)...
%PYTHON% 2>nul || set PYTHON=python
python rag.py --host %QDRANT_URL% --table data/demo_table.json --question "How can I efficiently query the average temperature per device over the last hour?"

echo All steps completed successfully!
echo If your app expects Qdrant at http://localhost:6333 and you used a fallback port, set QDRANT_URL accordingly.

REM --- optionally start Streamlit UI unless STREAMLIT_NO_START is set ---
if "%STREAMLIT_NO_START%"=="" (
    echo Starting Streamlit UI in a new window...
    REM use start to open a new terminal window and run streamlit
    start "Streamlit UI" cmd /c "streamlit run streamlit_ui.py"
) else (
    echo Skipping Streamlit startup because STREAMLIT_NO_START is set.
)