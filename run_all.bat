@echo off
echo ============================================
echo     Quant-Core FULL STACK RUNNER
echo ============================================

call .\.venv\Scripts\activate.bat

echo Starting Backend (FastAPI)...
start cmd /k "uvicorn backend.app:app --reload --port 8000"

echo Starting Streamlit App...
start cmd /k "streamlit run serving/dashboard.py"

echo Both backend & frontend started!
pause
