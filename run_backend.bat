@echo off
echo ============================================
echo       Starting FastAPI Backend
echo ============================================

call .\.venv\Scripts\activate.bat

echo Running backend on port 8000...
uvicorn backend.app:app --reload --port 8000

pause
