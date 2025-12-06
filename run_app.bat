@echo off
echo ============================================
echo   Quant-Core Trading MVP - Streamlit Runner
echo ============================================

REM Activate virtual environment
call .\.venv\Scripts\activate.bat

if %errorlevel% neq 0 (
    echo Could not activate virtual environment!
    pause
    exit /b 1
)

echo Virtual environment activated.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements if needed
echo Installing dependencies...
python -m pip install -r requirements.txt

echo Starting Streamlit...
streamlit run serving/dashboard.py

pause
