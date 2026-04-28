@echo off
cd /d "%~dp0"
pip install fastapi "uvicorn[standard]" requests openpyxl -q
uvicorn api:app --host 0.0.0.0 --port 8511 --reload
pause
