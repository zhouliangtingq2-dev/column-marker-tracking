@echo off
REM One-click demo on Windows: creates the venv on first run, then runs the whole workflow on the synthetic video.
cd /d "%~dp0"
if not exist .venv (
  python -m venv .venv
  call .venv\Scripts\activate
  pip install -r requirements.txt
) else (
  call .venv\Scripts\activate
)
python scripts\demo.py --root demo_runs %*
start "" "%CD%\demo_runs\report"
pause
