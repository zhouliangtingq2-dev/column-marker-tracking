#!/usr/bin/env bash
# One-command demo on Linux/macOS: creates the venv on first run, then runs the whole workflow on the synthetic video.
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  . .venv/bin/activate
  pip install -r requirements.txt
else
  . .venv/bin/activate
fi
python scripts/demo.py --root demo_runs "$@"
echo "report: demo_runs/report/report.md"
