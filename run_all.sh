#!/usr/bin/env bash
# One-command reproduction: tests -> experiments -> figures.
set -euo pipefail
cd "$(dirname "$0")"
echo "== [1/3] pytest =="
python3 -m pytest tests -q
echo "== [2/3] experiments -> results/*.csv =="
python3 src/experiments.py
echo "== [3/3] figures -> docs/figures/*.png =="
python3 src/make_figures.py
echo "OK: all reproduced."
