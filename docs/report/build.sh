#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
$PY gen_tables.py || true
for i in 1 2; do
  if ! xelatex -interaction=nonstopmode main.tex > build.log 2>&1; then
    echo "LaTeX failed:"; grep -nE "^!|^l\.[0-9]" build.log | head -20; exit 1
  fi
done
echo "built: $(pwd)/main.pdf"
pdfinfo main.pdf | grep -E "Pages|Page size"
