#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
RAW=data/fma/raw
mkdir -p "$RAW"
for f in fma_metadata.zip fma_small.zip; do
  if [ -s "$RAW/$f" ]; then echo "skip $f"; continue; fi
  echo "downloading $f"
  curl -L --retry 5 --retry-delay 5 -C - -o "$RAW/$f" "https://os.unil.cloud.switch.ch/fma/$f"
done
echo "DOWNLOAD_DONE"
ls -lh "$RAW"
