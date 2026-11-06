#!/usr/bin/env bash
# Download both datasets into data/.   Usage: bash scripts/download_data.sh
# Colab sometimes fails the Simula SSL certificate check (curl error 60), so we retry with
# --insecure and fall back to wget. These are public research datasets with no credentials involved.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$ROOT/data"
mkdir -p "$DATA"
cd "$DATA"

fetch() {  # fetch <out> <url>
  local out="$1" url="$2"
  [ -s "$out" ] && { echo "  $out already downloaded"; return 0; }
  curl -fSL --retry 3 -o "$out" "$url" \
    || curl -fSL --insecure --retry 3 -o "$out" "$url" \
    || wget -q --no-check-certificate -O "$out" "$url" \
    || { echo "ERROR: could not download $url"; rm -f "$out"; return 1; }
}

echo "[1/2] Kvasir-SEG ..."
if [ ! -d Kvasir-SEG/images ]; then
  fetch kvasir-seg.zip https://datasets.simula.no/downloads/kvasir-seg.zip && unzip -q -o kvasir-seg.zip
fi
echo "[2/2] BCCD ..."
if [ ! -d BCCD_Dataset-master/BCCD/JPEGImages ]; then
  fetch bccd.zip https://github.com/Shenggan/BCCD_Dataset/archive/refs/heads/master.zip && unzip -q -o bccd.zip
fi

K=$(ls Kvasir-SEG/images 2>/dev/null | wc -l)
B=$(ls BCCD_Dataset-master/BCCD/JPEGImages 2>/dev/null | wc -l)
echo "Kvasir-SEG images: $K | BCCD images: $B"
if [ "$K" -eq 0 ] || [ "$B" -eq 0 ]; then
  echo "ERROR: a dataset is empty - fix the download before running inference."; exit 1
fi
echo "Done. Data in $DATA"
