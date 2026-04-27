#!/usr/bin/env bash
# DWN_CPT_SRC.sh — Download and extract CPT Fortran source (v17.8.3)
# Source URL from the cptbin conda recipe (iri-nextgen channel).
set -euo pipefail

VERSION="17.8.3"
URL="https://academiccommons.columbia.edu/doi/10.7916/fdp6-v391/download"
DEST_DIR="$(cd "$(dirname "$0")/.." && pwd)/CPT_SRC"

if [ -d "$DEST_DIR/CPT/$VERSION" ]; then
    echo "[INFO] CPT source already present at $DEST_DIR/CPT/$VERSION"
    exit 0
fi

mkdir -p "$DEST_DIR"
TMPFILE="$(mktemp)"
trap 'rm -f "$TMPFILE"' EXIT

echo "[INFO] Downloading CPT $VERSION source..."
curl -fSL "$URL" -o "$TMPFILE"

echo "[INFO] Extracting to $DEST_DIR..."
tar xzf "$TMPFILE" -C "$DEST_DIR"

if [ -d "$DEST_DIR/$VERSION" ]; then
    echo "[INFO] CPT source extracted to $DEST_DIR/$VERSION"
    echo "[INFO] Key files:"
    ls "$DEST_DIR/$VERSION"/*.F95 2>/dev/null | head -15
else
    echo "[WARN] Expected directory $DEST_DIR/$VERSION not found. Contents:"
    ls "$DEST_DIR"
fi
