#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT="${1:-validation/downscaling_validation_report.md}"
OUTPUT="${2:-validation/downscaling_validation_report.pdf}"

case "$INPUT" in
  /*) INPUT_ABS="$INPUT" ;;
  *) INPUT_ABS="$ROOT/$INPUT" ;;
esac

case "$OUTPUT" in
  /*) OUTPUT_ABS="$OUTPUT" ;;
  *) OUTPUT_ABS="$ROOT/$OUTPUT" ;;
esac

if ! command -v pandoc >/dev/null 2>&1; then
  echo "pandoc is required to render the validation report PDF." >&2
  exit 1
fi

if [[ ! -f "$INPUT_ABS" ]]; then
  echo "Report source not found: $INPUT_ABS" >&2
  exit 1
fi

CSS="$ROOT/validation/report_pdf.css"
BUILD_DIR="$ROOT/validation/report_build"
HTML="$BUILD_DIR/$(basename "${OUTPUT_ABS%.pdf}").html"
FINAL_HTML="$BUILD_DIR/$(basename "${OUTPUT_ABS%.pdf}").final.html"
PREPARED_MD="$BUILD_DIR/$(basename "${OUTPUT_ABS%.pdf}").prepared.md"
DRAFT_PDF="$BUILD_DIR/$(basename "${OUTPUT_ABS%.pdf}").draft.pdf"

mkdir -p "$BUILD_DIR" "$(dirname "$OUTPUT_ABS")"

TITLE="$(awk '/^# / { sub(/^# /, ""); print; exit }' "$INPUT_ABS")"
REPORT_DATE="$(awk '/^Date:/ { sub(/^Date:[[:space:]]*/, ""); print; exit }' "$INPUT_ABS")"

awk -v title="$TITLE" -v report_date="$REPORT_DATE" '
  BEGIN {
    print "---"
    print "title: " title
    print "date: " report_date
    print "---"
    print ""
  }
  NR == 1 && /^# / { next }
  NR <= 3 && /^Date:/ { next }
  NR == 2 && /^$/ { next }
  { print }
' "$INPUT_ABS" > "$PREPARED_MD"

pandoc "$PREPARED_MD" \
  --from gfm+smart \
  --to html5 \
  --standalone \
  --toc \
  --toc-depth=2 \
  --section-divs \
  --embed-resources \
  --resource-path="$ROOT/validation:$ROOT" \
  --css="$CSS" \
  --include-after-body="$ROOT/validation/report_toc_pages.js" \
  --output "$HTML"

if [[ -n "${BROWSER_BIN:-}" ]]; then
  BROWSER="$BROWSER_BIN"
elif [[ -x "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" ]]; then
  BROWSER="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
elif [[ -x "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge" ]]; then
  BROWSER="/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"
else
  cat >&2 <<EOF
Rendered HTML to:
  $HTML

Set BROWSER_BIN to a Chromium-compatible browser to print the PDF, for example:
  BROWSER_BIN="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" $0 "$INPUT" "$OUTPUT"
EOF
  exit 1
fi

"$BROWSER" \
  --headless \
  --disable-gpu \
  --no-first-run \
  --no-pdf-header-footer \
  --virtual-time-budget=1000 \
  --print-to-pdf="$DRAFT_PDF" \
  "file://$HTML" >/dev/null 2>&1

if command -v pdftotext >/dev/null 2>&1 && command -v pdfinfo >/dev/null 2>&1; then
  "$ROOT/validation/fill_report_toc_pages.py" "$PREPARED_MD" "$DRAFT_PDF" "$HTML" >/dev/null
else
  echo "pdftotext/pdfinfo not found; TOC page numbers will use browser layout estimates." >&2
fi

cp "$HTML" "$FINAL_HTML"

"$BROWSER" \
  --headless \
  --disable-gpu \
  --no-first-run \
  --no-pdf-header-footer \
  --virtual-time-budget=1000 \
  --print-to-pdf="$OUTPUT_ABS" \
  "file://$FINAL_HTML" >/dev/null 2>&1

echo "Rendered PDF: $OUTPUT_ABS"
echo "Intermediate HTML: $HTML"
