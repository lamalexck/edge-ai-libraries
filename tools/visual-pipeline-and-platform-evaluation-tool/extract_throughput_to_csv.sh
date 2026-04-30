#!/bin/bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

INPUT_FILE=${1:-shared/output/SmartParking_results.txt}
OUTPUT_FILE=${2:-shared/output/SmartParking_per_stream.csv}

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: input file not found: $INPUT_FILE" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_FILE")"

tmp_file="$(mktemp)"
trap 'rm -f "$tmp_file"' EXIT

if ! awk '
BEGIN {
    current_variant = ""
    rows = 0
}

/^Pipeline for / {
    line = $0
    sub(/^Pipeline for /, "", line)
    sub(/ variant.*/, "", line)
    current_variant = line
    next
}

/FpsCounter\(overall/ && /per-stream=/ {
    if (current_variant == "") {
        next
    }

    line = $0
    sub(/^.*per-stream=/, "", line)
    sub(/ fps.*$/, "", line)
    if (line ~ /^[0-9]+(\.[0-9]+)?$/) {
        printf "%s,%s\n", current_variant, line
        rows++
    }
    next
}

END {
    if (rows == 0) {
        exit 2
    }
}
' "$INPUT_FILE" > "$tmp_file"; then
    status=$?
    if [[ $status -eq 2 ]]; then
        echo "Error: no throughput rows found in input file: $INPUT_FILE" >&2
    else
        echo "Error: failed while parsing input file: $INPUT_FILE" >&2
    fi
    exit "$status"
fi

rows_written="$(wc -l < "$tmp_file")"

{
    echo "variant,per_stream_avg_fps"
    cat "$tmp_file"
} > "$OUTPUT_FILE"

echo "Wrote ${rows_written} rows to ${OUTPUT_FILE}"
