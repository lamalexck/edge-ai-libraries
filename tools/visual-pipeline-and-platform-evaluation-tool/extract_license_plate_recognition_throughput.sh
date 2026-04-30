#!/bin/bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

INPUT_FILE=${1:-shared/output/LicensePlateRecognition_results.txt}
OUTPUT_FILE=${2:-shared/output/LicensePlateRecognition_per_stream.csv}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/extract_throughput_to_csv.sh" "$INPUT_FILE" "$OUTPUT_FILE"
