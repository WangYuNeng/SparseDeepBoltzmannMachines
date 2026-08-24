#!/usr/bin/env bash
# Generate 5120 images per digit (0-9) at batch_size=1024, i.e. 5 calls per
# digit x 10 digits = 50 calls total, run sequentially, saved as one .npy
# file per call (see --save-format in generate_labeled_batch.py).
#
# Usage: generate_dataset.sh [output_dir] [num_sweeps]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:-$SCRIPT_DIR/../generated_dataset}"
NUM_SWEEPS="${2:-10000}"

BATCH_SIZE=1024
IMAGES_PER_DIGIT=5120
CHUNKS_PER_DIGIT=$((IMAGES_PER_DIGIT / BATCH_SIZE))

mkdir -p "$OUTPUT_DIR"

for label in $(seq 0 9); do
    for chunk in $(seq 0 $((CHUNKS_PER_DIGIT - 1))); do
        start_idx=$((chunk * BATCH_SIZE))
        # seed MUST be unique per call (see generate_labeled_batch.py docstring)
        seed=$((label * CHUNKS_PER_DIGIT + chunk + 1))
        echo "=== label=$label chunk=$((chunk + 1))/$CHUNKS_PER_DIGIT seed=$seed start_idx=$start_idx ==="
        python "$SCRIPT_DIR/generate_labeled_batch.py" \
            "$label" "$seed" "$BATCH_SIZE" "$start_idx" "$OUTPUT_DIR" "$NUM_SWEEPS" \
            --save-format npy --progress
    done
done

total_calls=$((CHUNKS_PER_DIGIT * 10))
total_images=$((total_calls * BATCH_SIZE))
echo "Done: $total_calls calls, $total_images images total, written to $OUTPUT_DIR"
