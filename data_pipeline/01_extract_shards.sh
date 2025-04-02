#!/bin/sh
# Define source and target directories
SRC_DIR="/gpfs/scratch/ehpc42/datasets/datacomp/large_filtered/shards"
TARGET_DIR="/gpfs/scratch/ehpc42/datasets/datacomp/large_hybrid/tmp"

export SRC_DIR
export TARGET_DIR

# Use find to locate tar files in the source directory and process them in parallel
find "$SRC_DIR" -maxdepth 1 -type f -name "*.tar" | \
    xargs -I {} -P 32 sh -c '
        tarfile="{}"
        base=$(basename "$tarfile" .tar)
        echo "Processing: $base"

        mkdir -p "$TARGET_DIR/$base"

        start=$(date +%s)

        tar xf "$tarfile" --no-same-permissions -C "$TARGET_DIR/$base"

        end=$(date +%s)

        elapsed=$(( end - start ))
        echo "Extraction of $base took $elapsed seconds."
    '

