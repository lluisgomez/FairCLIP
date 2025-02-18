#!/bin/sh
# Define source and target directories
SRC_DIR="/gpfs/scratch/ehpc42/datasets/datacomp/small_filtered/shards"
TARGET_DIR="/gpfs/scratch/ehpc42/datasets/datacomp/small_hybrid/tmp"

# Use find to locate tar files in the source directory
find "$SRC_DIR" -maxdepth 1 -type f -name "*.tar" | while IFS= read -r tarfile; do
    # Extract the base name (without the .tar extension)
    base=$(basename "$tarfile" .tar)
    echo "Processing: $base"

    # Create the target directory for this file (if it doesn't already exist)
    mkdir -p "$TARGET_DIR/$base"

    # Record the start time
    start=$(date +%s)

    # Extract the tar archive into the target directory
    tar xf "$tarfile" -C "$TARGET_DIR/$base"

    # Record the end time
    end=$(date +%s)

    # Calculate and display the elapsed time
    elapsed=$(( end - start ))
    echo "Extraction of $base took $elapsed seconds."
done

