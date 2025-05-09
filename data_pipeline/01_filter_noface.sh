#!/bin/sh

# Load necessary modules
module purge
module load oneapi hdf5 python/3.12.1

# Define source and target
SRC_DIR="/gpfs/scratch/ehpc42/datasets/datacomp/large_filtered/shards"
TARGET_DIR="/gpfs/scratch/ehpc42/datasets/datacomp/large_hybrid/captions"
TMP_DIR="/gpfs/scratch/ehpc42/datasets/datacomp/large_hybrid/edits"

mkdir -p "$TARGET_DIR"
mkdir -p "$TMP_DIR"

# Process each .tar file
find "$SRC_DIR" -maxdepth 1 -type f -name "*.tar" | while read -r tarfile; do
    echo "Processing: $tarfile"
    python ./01_filter_noface.py --input "$tarfile" --output "$TARGET_DIR" --edits "$TMP_DIR"
done
