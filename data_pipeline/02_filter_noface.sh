#!/bin/sh

# Load necessary modules
module purge
module load oneapi hdf5 python/3.12.1

# Define source and target directories
SRC_DIR="/gpfs/scratch/ehpc42/datasets/datacomp/small_hybrid/tmp"
TARGET_DIR="/gpfs/scratch/ehpc42/datasets/datacomp/small_hybrid/captions"

# Use find to locate tar files in the source directory
find "$SRC_DIR" -maxdepth 1 -type d -name '*[0-9]*' | while IFS= read -r directory; do
    # Extract the base name (without the .tar extension)
    base=$(basename "$directory")
    echo "Processing: $base"

    python ./filter_noface.py --input $directory --output $TARGET_DIR

done

