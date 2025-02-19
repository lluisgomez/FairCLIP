#!/bin/sh

# Load necessary modules
module purge
module load oneapi hdf5 python/3.12.1

# Define the source directory containing the modified shards and the destination for the new tar files
SRC_DIR="/gpfs/scratch/ehpc42/datasets/datacomp/small_filtered/shards"
DEST_DIR="/gpfs/scratch/ehpc42/datasets/datacomp/small_hybrid/shards"
EDIT_DIR="/gpfs/scratch/ehpc42/datasets/datacomp/small_hybrid/edits"
mkdir -p "$DEST_DIR"

# Use find to locate tar files in the source directory
find "$SRC_DIR" -maxdepth 1 -type f -name "*.tar" | while IFS= read -r tarfile; do
    # Extract the base name (without the .tar extension)
    base=$(basename "$tarfile" .tar)
    echo "Creating tar archive for: $base"

    # Create a tar file from the original tar and updated samples
    python ./reshard.py --input $tarfile --output $DEST_DIR/$base.tar --edits $EDIT_DIR
done
