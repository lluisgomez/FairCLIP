#!/bin/sh
# Define the source directory containing the modified shards and the destination for the new tar files
SRC_DIR="/gpfs/scratch/ehpc42/datasets/datacomp/small_hybrid/tmp"
DEST_DIR="/gpfs/scratch/ehpc42/datasets/datacomp/small_hybrid/shards"
mkdir -p "$DEST_DIR"

# Use find to safely iterate over directories without globbing issues
find "$SRC_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | while IFS= read -r -d '' shard; do
    base=$(basename "$shard")
    echo "Creating tar archive for: $base"

    # Create a tar file that contains only the contents of the shard directory.
    tar --sort=name -cf "$DEST_DIR/$base.tar" -C "$shard" .
done
