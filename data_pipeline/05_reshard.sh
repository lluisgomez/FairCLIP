#!/bin/bash

# Job configuration
#SBATCH --nodes=1	       # each array job uses 1 node
#SBATCH --gres=gpu:4           # Leave this if GPUs are required, or reduce it if not
#SBATCH --ntasks=1             # Only one task, as you're running a single Python command
#SBATCH --cpus-per-task=80     # Request 80 CPUs (to match the required CPUs for 4 GPUs)
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=reshard_large 
#SBATCH --account=ehpc42         
#SBATCH --qos=acc_ehpc          
#SBATCH --partition=acc        
#SBATCH --time=14:00:00         # Adjust the wallclock time as needed

# Load necessary modules
module purge
module load oneapi hdf5 python/3.12.1

# Define the source directory containing the modified shards and the destination for the new tar files
SRC_DIR="/gpfs/scratch/ehpc42/datasets/datacomp/large_filtered/shards"
DEST_DIR="/gpfs/scratch/ehpc42/datasets/datacomp/large_hybrid/shards"
EDIT_DIR="/gpfs/scratch/ehpc42/datasets/datacomp/large_hybrid/edits"

start=0
end=43027
batch_size=80

for ((i=start; i<=end; i+=batch_size)); do
    echo "Launching batch starting from $i"

    for ((j=i; j<i+batch_size && j<=end; j++)); do
        base=$(printf "%08d" "$j")
        echo "  -> Starting $base"

        python /gpfs/projects/ehpc42/code/05_reshard.py \
            --input "$SRC_DIR/$base.tar" \
            --output "$DEST_DIR/$base.tar" \
            --edits "$EDIT_DIR/$base" &
    done

    wait
    echo "Batch starting at $i completed."
done

echo "All tasks completed."

