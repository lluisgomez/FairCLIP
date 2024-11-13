#!/bin/bash

# Job configuration
#SBATCH --nodes=32
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=image_generation 
#SBATCH --account=ehpc42         
#SBATCH --qos=acc_ehpc          
#SBATCH --partition=acc        
#SBATCH --time=4:00:00         # Adjust the wallclock time according to the scale 

#SBATCH --array=0-31           # Array job for 32 tasks (each node processes multiple files)

# Working directory and output/error files
#SBATCH -D .
#SBATCH --output=sdxlturbo_%j.out
#SBATCH --error=sdxlturbo_%j.err

#SBATCH --mail-type=all
#SBATCH --mail-user=lgomez@cvc.uab.cat

# Load necessary modules
module purge
module load oneapi hdf5 python/3.12.1

# Activate the open_clip environment 
source /gpfs/projects/ehpc42/sdxlturbo/bin/activate

# Define paths
DATA_DIR="/gpfs/scratch/ehpc42/datasets/datacomp/small/prompts"
OUTPUT_DIR="/gpfs/scratch/ehpc42/datasets/datacomp/small/images"

# Files to process in each task (335 JSON files distributed over 32 tasks, ~11 files per task)
START_INDEX=$(( SLURM_ARRAY_TASK_ID * 11 ))
END_INDEX=$(( START_INDEX + 10 ))

# List of JSON files assigned to this task
input_files=()
for i in $(seq $START_INDEX $END_INDEX); do
	    FILE_ID=$(printf "%08d" $i)
	        input_files+=("$DATA_DIR/${FILE_ID}.json")
	done

	# Run the Python script with batch size 32 and all files assigned to this task
	srun --exclusive -N1 -n1 --gpus=1 python generate_images.py --input_files "${input_files[@]}" --output_dir "$OUTPUT_DIR" --batch_size 32

