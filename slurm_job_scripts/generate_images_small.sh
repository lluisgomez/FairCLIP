#!/bin/bash

# Job configuration
#SBATCH --nodes=1
#SBATCH --gres=gpu:4           # Leave this if GPUs are required, or reduce it if not
#SBATCH --ntasks=1             # Only one task, as you're running a single Python command
#SBATCH --cpus-per-task=80     # Request 80 CPUs (to match the required CPUs for 4 GPUs)
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=image_generation 
#SBATCH --account=ehpc42         
#SBATCH --qos=acc_ehpc          
#SBATCH --partition=acc        
#SBATCH --time=4:00:00         # Adjust the wallclock time as needed

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
INPUT_DIR="/gpfs/scratch/ehpc42/datasets/datacomp/small_hybrid/prompts"
OUTPUT_DIR="/gpfs/scratch/ehpc42/datasets/datacomp/small_hybrid/tmp"

# Run the Python script
srun python /gpfs/projects/ehpc42/code/generate_images_multigpu.py --input_files "$INPUT_DIR" --output_dir "$OUTPUT_DIR" --batch_size 32

