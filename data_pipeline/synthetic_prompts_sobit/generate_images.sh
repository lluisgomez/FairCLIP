#!/bin/bash

# Job configuration
#SBATCH --nodes=1	       # each array job uses 1 node
#SBATCH --gres=gpu:4           # Leave this if GPUs are required, or reduce it if not
#SBATCH --ntasks=1             # Only one task, as you're running a single Python command
#SBATCH --cpus-per-task=80     # Request 80 CPUs (to match the required CPUs for 4 GPUs)
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=image_generation_SoBIT
#SBATCH --account=ehpc42         
#SBATCH --qos=acc_ehpc          
#SBATCH --partition=acc        
#SBATCH --time=48:00:00         # Adjust the wallclock time as needed


#SBATCH --array=0-49               # 20 array tasks: IDs 0..19


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
INPUT_DIR="/gpfs/projects/ehpc42/datasets/synth_So-B-IT/prompts"
OUTPUT_DIR="/gpfs/projects/ehpc42/datasets/synth_So-B-IT/shards"
mkdir -p "$OUTPUT_DIR"


# 1) Gather + sort all prompt files into a bash array
mapfile -t all_files < <( find "$INPUT_DIR" -maxdepth 1 -type f | sort )

# 2) Calculate chunk boundaries
task_id=$SLURM_ARRAY_TASK_ID
num_tasks=$SLURM_ARRAY_TASK_COUNT    # will be 20
num_files=${#all_files[@]}
# ceil division
chunk=$(( (num_files + num_tasks - 1) / num_tasks ))
start=$(( task_id * chunk ))
end=$(( start + chunk ))
(( end > num_files )) && end=$num_files

# 3) Slice out just this task's portion
subset=( "${all_files[@]:start:end-start}" )
echo "[$(date)] Task $task_id/$num_tasks: processing files $((start+1))-$end of $num_files"

# 4) Run your Python script on only this slice
#     We pass all file paths as individual args to --input_files
srun python /gpfs/projects/ehpc42/datasets/synth_So-B-IT/04_generate_images_multigpu.py \
     --input_files "${subset[@]}" \
     --output_dir "$OUTPUT_DIR" \
     --batch_size 32

