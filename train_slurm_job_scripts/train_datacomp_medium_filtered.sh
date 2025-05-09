#!/bin/bash

# Job configuration
#SBATCH --nodes=32
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=open_clip_medium_filtered
#SBATCH --account=ehpc42         
#SBATCH --qos=acc_ehpc          
#SBATCH --partition=acc        
#SBATCH --time=4:00:00         # Adjust the wallclock time according to the scale 

# Working directory and output/error files
#SBATCH -D .
#SBATCH --output=open_clip_%j.out
#SBATCH --error=open_clip_%j.err

#SBATCH --mail-type=all
#SBATCH --mail-user=lgomez@cvc.uab.cat

# Load necessary modules
module purge
module load oneapi hdf5 python/3.12.1

# Activate the open_clip environment 
source /gpfs/projects/ehpc42/open_clip/bin/activate

# Environment variables
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=$(echo "$HOSTNAMES" | wc -l)
echo "Number of nodes: $COUNT_NODE"
echo "Hostnames: $HOSTNAMES"
export CUDA_VISIBLE_DEVICES=0,1,2,3


# Ensure srun uses the same number of CPUs per task as SLURM
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# Navigate to the datacomp directory
cd /gpfs/projects/ehpc42/code/datacomp/

# Paths and settings
DATA_PATH="/gpfs/scratch/ehpc42/datasets/datacomp/medium_filtered/shards/"
SCALE="medium"
SEED=0
OUTPUT_DIR="/gpfs/projects/ehpc42/jobs_output"
NUM_CHECKPOINTS=8
EXP_NAME="datacomp-scale-${SCALE}-filtered-seed${SEED}"
PRECISION="amp"                           # Use "amp_bfloat16" for xlarge scale if needed

if [ "$SCALE" == "xlarge" ]; then
    PRECISION="amp_bfloat16"
fi

# Run the training job
srun --comment "<comment>" --cpu_bind=v --accel-bind=gn python train.py \
    --scale ${SCALE} \
    --data_dir ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --exp_name ${EXP_NAME} \
    --precision ${PRECISION} \
    --num_checkpoints ${NUM_CHECKPOINTS} \
    --seed ${SEED} \
    --accum_freq 1

