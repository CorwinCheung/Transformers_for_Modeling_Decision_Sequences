#!/bin/bash
#SBATCH --job-name=gpt-hparam-sweep
#SBATCH --account=kempner_dev
#SBATCH --output=%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00  # Adjust based on your needs
#SBATCH --mem=12GB
#SBATCH --partition=kempner
#SBATCH --array=1-3%2  # Adjust based on the number of hyperparameter combinations

module load cuda/11.7
module load anaconda
source activate <your-conda-env>  # Replace with your conda environment

# Read hyperparameters from CSV
IFS=',' read -r learning_rate batch_size sequence_length n_layer n_head n_embd max_steps <<< $(sed -n "$((SLURM_ARRAY_TASK_ID + 1))"p hyperparameters.csv)

# Run your training script with hyperparameters
python train.py \
    --learning_rate $learning_rate \
    --batch_size $batch_size \
    --sequence_length $sequence_length \
    --n_layer $n_layer \
    --n_head $n_head \
    --n_embd $n_embd \
    --max_steps $max_steps \
    --task_id $SLURM_ARRAY_TASK_ID
