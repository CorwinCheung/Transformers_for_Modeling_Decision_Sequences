#!/bin/bash
# SBATCH --job-name=gpt-hparam-sweep
# SBATCH --account=kempner_bsabatini_lab
# SBATCH --output=%A_%a.out
# SBATCH --nodes=1
# SBATCH --ntasks-per-node=1
# SBATCH --gpus-per-node=1
# SBATCH --cpus-per-task=8
# SBATCH --time=1:00:00  # Adjust based on your needs
# SBATCH --mem=12GB
# SBATCH --partition=kempner
# SBATCH --array=1-1%2  # Adjust based on the number of hyperparameter combinations

module load cuda/11.7
module load anaconda
source activate transformers  # Replace with your conda environment

# Read hyperparameters from CSV
IFS=',' read -r sequence_length n_layer n_head n_embd epochs learning_rate <<< $(sed -n "$((SLURM_ARRAY_TASK_ID + 1))"p cleaned_hyperparameters.csv)

# Run your training script with hyperparameters
python train.py \
    --sequence_length $sequence_length \
    --n_layer $n_layer \
    --n_head $n_head \
    --n_embd $n_embd \
    --epochs $epochs \
    --max_lr $learning_rate \
    --task_id $SLURM_ARRAY_TASK_ID