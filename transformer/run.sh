#!/bin/bash
#SBATCH --job-name=gpt-hparam-sweep
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --output=/n/holyscratch01/bsabatini_lab/Users/ccheung/output/%A_%a.out
#SBATCH --error=/n/holyscratch01/bsabatini_lab/Users/ccheung/error/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:00:00  # Adjust based on your needs
#SBATCH --mem=12GB
#SBATCH --partition=kempner
#SBATCH --array=1-1261%11

~/.conda/envs/transformers/bin/python -c "import torch; print('Torch Version:', torch.__version__)"

# Debugging: Check Python path and environment
~/.conda/envs/transformers/bin/python -c "import sys; print('Python Path:', sys.path)"

# Read hyperparameters from CSV
IFS=',' read -r sequence_length n_layer n_head n_embd epochs learning_rate <<< $(tail -n +2 cleaned_hyperparameters.csv | sed -n "$SLURM_ARRAY_TASK_ID"p)

# Debug parsed parameters
echo "Parsed Parameters:"
echo "Sequence Length: $sequence_length, Layers: $n_layer, Heads: $n_head, Embedding: $n_embd, Epochs: $epochs, Learning Rate: $learning_rate"

# Run the training script
~/.conda/envs/transformers/bin/python train.py \
    --sequence_length $sequence_length \
    --n_layer $n_layer \
    --n_head $n_head \
    --n_embd $n_embd \
    --epochs $epochs \
    --max_lr $learning_rate \
    --task_id $SLURM_ARRAY_TASK_ID
