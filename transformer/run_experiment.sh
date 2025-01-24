#!/bin/bash
#SBATCH --job-name=basic-workflow
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --output=/n/home00/cberon/seq_output/%A_%a.out
#SBATCH --error=/n/home00/cberon/seq_output/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00  
#SBATCH --mem=12GB
#SBATCH --partition=kempner_requeue

module load python/3.12.8-fasrc01
mamba activate transformers

python ~/code/Transformers_for_Modeling_Decision_Sequences/transformer/train.py --predict=True --epochs=1000


# Run the training script
~/.conda/envs/transformers/bin/python train.py \
    --sequence_length $sequence_length \
    --n_layer $n_layer \
    --n_head $n_head \
    --n_embd $n_embd \
    --epochs $epochs \
    --max_lr $learning_rate \
    --task_id $SLURM_ARRAY_TASK_ID
