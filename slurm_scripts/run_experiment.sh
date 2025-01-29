#!/bin/bash
#SBATCH --job-name=basic-workflow
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --output=/n/home00/cberon/code/Transformers_for_Modeling_Decision_Sequences/seq_output/%j.out
#SBATCH --error=/n/home00/cberon/code/Transformers_for_Modeling_Decision_Sequences/seq_output/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00  
#SBATCH --mem=12GB
#SBATCH --partition=kempner_requeue

module load python/3.12.5-fasrc01
mamba activate transformers

# python ~/code/Transformers_for_Modeling_Decision_Sequences/synthetic_data_generation/generate_data.py --run=3 --overwrite=True
# python ~/code/Transformers_for_Modeling_Decision_Sequences/evaluation/basic_evaluation.py --run=3

# python ~/code/Transformers_for_Modeling_Decision_Sequences/transformer/train.py --predict=True --epochs=1000 --run=3
# python ~/code/Transformers_for_Modeling_Decision_Sequences/transformer/inference/learning.py --run=3

python ~/code/Transformers_for_Modeling_Decision_Sequences/synthetic_data_generation/generate_data.py
python ~/code/Transformers_for_Modeling_Decision_Sequences/evaluation/basic_evaluation.py
python ~/code/Transformers_for_Modeling_Decision_Sequences/evaluation/graphs_on_trial_block_transitions.py


python ~/code/Transformers_for_Modeling_Decision_Sequences/transformer/train.py --predict=True --epochs=100
# python ~/code/Transformers_for_Modeling_Decision_Sequences/transformer/inference/learning.py --step_cutoff=100
# python ~/code/Transformers_for_Modeling_Decision_Sequences/transformer/inference/learning.py --step_cutoff=1000
# python ~/code/Transformers_for_Modeling_Decision_Sequences/transformer/inference/learning.py


# python ~/code/Transformers_for_Modeling_Decision_Sequences/transformer/train.py --predict=True --epochs=1000
# python ~/code/Transformers_for_Modeling_Decision_Sequences/transformer/inference/learning.py

# python ~/code/Transformers_for_Modeling_Decision_Sequences/transformer/train.py --predict=True --epochs=10000
# python ~/code/Transformers_for_Modeling_Decision_Sequences/transformer/inference/learning.py