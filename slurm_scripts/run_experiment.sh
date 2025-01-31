#!/bin/bash
#SBATCH --job-name=basic-workflow
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --output=/n/home00/cberon/code/Transformers_for_Modeling_Decision_Sequences/seq_output/%j.out
#SBATCH --error=/n/home00/cberon/code/Transformers_for_Modeling_Decision_Sequences/seq_output/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00  
#SBATCH --mem=12GB
#SBATCH --partition=kempner_requeue

module load python/3.12.5-fasrc01
mamba activate transformers

# Get latest run number from experiments directory
get_next_run() {
    local latest=$(ls -d experiments/run_* 2>/dev/null | sort -t_ -k2 -n | tail -n1 | sed 's/.*run_//')
    if [ -z "$latest" ]; then
        echo 1
    else
        echo $((latest + 1))
    fi
}

RUN_NUMBER=$(get_next_run)
echo "Starting run $RUN_NUMBER"

python ~/code/Transformers_for_Modeling_Decision_Sequences/synthetic_data_generation/generate_data.py --run $RUN_NUMBER --context_id "A"
python ~/code/Transformers_for_Modeling_Decision_Sequences/evaluation/basic_evaluation.py --run $RUN_NUMBER
python ~/code/Transformers_for_Modeling_Decision_Sequences/evaluation/graphs_on_trial_block_transitions.py --run $RUN_NUMBER

python ~/code/Transformers_for_Modeling_Decision_Sequences/transformer/train.py --predict=True --epochs=10000 --run $RUN_NUMBER 
python ~/code/Transformers_for_Modeling_Decision_Sequences/transformer/inference/learning.py --step_cutoff=100 --run $RUN_NUMBER
python ~/code/Transformers_for_Modeling_Decision_Sequences/transformer/inference/learning.py --step_cutoff=1000 --run $RUN_NUMBER
python ~/code/Transformers_for_Modeling_Decision_Sequences/transformer/inference/learning.py --run $RUN_NUMBER

# Automatically remove large learning files
rm "/n/home00/cberon/code/Transformers_for_Modeling_Decision_Sequences/experiments/run_${RUN_NUMBER}/learning_model"*"val_preds.txt"

python ~/code/Transformers_for_Modeling_Decision_Sequences/transformer/inference/guess_using_transformer.py --run $RUN_NUMBER
python ~/code/Transformers_for_Modeling_Decision_Sequences/transformer/inference/evaluate_transformer_guess.py --run $RUN_NUMBER
python ~/code/Transformers_for_Modeling_Decision_Sequences/transformer/inference/graphs_transformer_vs_ground_truth.py --run $RUN_NUMBER