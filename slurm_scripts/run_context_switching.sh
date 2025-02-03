#!/bin/bash
#SBATCH --job-name=multi-context
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --output=/n/home00/cberon/code/Transformers_for_Modeling_Decision_Sequences/seq_output/%j.out
#SBATCH --error=/n/home00/cberon/code/Transformers_for_Modeling_Decision_Sequences/seq_output/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00  
#SBATCH --mem=6GB
#SBATCH --partition=kempner_requeue

BASE_PATH="/n/home00/cberon/code/Transformers_for_Modeling_Decision_Sequences"

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

python ${BASE_PATH}/synthetic_data_generation/generate_data.py --run $RUN_NUMBER --multiple_contexts=True

python ${BASE_PATH}/evaluation/basic_evaluation.py --run $RUN_NUMBER
python ${BASE_PATH}/evaluation/graphs_on_trial_block_transitions.py --run $RUN_NUMBER

python ${BASE_PATH}/transformer/train.py --predict=True --epochs=100000 --run $RUN_NUMBER 
python ${BASE_PATH}/transformer/inference/learning.py --step_cutoff=100 --run $RUN_NUMBER
python ${BASE_PATH}/transformer/inference/learning.py --step_cutoff=1000 --run $RUN_NUMBER
python ${BASE_PATH}/transformer/inference/learning.py --run $RUN_NUMBER --step_cutoff=10000
python ${BASE_PATH}/transformer/inference/learning.py --run $RUN_NUMBER --step_cutoff=100000
python ${BASE_PATH}/transformer/inference/learning.py --run $RUN_NUMBER --step_cutoff=1000000
python ${BASE_PATH}/transformer/inference/learning.py --run $RUN_NUMBER
# Automatically remove large learning files
rm "${BASE_PATH}/experiments/run_${RUN_NUMBER}/learning_model"*"val_preds.txt"

python ${BASE_PATH}/transformer/inference/guess_using_transformer.py --run $RUN_NUMBER
python ${BASE_PATH}/transformer/inference/evaluate_transformer_guess.py --run $RUN_NUMBER
python ${BASE_PATH}/transformer/inference/graphs_transformer_vs_ground_truth.py --run $RUN_NUMBER
python ${BASE_PATH}/transformer/inference/plot_checkpoint_comparison.py --run $RUN_NUMBER

# Find checkpoint files and extract base names
for model_file in "${BASE_PATH}/experiments/run_${RUN_NUMBER}/model_"*"cp"*".pth"; do
    if [ -f "$model_file" ]; then
        # Extract basename and remove .pth extension
        model_name=$(basename "$model_file" .pth)
        echo -e "\nProcessing checkpoint: $model_name"
        python ${BASE_PATH}/transformer/inference/guess_using_transformer.py --run $RUN_NUMBER --model "$model_name"
        python ${BASE_PATH}/inference/evaluate_transformer_guess.py --run $RUN_NUMBER --model "$model_name"
        python ${BASE_PATH}/transformer/inference/graphs_transformer_vs_ground_truth.py --run $RUN_NUMBER --model "$model_name"
    fi
done

