#!/bin/bash
#SBATCH --job-name=basic-workflow
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00  
#SBATCH --mem=40GB
#SBATCH --partition=kempner_requeue
#SBATCH --output=/n/home00/cberon/code/Transformers_for_Modeling_Decision_Sequences/slurm_scripts/slurm_output/%j.out
#SBATCH --error=/n/home00/cberon/code/Transformers_for_Modeling_Decision_Sequences/slurm_scripts/slurm_output/%j.err

BASE_PATH="/n/home00/cberon/code/Transformers_for_Modeling_Decision_Sequences"
INFERENCE_PATH="${BASE_PATH}/transformer/inference"

module load python/3.12.5-fasrc01
mamba activate transformers

# Get latest run number
get_next_run() {
    local latest=$(ls -d ${BASE_PATH}/experiments/run_* 2>/dev/null | sort -t_ -k2 -n | tail -n1 | sed 's/.*run_//')
    if [ -z "$latest" ]; then
        echo 1
    else
        echo $((latest + 1))
    fi
}

RUN_NUMBER=$(get_next_run)
echo "Starting run $RUN_NUMBER"

printf '%*s\n' 80 '' | tr ' ' '-'
echo -e "generate_data.py\n"
python ${BASE_PATH}/synthetic_data_generation/generate_data.py --run $RUN_NUMBER --context_id "A"

printf '%*s\n' 80 '' | tr ' ' '-'
echo -e "basic_evaluation.py\n"
python ${BASE_PATH}/evaluation/basic_evaluation.py --run $RUN_NUMBER

printf '%*s\n' 80 '' | tr ' ' '-'
echo -e "graphs_on_trial_block_transitions.py\n"
python ${BASE_PATH}/evaluation/graphs_on_trial_block_transitions.py --run $RUN_NUMBER

printf '%*s\n' 80 '' | tr ' ' '-'
echo -e "train.py\n"
python ${BASE_PATH}/transformer/train.py --predict=True --epochs=10000 --run $RUN_NUMBER 

printf '%*s\n' 80 '' | tr ' ' '-'
echo -e "learning.py\n"
python ${INFERENCE_PATH}/learning.py --run $RUN_NUMBER --step_max=100
python ${INFERENCE_PATH}/learning.py --run $RUN_NUMBER --step_max=1000
python ${INFERENCE_PATH}/learning.py --run $RUN_NUMBER --step_min=1000 --step_max=10000
python ${INFERENCE_PATH}/learning.py --run $RUN_NUMBER --step_min=10000 --step_max=100000
python ${INFERENCE_PATH}/learning.py --run $RUN_NUMBER

# Automatically remove large learning files
rm "${BASE_PATH}/experiments/run_${RUN_NUMBER}/seqs/learning_model"*"val_preds.txt"

printf '%*s\n' 80 '' | tr ' ' '-'
echo -e "guess_using_transformer.py\n"
python ${INFERENCE_PATH}/guess_using_transformer.py --run $RUN_NUMBER

printf '%*s\n' 80 '' | tr ' ' '-'
echo -e "evaluate_transformer_guess.py\n"
python ${INFERENCE_PATH}/evaluate_transformer_guess.py --run $RUN_NUMBER

printf '%*s\n' 80 '' | tr ' ' '-'
echo -e "graphs_transformer_vs_ground_truth.py\n"
python ${INFERENCE_PATH}/graphs_transformer_vs_ground_truth.py --run $RUN_NUMBER
