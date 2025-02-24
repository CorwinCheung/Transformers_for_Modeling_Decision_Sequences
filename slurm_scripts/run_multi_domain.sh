#!/bin/bash
#SBATCH --job-name=multi-domain-test
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --output=slurm_output/%j.out
#SBATCH --error=slurm_output/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00  
#SBATCH --mem=120GB
#SBATCH --partition=kempner_requeue

BASE_PATH="."  # Get parent directory of script location
INFERENCE_PATH="${BASE_PATH}/transformer/inference"

module load python/3.12.5-fasrc01
# module load cuda/12.2.0-fasrc01

# Initialize Conda/Mamba properly
eval "$(conda shell.bash hook)"  # Initialize shell hook

# Activate the environment using the full path
mamba activate ~/.conda/envs/transformers || source ~/.conda/envs/transformers/bin/activate

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
RUN_NUMBER=20  # Override for testing; remove if you want automatic run numbering.
echo "Starting run $RUN_NUMBER"

# Reduced num_steps to 1000 for testing
python ${BASE_PATH}/synthetic_data_generation/generate_data.py --run $RUN_NUMBER --multiple_domains --no_overwrite
python ${BASE_PATH}/evaluation/basic_evaluation.py --run $RUN_NUMBER
python ${BASE_PATH}/evaluation/graphs_on_trial_block_transitions.py --run $RUN_NUMBER

# Reduced epochs to 10 for testing
python ${BASE_PATH}/transformer/train.py --predict --epochs=10000 --run $RUN_NUMBER --eval_interval=10000 --checkpoint_interval=1000 --enforce_data_epochs

# Only run first two step ranges for testing
python ${INFERENCE_PATH}/learning.py --run $RUN_NUMBER --step_min=1000 --step_max=10000
python ${INFERENCE_PATH}/learning.py --run $RUN_NUMBER --step_min=10000 --step_max=100000
python ${INFERENCE_PATH}/learning.py --run $RUN_NUMBER --step_min=100000 --step_max=1000000
# python ${INFERENCE_PATH}/learning.py --run $RUN_NUMBER --step_min=1000000
python ${INFERENCE_PATH}/learning.py --run $RUN_NUMBER --step_min=0 # all data

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

# Find checkpoint files and extract base names
for model_file in "${BASE_PATH}/experiments/run_${RUN_NUMBER}/models/model_"*"cp"*".pth"; do
    if [ -f "$model_file" ]; then
        # Extract basename and remove .pth extension
        model_name=$(basename "$model_file" .pth)
        printf '%*s\n' 80 '' | tr ' ' '-'
        echo -e "\nProcessing checkpoint: $model_name"
        python ${INFERENCE_PATH}/guess_using_transformer.py --run $RUN_NUMBER --model_name "$model_name"
        python ${INFERENCE_PATH}/evaluate_transformer_guess.py --run $RUN_NUMBER --model_name "$model_name"
        python ${INFERENCE_PATH}/graphs_transformer_vs_ground_truth.py --run $RUN_NUMBER --model_name "$model_name"
    fi
done

# Must follow checkpoint predictions
printf '%*s\n' 80 '' | tr ' ' '-'
echo -e "plot_checkpoint_comparison.py\n"
python ${INFERENCE_PATH}/plot_checkpoint_comparison.py --run $RUN_NUMBER
