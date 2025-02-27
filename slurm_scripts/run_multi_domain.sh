#!/bin/bash
#SBATCH --job-name=multi-domain-test
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --output=slurm_output/%j.out
#SBATCH --error=slurm_output/%j.err
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00  
#SBATCH --mem=60GB
#SBATCH --partition=kempner

# Source common functions
source "$(dirname "$0")/common_functions.sh"

# Setup environment
setup_environment

# Initialize run number (optionally override)
initialize_run

print_section_header "Data Generation"
python ${BASE_PATH}/synthetic_data_generation/generate_data.py --run $RUN_NUMBER --multiple_domains --num_steps 1000000

print_section_header "Basic Evaluation"
python ${BASE_PATH}/evaluation/basic_evaluation.py --run $RUN_NUMBER
python ${BASE_PATH}/evaluation/graphs_on_trial_block_transitions.py --run $RUN_NUMBER

print_section_header "Model Training"

# Setup distributed environment
setup_distributed_environment

# Record the start time
start_time=$(date +%s)

# Launch distributed training with srun
srun python ${BASE_PATH}/transformer/train.py \
    --predict \
    --epochs=100 \
    --run_number $RUN_NUMBER

# Record the end time
end_time=$(date +%s)
total_time=$((end_time-start_time))
echo "Total Training Time= $total_time seconds"

# Setup GPU environment for multi-domain learning
setup_gpu_environment

# Define learning commands for multi-domain
print_section_header "Learning Analysis"
LEARNING_COMMANDS=(
    "python ${INFERENCE_PATH}/learning.py --run $RUN_NUMBER --step_min=1000 --step_max=10000"
    "python ${INFERENCE_PATH}/learning.py --run $RUN_NUMBER --step_min=10000 --step_max=100000"
    "python ${INFERENCE_PATH}/learning.py --run $RUN_NUMBER --step_min=100000 --step_max=1000000"
    "python ${INFERENCE_PATH}/learning.py --run $RUN_NUMBER --step_min=0" # all data
)

# Run learning commands
run_on_gpus "${LEARNING_COMMANDS[@]}"

# Automatically remove large learning files
rm "${BASE_PATH}/experiments/run_${RUN_NUMBER}/seqs/learning_model"*"val_preds.txt"

print_section_header "Transformer Evaluation"
python ${INFERENCE_PATH}/guess_using_transformer.py --run $RUN_NUMBER
python ${INFERENCE_PATH}/evaluate_transformer_guess.py --run $RUN_NUMBER
python ${INFERENCE_PATH}/graphs_transformer_vs_ground_truth.py --run $RUN_NUMBER

# Process checkpoints in parallel
print_section_header "Processing Checkpoints"
process_checkpoints

# Plot checkpoint comparison
print_section_header "Checkpoint Comparison"
python ${INFERENCE_PATH}/plot_checkpoint_comparison.py --run $RUN_NUMBER