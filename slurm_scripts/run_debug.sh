#!/bin/bash
#SBATCH --job-name=basic-workflow
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4          
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00  
#SBATCH --mem=80GB
#SBATCH --partition=kempner
#SBATCH --output=slurm_output/%j.out
#SBATCH --error=slurm_output/%j.err

# Source common functions
source "./slurm_scripts/common_functions.sh"

# Setup environment
setup_environment

# Initialize run number
initialize_run
# initialize_run 35  # Uncomment to override run number

# Data generation and basic evaluation
python ${BASE_PATH}/synthetic_data_generation/generate_data.py --run $RUN_NUMBER --domain_id "A" --num_steps_train=10000 --num_steps_val=100000 --no_overwrite
python ${BASE_PATH}/evaluation/basic_evaluation.py --run $RUN_NUMBER
python ${BASE_PATH}/evaluation/graphs_on_trial_block_transitions.py --run $RUN_NUMBER

# Setup distributed environment
setup_distributed_environment

# Run distributed training
srun python ${BASE_PATH}/transformer/train.py --epochs=1000 --run $RUN_NUMBER --checkpoint_interval=100 --eval_interval=1000 --predict # --enforce_data_epochs

# Setup GPU environment for inference
setup_gpu_environment

# Define learning commands
LEARNING_COMMANDS=(
    "python ${INFERENCE_PATH}/learning.py --run $RUN_NUMBER --step_max=100"
    "python ${INFERENCE_PATH}/learning.py --run $RUN_NUMBER --step_max=1000"
    "python ${INFERENCE_PATH}/learning.py --run $RUN_NUMBER --step_min=1000 --step_max=10000"
    "python ${INFERENCE_PATH}/learning.py --run $RUN_NUMBER --step_min=10000 --step_max=100000"
    "python ${INFERENCE_PATH}/learning.py --run $RUN_NUMBER"
    "python ${INFERENCE_PATH}/guess_using_transformer.py --run $RUN_NUMBER" # tacking this on because it can only speed things up
)

# Run learning commands in parallel where possible
run_on_gpus "${LEARNING_COMMANDS[@]}"

# Transformer evaluation
python ${INFERENCE_PATH}/evaluate_transformer_guess.py --run $RUN_NUMBER
python ${INFERENCE_PATH}/graphs_transformer_vs_ground_truth.py --run $RUN_NUMBER
python ${BASE_PATH}/evaluation/inspect_data.py --run $RUN_NUMBER

# Process checkpoints in parallel
print_section_header "Processing checkpoints"
process_checkpoints

# Plot checkpoint comparison
print_section_header "plot_checkpoint_comparison.py"
python ${INFERENCE_PATH}/plot_checkpoint_comparison.py --run $RUN_NUMBER