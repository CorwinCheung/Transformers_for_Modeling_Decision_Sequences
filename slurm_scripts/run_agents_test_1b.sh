#!/bin/bash
#SBATCH --job-name=multi-domain-adapt
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --output=slurm_output/%j.out
#SBATCH --error=slurm_output/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00  
#SBATCH --mem=60GB
#SBATCH --partition=kempner


# Source common functions
source "./slurm_scripts/common_functions.sh"

# Setup environment
setup_environment

# Initialize run number (optionally override)
initialize_run

print_section_header "Data Generation"

python ${BASE_PATH}/synthetic_data_generation/generate_data.py \
    --run $RUN_NUMBER \
    --num_steps_train=1_000_000 \
    --num_steps_val=100_000 \
    --no_overwrite \
    --multiple_domains \
    --config_file evidence_based_agent_domains.ini

print_section_header "Basic Evaluation"
python ${BASE_PATH}/evaluation/basic_evaluation.py --run $RUN_NUMBER
python ${BASE_PATH}/evaluation/graphs_on_trial_block_transitions.py --run $RUN_NUMBER

print_section_header "Model Training"

# Record the start time
start_time=$(date +%s)

# Run training directly 
python ${BASE_PATH}/transformer/train.py \
    --n_layer=8 \
    --sequence_length=6 \
    --n_head=8 \
    --n_embd=64 \
    --epochs=100 \
    --run_number $RUN_NUMBER

# Record the end time
end_time=$(date +%s)
total_time=$((end_time-start_time))
echo "Total Training Time= $total_time seconds"

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
