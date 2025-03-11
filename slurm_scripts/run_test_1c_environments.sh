#!/bin/bash
#SBATCH --job-name=multi-domain-adapt
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --output=slurm_output/%j.out
#SBATCH --error=slurm_output/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00  
#SBATCH --mem=60GB
#SBATCH --partition=kempner


# Source common functions
source "./slurm_scripts/common_functions.sh"

# Setup environment
setup_environment

# Accept parameters from master runner
RUN_NUMBER=${1:-$(get_next_run)}
N_LAYER=${2:-4}
N_HEAD=${3:-4}
EPOCHS=${4:-100}
TRAIN_STEPS=${5:-100000}
CONTEXT_LENGTH=${6:-12}
EMBD_DIM=${7:-64}
BATCH_SIZE=${8:-256}
DOMAIN_CONFIG=${9:-"vary_environment_domains.ini"}

# Export run number
export RUN_NUMBER
echo "Using run number: $RUN_NUMBER"

print_section_header "Data Generation"
# Generate training data from domain B and validation from domains A and C
python ${BASE_PATH}/synthetic_data_generation/generate_data.py \
    --run $RUN_NUMBER \
    --num_steps_train=$TRAIN_STEPS \
    --num_steps_val=100_000 \
    --no_overwrite \
    --multiple_domains \
    --config_file $DOMAIN_CONFIG

print_section_header "Basic Evaluation"
python ${BASE_PATH}/evaluation/basic_evaluation.py --run $RUN_NUMBER
python ${BASE_PATH}/evaluation/graphs_on_trial_block_transitions.py --run $RUN_NUMBER

print_section_header "Model Training"

# Record the start time
start_time=$(date +%s)

# Run training directly 
python ${BASE_PATH}/transformer/train.py \
    --n_layer=$N_LAYER \
    --n_head=$N_HEAD \
    --n_embd=$EMBD_DIM \
    --sequence_length=$CONTEXT_LENGTH \
    --epochs=$EPOCHS \
    --batch_size=$BATCH_SIZE \
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
