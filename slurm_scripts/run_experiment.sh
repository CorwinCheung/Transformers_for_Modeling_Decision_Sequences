#!/bin/bash
#SBATCH --job-name=basic-workflow
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1          
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00  
#SBATCH --mem=150GB
#SBATCH --partition=kempner
#SBATCH --output=slurm_output/%j.out
#SBATCH --error=slurm_output/%j.err

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
DOMAIN_CONFIG=${9:-"domains.ini"}
DOMAIN_ID=${10:-"B"}

export DOMAIN_ID=$DOMAIN_ID
export DOMAIN_CONFIG=$DOMAIN_CONFIG
export EXPERIMENT_TYPE="basic"

# Export run number
export RUN_NUMBER
echo "Using run number: $RUN_NUMBER"

# Data generation and basic evaluation
print_section_header "Data Generation"
python ${BASE_PATH}/synthetic_data_generation/generate_data.py \
    --run $RUN_NUMBER \
    --domain_id $DOMAIN_ID \
    --num_steps_val=1_000_000 \
    --no_overwrite \
    --num_steps_train=$TRAIN_STEPS \
    --config_file "$DOMAIN_CONFIG" \
    --multiple_domains
python ${BASE_PATH}/evaluation/basic_evaluation.py --run $RUN_NUMBER
python ${BASE_PATH}/evaluation/graphs_on_trial_block_transitions.py --run $RUN_NUMBER

# Setup distributed environment
setup_distributed_environment

print_section_header "Model Training"
srun --cpu-bind=none python ${BASE_PATH}/transformer/train.py \
    --epochs=$EPOCHS \
    --run $RUN_NUMBER \
    --batch_size=$BATCH_SIZE \
    --n_layer=$N_LAYER \
    --n_head=$N_HEAD \
    --n_embd=$EMBD_DIM \
    --sequence_length=$CONTEXT_LENGTH \
    --checkpoint_interval="log"

# Setup GPU environment for inference
setup_gpu_environment

# Transformer evaluation
print_section_header "Transformer Evaluation"
python ${INFERENCE_PATH}/guess_using_transformer.py --run $RUN_NUMBER
python ${INFERENCE_PATH}/evaluate_transformer_guess.py --run $RUN_NUMBER
python ${INFERENCE_PATH}/graphs_transformer_vs_ground_truth.py --run $RUN_NUMBER
python ${BASE_PATH}/evaluation/inspect_data.py --run $RUN_NUMBER

# Process checkpoints in parallel
print_section_header "Processing Checkpoints"
process_checkpoints

# Plot checkpoint comparison
print_section_header "Checkpoint Comparison"
python ${INFERENCE_PATH}/plot_checkpoint_comparison.py --run $RUN_NUMBER

print_section_header "Learning Analysis"
LEARNING_COMMANDS=(
    "python ${INFERENCE_PATH}/learning.py --run $RUN_NUMBER --step_max=100"
    "python ${INFERENCE_PATH}/learning.py --run $RUN_NUMBER --step_max=1000"
    "python ${INFERENCE_PATH}/learning.py --run $RUN_NUMBER --step_min=1000 --step_max=10000"
    "python ${INFERENCE_PATH}/learning.py --run $RUN_NUMBER --step_min=10000 --step_max=100000"
    "python ${INFERENCE_PATH}/learning.py --run $RUN_NUMBER"
)

# Run learning commands (will run sequentially if only 1 GPU)
run_on_gpus "${LEARNING_COMMANDS[@]}"
