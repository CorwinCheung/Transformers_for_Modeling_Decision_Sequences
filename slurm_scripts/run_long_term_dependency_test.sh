#!/bin/bash
#SBATCH --job-name=long-term-test
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1          
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00  
#SBATCH --mem=80GB
#SBATCH --partition=kempner

# Source common functions
source "./slurm_scripts/common_functions.sh"

# Setup environment
setup_environment

# Accept parameters from master runner
RUN_NUMBER=${1:-$(get_next_run)}
N_LAYER=${2:-1}
N_HEAD=${3:-1}
EPOCHS=${4:-100}
TRAIN_STEPS=${5:-100000}
CONTEXT_LENGTH=${6:-12}
EMBD_DIM=${7:-64}
BATCH_SIZE=${8:-256}
DOMAIN_CONFIG=${9:-"three_domains.ini"}

# Export run number
export RUN_NUMBER
echo "Using run number: $RUN_NUMBER"

# Data generation with long-term dependencies
print_section_header "Data Generation"
python ${BASE_PATH}/synthetic_data_generation/generate_long_term_data.py \
    --run $RUN_NUMBER \
    --num_steps_train=$TRAIN_STEPS \
    --num_steps_val=1_000_000 \
    --train_pattern=11 \
    --val_pattern=13 \
    --no_overwrite \
    --config_file "$DOMAIN_CONFIG"

# Setup distributed environment
setup_distributed_environment

print_section_header "Model Training"
srun --cpu-bind=none python ${BASE_PATH}/transformer/train.py \
    --predict \
    --epochs=$EPOCHS \
    --run $RUN_NUMBER \
    --batch_size=$BATCH_SIZE \
    --n_layer=$N_LAYER \
    --n_head=$N_HEAD \
    --n_embd=$EMBD_DIM \
    --sequence_length=$CONTEXT_LENGTH

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