#!/bin/bash
#SBATCH --job-name=multi-domain-test
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
DOMAIN_CONFIG=${9:-"domains.ini"}

# Export run number
export RUN_NUMBER
echo "Using run number: $RUN_NUMBER"

print_section_header "Data Generation"
python ${BASE_PATH}/synthetic_data_generation/generate_data.py --run $RUN_NUMBER --multiple_domains --num_steps_val=1_000_000 --no_overwrite --num_steps_train=$TRAIN_STEPS --config_file $DOMAIN_CONFIG

print_section_header "Basic Evaluation"
python ${BASE_PATH}/evaluation/basic_evaluation.py --run $RUN_NUMBER
python ${BASE_PATH}/evaluation/graphs_on_trial_block_transitions.py --run $RUN_NUMBER

print_section_header "Model Training"

# Setup distributed environment
setup_distributed_environment

# Record the start time
start_time=$(date +%s)

# Launch distributed training with srun
srun --cpu-bind=none python ${BASE_PATH}/transformer/train.py \
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

# Setup GPU environment for multi-domain learning
setup_gpu_environment

# Automatically remove large learning files
rm "${BASE_PATH}/experiments/run_${RUN_NUMBER}/seqs/learning_model"*"val_preds.txt" 2>/dev/null || true

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