#!/bin/bash
#SBATCH --job-name=multi-domain-test
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --output=slurm_output/%j.out
#SBATCH --error=slurm_output/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00  
#SBATCH --mem=60GB
#SBATCH --partition=kempner

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

printf '%*s\n' 80 '' | tr ' ' '-'
echo -e "\ngenerate_data.py"
python ${BASE_PATH}/synthetic_data_generation/generate_data.py --run $RUN_NUMBER --multiple_domains --num_steps 10000

printf '%*s\n' 80 '' | tr ' ' '-'
echo -e "basic_evaluation.py\n"
python ${BASE_PATH}/evaluation/basic_evaluation.py --run $RUN_NUMBER
python ${BASE_PATH}/evaluation/graphs_on_trial_block_transitions.py --run $RUN_NUMBER

printf '%*s\n' 80 '' | tr ' ' '-'
echo -e "train.py\n"

# Set up distributed training environment variables
# Use SLURM_JOB_ID to create a unique port number (to avoid conflicts)
export MASTER_PORT=$((10000 + ($SLURM_JOB_ID % 50000)))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE=$WORLD_SIZE"
echo "MASTER_PORT=$MASTER_PORT"

# Define master address
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR=$MASTER_ADDR"

# Record the start time
start_time=$(date +%s)

# Launch distributed training
srun python ${BASE_PATH}/transformer/train.py \
    --predict \
    --epochs=100 \
    --run_number $RUN_NUMBER

# Record the end time
end_time=$(date +%s)
total_time=$((end_time-start_time))
echo "Total Training Time= $total_time seconds"

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