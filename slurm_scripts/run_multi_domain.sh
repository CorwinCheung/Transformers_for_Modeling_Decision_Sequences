#!/bin/bash
#SBATCH --job-name=multi-domain-test
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --output=slurm_output/%j.out
#SBATCH --error=slurm_output/%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4         # 4 tasks per node (one per GPU)
#SBATCH --gpus-per-node=4          # 4 gpus per node
#SBATCH --gpu-bind=per_task:1        # Bind one GPU per task
#SBATCH --cpus-per-task=2          # Each task gets 4 CPUs
#SBATCH --time=04:00:00
#SBATCH --mem=60GB
#SBATCH --partition=gpu_test

BASE_PATH="."  # Get parent directory of script location
INFERENCE_PATH="${BASE_PATH}/transformer/inference"

module load python/3.12.5-fasrc01

# Initialize Conda/Mamba properly
eval "$(conda shell.bash hook)"  # Initialize shell hook

# Activate the environment using the full path
mamba activate ~/.conda/envs/transformers || source ~/.conda/envs/transformers/bin/activate

# Get latest run number from experiments directory
get_next_run() {
    local latest=$(find experiments -maxdepth 1 -type d -regex '.*/run_[0-9]+$' 2>/dev/null | sort -t_ -k2 -n | tail -n1 | sed 's/.*run_//')
    if [ -z "$latest" ]; then
        echo 1
    else
        echo $((latest + 1))
    fi
}

RUN_NUMBER=$(get_next_run)
# RUN_NUMBER=1  # Override for testing; remove if you want automatic run numbering.
echo "Starting run $RUN_NUMBER"

# Get node information
echo "Getting node information..."
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
echo "Head node hostname: $head_node"

# Try to get IP address using more reliable methods
echo "Attempting to get head node IP..."

# Method 1: Get IP from node info
echo "Method 1: Using scontrol show node"
node_info=$(scontrol show node $head_node)
echo "Node info: $node_info"
head_node_ip=$(echo "$node_info" | grep -oP "NodeAddr=\K[0-9.]+")
echo "Method 1 result: $head_node_ip"

# Method 2: Try getent
if [ -z "$head_node_ip" ] || [[ "$head_node_ip" == *"holygpu"* ]]; then
    echo "Method 2: Using getent"
    head_node_ip=$(getent hosts $head_node | awk '{print $1}')
    echo "Method 2 result: $head_node_ip"
fi

# Method 3: Try nslookup
if [ -z "$head_node_ip" ] || [[ "$head_node_ip" == *"holygpu"* ]]; then
    echo "Method 3: Using nslookup"
    head_node_ip=$(nslookup $head_node | grep -oP 'Address: \K[0-9.]+' | tail -1)
    echo "Method 3 result: $head_node_ip"
fi

# Validate IP address format
if [[ ! $head_node_ip =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "ERROR: Failed to get valid IP address. Got: $head_node_ip"
    echo "Dumping network information for debugging:"
    echo "-------------------------------------------"
    echo "Full node info:"
    scontrol show node $head_node
    echo "-------------------------------------------"
    echo "Network interfaces:"
    srun -N1 -n1 --nodelist=$head_node ip addr show
    echo "-------------------------------------------"
    exit 1
fi

# Set a fixed port for rendezvous
RDZV_PORT=29500

# Print complete debug information
echo "Complete Debug Information:"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "Head node hostname: $head_node"
echo "Head node IP: $head_node_ip"
echo "Node list: ${nodes[@]}"
echo "Rendezvous endpoint will be: $head_node_ip:$RDZV_PORT"

# Improve NCCL performance and debugging
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_DEBUG_SUBSYS=ALL

# Add debug output for SLURM task mapping
echo "SLURM_NODEID: $SLURM_NODEID"
echo "SLURM_LOCALID: $SLURM_LOCALID"
echo "SLURM_PROCID: $SLURM_PROCID"

# Modified srun command with proper GPU binding
srun \
    --cpu-bind=none \
    --ntasks-per-node=4 \
    --gpus-per-task=1 \
    torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc-per-node=4 \
    --node-rank=$SLURM_NODEID \
    --rdzv-id=$SLURM_JOB_ID \
    --rdzv-backend=c10d \
    --rdzv-endpoint="${head_node_ip}:${RDZV_PORT}" \
    ${BASE_PATH}/transformer/train.py \
    --predict \
    --epochs=100 \
    --run_number $RUN_NUMBER

printf '%*s\n' 80 '' | tr ' ' '-'
echo -e "learning.py\n"
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
