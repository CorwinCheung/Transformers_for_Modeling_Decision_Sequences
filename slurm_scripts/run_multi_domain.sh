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

BASE_PATH="."  # Get parent directory of script location
INFERENCE_PATH="${BASE_PATH}/transformer/inference"

module load python/3.12.5-fasrc01

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
# RUN_NUMBER=1  # Override for testing; remove if you want automatic run numbering.
echo "Starting run $RUN_NUMBER"

printf '%*s\n' 80 '' | tr ' ' '-'
echo -e "\ngenerate_data.py"
python ${BASE_PATH}/synthetic_data_generation/generate_data.py --run $RUN_NUMBER --multiple_domains --num_steps 1000

printf '%*s\n' 80 '' | tr ' ' '-'
echo -e "basic_evaluation.py\n"
python ${BASE_PATH}/evaluation/basic_evaluation.py --run $RUN_NUMBER

printf '%*s\n' 80 '' | tr ' ' '-'
echo -e "graphs_on_trial_block_transitions.py\n"
python ${BASE_PATH}/evaluation/graphs_on_trial_block_transitions.py --run $RUN_NUMBER

printf '%*s\n' 80 '' | tr ' ' '-'
echo -e "train.py\n"

# Set up distributed training environment variables
export MASTER_PORT=29500  # Use standard PyTorch DDP port
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE=$WORLD_SIZE"
echo "MASTER_PORT=$MASTER_PORT"

# === Fix IPv4/IPv6 socket issues ===
# Force IPv4 for all connections
export NCCL_SOCKET_FAMILY=IPv4

# Get direct IPv4 address for master node
MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_IP=$(srun --nodes=1 --ntasks=1 -w $MASTER_NODE bash -c "hostname -I | awk '{print \$1}'")
export MASTER_ADDR=$MASTER_IP
echo "Using MASTER_ADDR=$MASTER_IP (direct IPv4 address)"

# Force Python to prefer IPv4
cat > /tmp/force_ipv4_${SLURM_JOB_ID}.py << 'EOF'
import socket
original_getaddrinfo = socket.getaddrinfo

def getaddrinfo_ipv4(*args, **kwargs):
    responses = original_getaddrinfo(*args, **kwargs)
    return [res for res in responses if res[0] == socket.AF_INET]

socket.getaddrinfo = getaddrinfo_ipv4
EOF

# Create a wrapper script that forces IPv4
cat > /tmp/run_with_ipv4_${SLURM_JOB_ID}.py << 'EOF'
import sys
import socket

# Force IPv4
original_getaddrinfo = socket.getaddrinfo
def getaddrinfo_ipv4(*args, **kwargs):
    responses = original_getaddrinfo(*args, **kwargs)
    return [res for res in responses if res[0] == socket.AF_INET]
socket.getaddrinfo = getaddrinfo_ipv4

# Execute the original script
script_path = sys.argv[1]
sys.argv = sys.argv[1:]
with open(script_path) as f:
    exec(f.read())
EOF

# Test the connection
echo "Testing connection to master node..."
srun --nodes=${SLURM_NNODES} --ntasks=${SLURM_NNODES} --ntasks-per-node=1 \
     bash -c "echo \"Node \$(hostname): Checking ping to $MASTER_IP\"; ping -c 1 $MASTER_IP || echo 'Ping failed'"

# Fix IPv4/IPv6 protocol issues
# Force IPv4 for all connections
export NCCL_SOCKET_IFNAME=ib0  # InfiniBand interface
export GLOO_SOCKET_IFNAME=ib0

# Disable IPv6 for Python
export PYTHONUNBUFFERED=1  # Ensure output is not buffered
export PYTHONHTTPSVERIFY=0  # Disable HTTPS verification if needed

# Force PyTorch to use IPv4
cat > /tmp/force_ipv4_${SLURM_JOB_ID}.py << 'EOF'
import socket
import sys

# Force IPv4
_orig_getaddrinfo = socket.getaddrinfo
def _getaddrinfo_ipv4(*args, **kwargs):
    responses = _orig_getaddrinfo(*args, **kwargs)
    return [res for res in responses if res[0] == socket.AF_INET]
socket.getaddrinfo = _getaddrinfo_ipv4

# Continue with the real script
sys.path.insert(0, '.')
EOF

# Define wrapper command
PYTHON_CMD="-c \"import sys; exec(open('/tmp/force_ipv4_${SLURM_JOB_ID}.py').read()); exec(open(sys.argv[1]).read())\" ${BASE_PATH}/transformer/train.py"

# Create a file to synchronize the start of the main process
SYNC_FILE="/tmp/ddp_sync_${SLURM_JOB_ID}"
rm -f $SYNC_FILE
touch $SYNC_FILE

# Set TCP timeout parameters
export NCCL_SOCKET_TIMEOUT=300  # 5 minutes in seconds

# Set maximum retries for NCCL operations
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=23  # In seconds
export NCCL_DEBUG_SUBSYS=ALL

# For the coordination script, run only on the first node
srun --nodes=1 --ntasks=1 --export=ALL python - <<EOF
import os
import socket
import time

# Create a marker to indicate master is ready
ready_file = "${SYNC_FILE}"
hostname = socket.gethostname()
master_addr = os.environ.get('MASTER_ADDR')
master_port = os.environ.get('MASTER_PORT')

print(f"Coordination script running on {hostname}")
print(f"MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")

# Make sure the master node is ready to accept connections
with open(ready_file, 'w') as f:
    f.write(f"{hostname} ready at {time.time()}")

print("Coordination complete. Main job can start.")
EOF

# Wait briefly to ensure the file is written
sleep 5

# For the main job, explicitly specify nodes, tasks, and GPUs
echo "Starting main distributed training job..."
srun --nodes=${SLURM_NNODES} \
     --ntasks-per-node=${SLURM_NTASKS_PER_NODE} \
     --export=ALL \
     python ${PYTHON_CMD} \
     --predict \
     --epochs=10 \
     --run_number $RUN_NUMBER

# Record the end time
end_time=$(date +%s)
total_time=$((end_time-start_time))
echo "Total Training Time= $total_time seconds"

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