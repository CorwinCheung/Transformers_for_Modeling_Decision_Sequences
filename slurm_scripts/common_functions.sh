#!/bin/bash
# Common functions for SLURM scripts

# === Environment Setup ===
setup_environment() {
    # Set base paths
    BASE_PATH="."  # Current directory
    INFERENCE_PATH="${BASE_PATH}/transformer/inference"
    
    # Load modules
    module load python/3.12.5-fasrc01
    module load cuda/12.2.0-fasrc01
    
    # Initialize Conda/Mamba
    eval "$(conda shell.bash hook)"
    
    # Activate the environment using the full path
    mamba activate ~/.conda/envs/transformers || source ~/.conda/envs/transformers/bin/activate
    
    # Export path variables for other functions to use
    export BASE_PATH INFERENCE_PATH
}

# === Run Number Management ===
get_next_run() {
    local latest=$(ls -d ${BASE_PATH}/experiments/run_* 2>/dev/null | sort -t_ -k2 -n | tail -n1 | sed 's/.*run_//')
    if [ -z "$latest" ]; then
        echo 1
    else
        echo $((latest + 1))
    fi
}

initialize_run() {
    # Get run number or use provided override
    local override_run=$1
    
    if [ -n "$override_run" ]; then
        RUN_NUMBER=$override_run
    else
        RUN_NUMBER=$(get_next_run)
    fi
    
    echo "Starting run $RUN_NUMBER"
    export RUN_NUMBER
}

# === GPU Management ===
setup_gpu_environment() {
    # Detect number of available GPUs
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    if [ -n "$SLURM_GPUS_PER_NODE" ]; then
        NUM_GPUS=$SLURM_GPUS_PER_NODE
    fi
    echo "Detected $NUM_GPUS GPU(s) for execution"
    export NUM_GPUS
}

# Function to assign a GPU ID based on index and available GPUs
assign_gpu() {
    local idx=$1
    echo $((idx % NUM_GPUS))
}

# === Distributed Training Setup ===
setup_distributed_environment() {
    # Set up distributed training environment variables
    export MASTER_PORT=12355 # Default port that's usually available
    export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
    echo "WORLD_SIZE=$WORLD_SIZE"
    echo "MASTER_PORT=$MASTER_PORT"
    
    # Define a master address for communication between GPUs
    master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    export MASTER_ADDR=$master_addr
    echo "MASTER_ADDR=$MASTER_ADDR"
}

# === Parallel Execution ===
# Run commands in parallel on different GPUs if possible, otherwise sequentially
run_on_gpus() {
    local commands=("$@")
    local num_commands=${#commands[@]}
    
    # Array to track processes
    pids=()
    
    if [ "$NUM_GPUS" -gt 1 ]; then
        echo "Running commands in parallel across GPUs"
        
        # Process each command
        for i in "${!commands[@]}"; do
            local cmd="${commands[$i]}"
            local gpu_id=$(assign_gpu $i)
            
            # Check if we need to wait for a previous job on this GPU
            if [ -n "${pids[$gpu_id]}" ]; then
                if kill -0 ${pids[$gpu_id]} 2>/dev/null; then
                    echo "Waiting for previous job on GPU $gpu_id to finish..."
                    wait ${pids[$gpu_id]}
                fi
            fi
            
            # Run the command on the assigned GPU
            echo "Running on GPU $gpu_id: $cmd"
            CUDA_VISIBLE_DEVICES=$gpu_id $cmd &
            pids[$gpu_id]=$!
        done
        
        # Wait for all processes to complete
        wait
    else
        echo "Running commands sequentially on single GPU"
        for cmd in "${commands[@]}"; do
            CUDA_VISIBLE_DEVICES=0 $cmd
        done
    fi
}


process_checkpoints() {
    # Find checkpoint files and extract base names
    checkpoint_files=()
    for model_file in "${BASE_PATH}/experiments/run_${RUN_NUMBER}/models/model_"*"cp"*".pth"; do
        if [ -f "$model_file" ]; then
            checkpoint_files+=("$model_file")
        fi
    done
    
    if [ ${#checkpoint_files[@]} -eq 0 ]; then
        echo "No checkpoints found to process"
        return
    fi
    
    echo "Found ${#checkpoint_files[@]} checkpoints to process"
    
    # Initialize array to keep track of running processes for each GPU
    pids=()
    
    # Process each checkpoint on an available GPU, but ensure commands run in sequence
    for i in "${!checkpoint_files[@]}"; do
        model_file="${checkpoint_files[$i]}"
        model_name=$(basename "$model_file" .pth)
        
        # Assign GPU using modular arithmetic
        # Default to 1 GPU if NUM_GPUS not set
        NUM_GPUS=${NUM_GPUS:-1}
        gpu_id=$((i % NUM_GPUS))
        
        print_section_header "Processing checkpoint: $model_name on GPU $gpu_id"
        
        # If previous job was on this GPU, wait for it to finish
        if [ -n "${pids[$gpu_id]}" ]; then
            if kill -0 ${pids[$gpu_id]} 2>/dev/null; then
                echo "Waiting for previous job on GPU $gpu_id to finish..."
                wait ${pids[$gpu_id]}
            fi
        fi
        
        # Run the processing commands in sequence for this checkpoint on the assigned GPU
        (
            # These must run in sequence for each checkpoint
            CUDA_VISIBLE_DEVICES=$gpu_id python ${INFERENCE_PATH}/guess_using_transformer.py --run $RUN_NUMBER --model_name $model_name
            CUDA_VISIBLE_DEVICES=$gpu_id python ${INFERENCE_PATH}/evaluate_transformer_guess.py --run $RUN_NUMBER --model_name $model_name
            # CUDA_VISIBLE_DEVICES=$gpu_id python ${INFERENCE_PATH}/graphs_transformer_vs_ground_truth.py --run $RUN_NUMBER --model_name $model_name
        ) &
        
        # Store process ID for this GPU
        pids[$gpu_id]=$!
    done
    
    # Wait for all remaining processes to complete
    wait
}

# === Formatting and Logging ===
print_section_header() {
    local title="$1"
    printf '%*s\n' 80 '' | tr ' ' '-'
    echo -e "\n$title\n"
} 