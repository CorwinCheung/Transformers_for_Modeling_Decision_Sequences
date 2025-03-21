source "./slurm_scripts/common_functions.sh"
setup_environment

# Parameters for experiment sweeps (or single)
LAYERS_ARRAY=(4)
HEADS_ARRAY=(4)
EPOCHS_ARRAY=(100)
TRAIN_STEPS_ARRAY=(100000)
# CONTEXT_LENGTH_ARRAY=(3 6 24 36)
# EMBD_DIM_ARRAY=(4)
CONTEXT_LENGTH_ARRAY=(24)
EMBD_DIM_ARRAY=(64)
BATCH_SIZE_ARRAY=(256)
DOMAIN_CONFIG_ARRAY=("sticky_unsticky_agent_domains.ini")
EXPERIMENT_TYPE="agents_test"  # define the experiment you are running

# Options are:
#   "basic": run_experiment.sh
#   "multi_domain": run_test_1a_multi_domain.sh
#   "agents_test": run_test_1b_agents.sh
#   "environment_test": run_test_1c_environments.sh

TRACKER_FILE="tracker.txt"

# Initialize starting run number - scan existing runs once at the beginning
# initialize_run
NEXT_RUN_NUMBER=3

# Function to submit a single experiment job
submit_experiment() {
    local experiment_name=$1
    local layers=$2
    local heads=$3
    local epochs=$4
    local train_steps=$5
    local context_length=$6
    local embd_dim=$7
    local batch_size=$8
    local domain_config=$9
    local run_number=${10}
    
    # Create a temporary script for this specific experiment
    temp_script=$(mktemp)
    
    # Write SLURM directives to the temp script
    cat > $temp_script << EOL
#!/bin/bash
#SBATCH --job-name=${experiment_name}
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=80GB
#SBATCH --partition=kempner
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

source "./slurm_scripts/common_functions.sh"
setup_environment

# Use the pre-assigned run number instead of generating a new one
RUN_NUMBER=${run_number}

# Create logs directory for this run and redirect outputs to separate log files
RUN_DIR="experiments/run_\${RUN_NUMBER}"
LOG_DIR="\${RUN_DIR}/logs"
mkdir -p "\${LOG_DIR}"

# Redirect stdout to process.log and stderr to error.log
exec > "\${LOG_DIR}/process.log" 2> "\${LOG_DIR}/error.log"

echo "======================================================="
echo "Starting experiment: $experiment_name"
echo "Model config: $layers layers, $heads heads, $embd_dim embedding dim"
echo "Training config: $epochs epochs, $train_steps train steps, batch size $batch_size"
echo "Context length: $context_length"
echo "Domain config: $domain_config"
echo "Run number: \${RUN_NUMBER}"
echo "======================================================="

# Determine which script to run
case "$EXPERIMENT_TYPE" in
    "basic")
        SCRIPT="./slurm_scripts/run_experiment.sh"
        ;;
    "agents_test")
        SCRIPT="./slurm_scripts/run_test_1b_agents.sh"
        ;;
    "environment_test") 
        SCRIPT="./slurm_scripts/run_test_1c_environments.sh"
        ;;
    "multi_domain")
        SCRIPT="./slurm_scripts/run_test_1a_multi_domain.sh"
        ;;
    *)
        echo "Unknown experiment type: $EXPERIMENT_TYPE"
        exit 1
        ;;
esac

# Update tracker file in the requested format BEFORE running the experiment
echo " " >> $TRACKER_FILE
echo "run\${RUN_NUMBER}: $EXPERIMENT_TYPE, $epochs epochs, $train_steps train steps, $layers layers, $heads heads, $context_length context length, $embd_dim embedding dimensions, $domain_config" >> $TRACKER_FILE

# Run the experiment
bash \$SCRIPT \${RUN_NUMBER} $layers $heads $epochs $train_steps $context_length $embd_dim $batch_size "$domain_config"

echo "Experiment completed: run\${RUN_NUMBER}"
EOL

    # Submit the job
    echo "Submitting experiment: $experiment_name with run number $run_number"
    sbatch $temp_script
    
    # Clean up the temp script
    rm $temp_script
    
    # Add a small delay to avoid overwhelming the scheduler
    sleep 1
}


# Submit jobs for each combination in the parameter sweep
for layers in "${LAYERS_ARRAY[@]}"; do
    for heads in "${HEADS_ARRAY[@]}"; do
        for epochs in "${EPOCHS_ARRAY[@]}"; do
            for train_steps in "${TRAIN_STEPS_ARRAY[@]}"; do
                for context_length in "${CONTEXT_LENGTH_ARRAY[@]}"; do
                    for embd_dim in "${EMBD_DIM_ARRAY[@]}"; do
                        for batch_size in "${BATCH_SIZE_ARRAY[@]}"; do
                            for domain_config in "${DOMAIN_CONFIG_ARRAY[@]}"; do
                                experiment_name="l${layers}_h${heads}_e${epochs}_c${context_length}_d${embd_dim}"
                                submit_experiment "$experiment_name" "$layers" "$heads" "$epochs" "$train_steps" "$context_length" "$embd_dim" "$batch_size" "$domain_config" "${NEXT_RUN_NUMBER}"
                                NEXT_RUN_NUMBER=$((NEXT_RUN_NUMBER + 1))
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "All experiment jobs submitted."