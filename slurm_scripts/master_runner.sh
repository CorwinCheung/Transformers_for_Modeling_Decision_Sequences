# Options are:
#   "basic": run_experiment.sh
#   "multi_domain": run_test_1a_multi_domain.sh
#   "agents_test": run_test_1b_agents.sh
#   "environment_test": run_test_1c_environments.sh
#   "long_term_test": run_long_term_dependency_test.sh

TRACKER_FILE="tracker.txt"

# Initialize starting run number - scan existing runs once at the beginning
initialize_run 8
NEXT_RUN_NUMBER=$RUN_NUMBER

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
    "long_term_test")
        SCRIPT="./slurm_scripts/run_long_term_dependency_test.sh"
        ;;
    *)
        echo "Unknown experiment type: $EXPERIMENT_TYPE"
        exit 1
        ;;
esac 