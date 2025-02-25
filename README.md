# Transformers_for_Modeling_Decision_Sequences
I will use language sequencing models to capture the decisions mice make in a dynamic, probabilistic task. This will involve modeling experimental data from real mouse decisions and synthetic data generated via descriptive models, which have been shown to capture general behavioral features in bandit tasks. 

With the experimental data, I will assess the predictive ability of a model to forecast a specific mouse’s behavior given its history. For synthetic data, I will evaluate the model’s ability to capture dependencies that reflect tendencies of synthetic choices. 
I will explore various transformer architectures and sizes for this task, testing the models on variations of bandit tasks (e.g. action-outcome probabilities changing independently). Additionally, I will examine the representations within the transformer’s input layer, attention blocks, and output layer and compare these to the neural data from the mice, including photometry data and the new high-dimensional Neuropixel data the Sabatini lab is collecting. I believe these representations can provide insights into the features that influence mouse behavior and serve as a basis for understanding neural activity.

## Research Questions
1. **Performance in Decision Sequences**
   - How well can transformers adapt to different groups of environments/contexts?
   - What are the long-term dependencies in decision-making?

2. **Interpretability of Transformers**
   - How is information represented in attention weight patterns?
   - Can we understand MLPs as state transition functions?
   - What is the stability/variance behavior?
   - Long Term Goal: use transformer attention patterns and representations to generate testable predictions about neural mechanisms, compare with neural recording data (photometry and Neuropixel) and understand the features driving behavioral flexibility

3. **Synthetic Data Pipeline**
   - Different encoding schemes
   - Validity/fidelity checks


Project Organization
----------
    ├── README.md                      <- Project overview and motivation
    ├── requirements.txt               <- Packages and dependencies
    ├── environment.yml                <- Environment setup
    │
    ├── evaluation/                    <- Evaluation and analysis code
    │   ├── basic_evaluation.py        <- Core evaluation metrics
    │   ├── graphs_on_trial_block_transitions.py <- Trial analysis visualizations  
    │   └── graph_helper.py            <- Shared graphing utilities
    │
    ├── synthetic_data_generation/     <- Code for generating synthetic data
    │   ├── agent.py                   <- Agent implementations for decision making
    │   ├── environment.py             <- Environment implementations
    │   └── generate_data.py           <- Data generation pipeline
    │
    │
    ├── transformer/                   <- Transformer model implementation
    │   ├── inference/                 <- Model inference code
    │   │   ├── evaluate_transformer_guess.py    <- Evaluation metrics
    │   │   ├── guess_using_transformer.py       <- Model prediction pipeline
    │   │   └── graphs_transformer_vs_ground_truth.py <- Results visualization
    │   │
    │   ├── inspect_model.py           <- Model inspection utilities
    │   ├── transformer.py             <- Core transformer implementation
    │   ├── train.py                   <- Training pipeline
    │   ├── predictions.txt            <- Model predictions output
    │   ├── models/                    <- Saved model checkpoints
    │   │   ├── model_seen92K.pth      <- 92K token checkpoint
    │   │   └── model_seen92M.pth      <- 92M token checkpoint
    │   └── model_metadata.txt         <- Model training metadata
    │
    ├── utils/                         <- Utility functions and helpers
    │   ├── parse_data.py              <- Data parsing utilities
    │   ├── file_management.py         <- File I/O operations
    │   └── visualization.py           <- Plotting helpers
    │
    ├── slurm_scripts/                 <- HPC job submission scripts
    │   ├── train.sh                   <- Training job script
    │   ├── evaluate.sh                <- Evaluation job script
    │   └── slurm_output/              <- Job output logs
    │
    ├── experiments/                   <- Experiment configurations and results
    │   ├── configs/                   <- Experiment parameter configs
    │   └── results/                   <- Experiment output data
    │
    └── graphs/                        <- Generated visualization outputs
        ├── model_*/                   <- Model-specific visualizations
        │   ├── conditional_switching.png
        │   ├── selecting_high_reward_port.png
        │   └── switch_probabilities.png
        │
        └── rflr_*/                    <- Baseline model visualizations
            ├── conditional_switching.png
            ├── selecting_high_reward_port.png
            └── switch_probabilities.png