# Transformers_for_Modeling_Decision_Sequences
I will use language sequencing models to capture the decisions mice make in a dynamic, probabilistic task. This will involve modeling experimental data from real mouse decisions and synthetic data generated via descriptive models, which have been shown to capture general behavioral features in bandit tasks. 

With the experimental data, I will assess the predictive ability of a model to forecast a specific mouse’s behavior given its history. For synthetic data, I will evaluate the model’s ability to capture dependencies that reflect tendencies of synthetic choices. 
I will explore various transformer architectures and sizes for this task, testing the models on variations of bandit tasks (e.g. action-outcome probabilities changing independently). Additionally, I will examine the representations within the transformer’s input layer, attention blocks, and output layer and compare these to the neural data from the mice, including photometry data and the new high-dimensional Neuropixel data the Sabatini lab is collecting. I believe these representations can provide insights into the features that influence mouse behavior and serve as a basis for understanding neural activity.


Project Organization
----------
    ├── README.md                      <- Project overview and usage instructions.
    ├── requirements.txt               <- Environment setup and dependency management.

    ├── data
    │   ├── 2ABT_behavior_run_i.txt    <- Sequence output (10,000 Ls and Rs) for the ith generation.
    │   ├── 2ABT_high_port_run_i.txt   <- Simulation state (10,000 0s for left, 1s for right port) for the ith generation.
    │   └── metadata.txt               <- Stores num steps, environment, and agent settings for each run.

    ├── synthetic_data_generation
    │   ├── agent.py                   <- Estimates log odds and apply policy to make choice. Could be another agent
    │   ├── environment.py             <- Original_2ABT_Spouts class for simulation. Could be other env
    │   └── generate_data.py           <- Main function to generate sequential data.

    ├── evaluation
    │   ├── basic_evaluation.py        <- Basic metrics: reward, choice correctness(selects high port?), transitions.
    │   ├── graphs_on_trial_block_transitions.py <- Graphs prob switch, high port, and conditional probs(last 3 steps).
    │   └── graph_helper.py            <- Functions shared between graphs_on_trial... and graphs_transformer_vs_...

    ├── transformer
    │   ├── inference
    │   │   ├── evaluate_transformer_guess.py      <- Prints confusion matrices, switch rates.
    │   │   ├── guess_using_transformer.py         <- Uses transformer to predict next token on a certain test file.
    │   │   └── graphs_transformer_vs_ground_truth.py <- Compares transformer guesses with ground truth.
    │   ├── inspect_model.py           <- Outputs model parameters by layer.
    │   ├── transformer.py             <- Transformer (GPT-2 style) implementation.
    │   ├── train.py                   <- Trains the transformer and stores the path to it
    │   ├── predictions.txt            <- Model-generated prediction sequence.
    │   ├── model_seen92K.pth          <- Model trained on 92,000 tokens.
    │   ├── model_seen92M.pth          <- Model trained on 92 million tokens.
    │   └── model_metadata.txt         <- Metadata on each model: params, training data, tokens seen.

    ├── graphs                    <- stores graphs by the number of steps and the model vs rflr for 3 graphs
    │   └── model_1M_F_conditional_switching.png
    │   └── model_1M_G_selecting_high_reward_port.png
    │   └── model_1M_G_switch_probabilities.png
    |
    │   └── rflr_1M_F_conditional_switching.png
    │   └── rflr_1M_G_selecting_high_reward_port.png
    │   └── rflr_1M_G_switch_probabilities.png
