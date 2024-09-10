# Transformers_for_Modeling_Decision_Sequences
I will use language sequencing models to capture the decisions mice make in a dynamic, probabilistic task. This will involve modeling experimental data from real mouse decisions and synthetic data generated via descriptive models, which have been shown to capture general behavioral features in bandit tasks. 

With the experimental data, I will assess the predictive ability of a model to forecast a specific mouse’s behavior given its history. For synthetic data, I will evaluate the model’s ability to capture dependencies that reflect tendencies of synthetic choices. 
I will explore various transformer architectures and sizes for this task, testing the models on variations of bandit tasks (e.g. action-outcome probabilities changing independently). Additionally, I will examine the representations within the transformer’s input layer, attention blocks, and output layer and compare these to the neural data from the mice, including photometry data and the new high-dimensional Neuropixel data the Sabatini lab is collecting. I believe these representations can provide insights into the features that influence mouse behavior and serve as a basis for understanding neural activity.


