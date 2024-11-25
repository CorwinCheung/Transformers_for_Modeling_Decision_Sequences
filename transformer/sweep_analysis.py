import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("sweep.csv")

# Expand the summary and config columns from JSON strings to dictionaries
df['summary'] = df['summary'].apply(eval)
df['config'] = df['config'].apply(eval)

# Create separate DataFrames for summary and config data
summary_df = pd.json_normalize(df['summary'])
config_df = pd.json_normalize(df['config'])

# Concatenate to combine all details for easier analysis
full_df = pd.concat([summary_df, config_df], axis=1)

# Generate summary statistics for selected metrics
metrics = ['loss', 'val_loss', '_runtime']
summary_stats = full_df[metrics].describe()

# Print summary statistics
print("Summary Statistics for Metrics:")
print(summary_stats)

# Analyzing hyperparameters
hyperparams = ['T', 'n_embd', 'n_head', 'n_layer', 'max_steps']
hyperparam_analysis = config_df[hyperparams].apply(pd.value_counts)

# Print analysis of hyperparameters
print("\nHyperparameter Distribution:")
print(hyperparam_analysis)

epochs = 1
steps = 16
heads = 2
layers = 2
filtered_df = full_df[(full_df['max_steps'] == steps) & 
                      (full_df['n_head'] == heads) & 
                      (full_df['n_layer'] == layers)]

# Group by `T` and `N_embd` to analyze val_loss
grouped = filtered_df.groupby(['T', 'n_embd'])['val_loss'].mean().reset_index()

# Pivot table to facilitate plotting
pivot = grouped.pivot(index='T', columns='n_embd', values='val_loss')

# Plot the validation loss vs. T for different values of N_embd
plt.figure(figsize=(10, 6))
for n_embd in pivot.columns:
    plt.plot(pivot.index, pivot[n_embd], label=f"N_embd={n_embd}")

plt.title(f"Validation Loss for fixed epochs={epochs}, heads={heads}, layers={layers}")
plt.xlabel("T (Sequence Length)")
plt.ylabel("Validation Loss")
plt.legend(title="Embedding Size (N_embd)")
plt.grid(True)
plt.savefig(f"H2L2 Val Loss evolution epochs={epochs}.png")