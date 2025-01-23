import pandas as pd
from utils.file_management import get_experiment_file
import glob
import os

def analyze_predictions(run=None):
    # # Load predictions
    # pred_file = get_experiment_file("validation_predictions.txt", run)
    # df = pd.read_csv(pred_file, sep='\t')
    
    # # Calculate accuracy
    # accuracy = (df['True'] == df['Predicted']).mean()
    
    # # Analyze predictions by context
    # context_accuracy = df.groupby('Context').agg({
    #     'True': 'first',
    #     'Predicted': 'first'
    # }).apply(lambda x: x['True'] == x['Predicted']).mean()
    
    # return {
    #     'overall_accuracy': accuracy,
    #     'context_accuracy': context_accuracy,
    #     'predictions_df': df
    # }

    run_dir = get_run_dir(run)
    
    # Find all prediction files using glob
    pred_files = glob.glob(os.path.join(run_dir, "learning", "val_preds_*.txt"))
    

    # Load all prediction files for detailed analysis
    all_predictions = []
    for pred_file in pred_files:
        pred_path = get_experiment_file(pred_file, run)
        preds = pd.read_csv(pred_path, sep='\t')
        all_predictions.append(preds)

    # Analyze how predictions for specific contexts change over training
    context_evolution = pd.concat(all_predictions)
    context_accuracy = context_evolution.groupby(['Step', 'Context']).agg({
        'True': 'first',
        'Predicted': lambda x: (x == context_evolution['True']).mean()
    }).reset_index()
    
    return {
        'context_evolution': context_evolution,
        'context_accuracy': context_accuracy
    }