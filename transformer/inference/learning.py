from pathlib import Path

import pandas as pd

from utils.file_management import (get_experiment_file, get_latest_run,
                                   parse_model_info)


def load_predictions(run=None, model_name=None):
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

    """Load and process all prediction files for a run efficiently."""
    run = run or get_latest_run()
    # Load predictions    # Get model info from metadata
    model_info = parse_model_info(run, model_name=model_name)
    model_name = model_info['model_name']

    pred_file = get_experiment_file(f"learning_{model_name}_val_preds.txt", run)

    predictions = pd.read_csv(pred_file, sep='\t')
    # Sort by step and original index
    predictions = predictions.sort_values(['Step', 'Idx'])

    return predictions, model_info
