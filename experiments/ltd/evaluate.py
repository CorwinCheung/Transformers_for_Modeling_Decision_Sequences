import os
import sys
import torch
from pattern_utils import (
    create_patterns,
    load_model_directly,
    generate_predictions,
    evaluate_pattern_accuracy
)

def run_evaluation(model_name, pattern_length, context_length=12):
    """Run evaluation for a specific model and pattern length"""
    print(f"\n===== Evaluating {model_name} on pattern length {pattern_length} =====")
    
    test_file = f"test_pattern_{pattern_length}.txt"
    if not os.path.exists(test_file):
        print(f"Error: Test file {test_file} not found.")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join("models", f"{model_name}.pth")
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return
    
    model = load_model_directly(model_path, device)
    print(f"Model loaded from {model_path}")
    
    os.makedirs("results", exist_ok=True)
    pred_file = os.path.join("results", f"pred_{model_name}_on_pattern{pattern_length}.txt")
    
    generate_predictions(
        model=model,
        input_file=test_file,
        output_file=pred_file,
        context_length=context_length,
        device=device
    )
    
    result = evaluate_pattern_accuracy(
        predictions_file=pred_file,
        ground_truth_file=test_file,
        pattern_length=pattern_length
    )
    
    if result:
        print("\n===== Results =====")
        print(f"Overall Accuracy: {result['overall_accuracy']:.4f}")
        print(f"Pattern Position Accuracy: {result['pattern_accuracy']:.4f}")
        print(f"Non-Pattern Position Accuracy: {result['non_pattern_accuracy']:.4f}")
        print(f"Pattern Position 'R' Prediction Rate: {result['pattern_r_prediction_rate']:.4f}")
        print(f"Pattern Positions containing 'R' in ground truth: {result['pattern_r_ground_truth_rate']:.4f}")
    
    return result

def main():
    # First, create pattern files if needed
    create_patterns()
    
    # Define evaluations to run
    evaluations = [
        {"model": "model_pattern11", "pattern": 11},
        {"model": "model_pattern11", "pattern": 15},
        {"model": "model_pattern15", "pattern": 11},
        {"model": "model_pattern15", "pattern": 15},
    ]
    
    # Run all evaluations
    results = {}
    for eval_config in evaluations:
        model_name = eval_config["model"]
        pattern_length = eval_config["pattern"]
        key = f"{model_name}_on_pattern{pattern_length}"
        results[key] = run_evaluation(model_name, pattern_length, context_length=12)
    
    # Print summary
    print("\n===== SUMMARY =====")
    for key, result in results.items():
        if result:
            print(f"{key}: Pattern Accuracy = {result['pattern_accuracy']:.4f}")

if __name__ == "__main__":
    # Add project root to Python path
    sys.path.append(os.path.abspath(os.path.join(__file__, '../../')))
    main() 