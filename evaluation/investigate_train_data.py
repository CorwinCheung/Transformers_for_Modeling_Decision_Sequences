import matplotlib.pyplot as plt

# Load files
with open("../data/2ABT_behavior_run_2.txt", "r") as f:
    train_data = f.read().strip()
    train_data = train_data.replace("\n","")
    
with open("../data/2ABT_behavior_run_3.txt", "r") as f:
    test_data = f.read().strip()
    test_data = test_data.replace("\n","")

# Split data into sequences of context length 12
context_length = 12
train_sequences = [train_data[i:i+context_length] for i in range(len(train_data) - context_length + 1)]
test_sequences = [test_data[i:i+context_length] for i in range(len(test_data) - context_length + 1)]

# Count overlaps
train_set = set(train_sequences)
test_set = set(test_sequences)
overlap = train_set.intersection(test_set)
overlap_percentage = len(overlap) / len(test_set) * 100

print(f"Total sequences in training data: {len(train_sequences)}")
print(f"Total sequences in testing data: {len(test_sequences)}")
print(f"Number of overlapping sequences: {len(overlap)}")
print(f"Overlap percentage: {overlap_percentage:.2f}%")

# Unique 12-mers in test but not in train
test_only_unique =  test_set - train_set
train_only_unique =  train_set - test_set
print("train unique: ", len(train_set))
print("test unique: ", len(test_set))
print("test only unique: ", len(test_only_unique))
print("test only unique: ", list(test_only_unique)[:5])


# Load predictions
with open("../transformer/inference/Preds_for_3_with_model_seen92M.txt", "r") as f:
    predictions = f.read().strip()
    predictions = predictions.replace("\n", "")

with open("../transformer/inference/Preds_for_2_with_model_seen92M.txt", "r") as f:
    predictions_train = f.read().strip()
    predictions_train = predictions_train.replace("\n", "")

# Ensure predictions align with the test data length
assert len(predictions) == len(test_data), "Predictions length does not match test data length."

# Generate actual tokens for each sequence
test_targets = [test_data[i + context_length - 1] for i in range(len(test_data) - context_length + 1)]
train_targets = [train_data[i + context_length - 1] for i in range(len(train_data) - context_length + 1)]

# Map predictions to test sequences
predicted_tokens = predictions[context_length - 1:]  # Offset by context length - 1
predictions_train_tokens = predictions_train[context_length - 1:]
assert len(predicted_tokens) == len(test_sequences), "Predictions length does not match test sequences length."

# Calculate accuracy for all test sequences
correct_predictions = [pred == actual for pred, actual in zip(predicted_tokens, test_targets)]
correct_predictions_train = [pred == actual for pred, actual in zip(predictions_train_tokens, train_targets)]
overall_accuracy = sum(correct_predictions) / len(correct_predictions)
overall_accuracy_train = sum(correct_predictions_train) / len(correct_predictions_train)

print(len(test_sequences))
print(len(test_only_unique))
print(len(train_only_unique))
test_only_indices = [i for i, seq in enumerate(test_sequences) if seq in test_only_unique]
print("the number of sequences in test that are unique to test is ", len(test_only_indices))
test_only_correct = [correct_predictions[i] for i in test_only_indices]
test_only_accuracy = sum(test_only_correct) / len(test_only_correct)
train_only_indices = [i for i, seq in enumerate(train_sequences) if seq in train_only_unique]
print("the number of sequences in train that are unique to train is ", len(train_only_indices))
train_only_correct = [correct_predictions_train[i] for i in train_only_indices]
train_only_accuracy = sum(train_only_correct) / len(train_only_correct)

# Display results
print(f"Overall accuracy on test data: {overall_accuracy * 100:.2f}%")
print(f"Accuracy on test-only unique 12-mers: {test_only_accuracy * 100:.2f}%")
print(f"Overall accuracy on train data: {overall_accuracy_train * 100:.2f}%")
print(f"Accuracy on train-only unique 12-mers: {train_only_accuracy * 100:.2f}%")

# Visualization of accuracies
categories = ['Overall Test Accuracy', 'Test-Only Unique Accuracy', 'Train-Only Unique Accuracy', 'Overall Train Accuracy']
values = [overall_accuracy * 100, test_only_accuracy * 100, train_only_accuracy * 100, overall_accuracy_train * 100]

plt.figure(figsize=(8, 6))
plt.bar(categories, values)
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy on Test Data")
plt.show()

print("Sample test_sequences:")
print(test_sequences[:5])

print("\nSample test_only_unique:")
print(list(test_only_unique)[:5])