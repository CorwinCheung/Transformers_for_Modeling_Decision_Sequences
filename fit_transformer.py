import torch
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader
import transformer.transformer as transformer  # Importing from transformer.py
import os

def tokenize(data_string, tokenizer, max_length=512):
    """
    Tokenize the input string of mouse behavior sequences.
    
    Args:
    - data_string (str): A string containing sequences of tokens (e.g., 'LrRLlR').
    - tokenizer: The GPT-2 tokenizer.
    - max_length (int): Maximum sequence length for tokenization.
    
    Returns:
    - input_ids: The tokenized input IDs.
    """
    lines = data_string.splitlines()
    tokenized = tokenizer(lines, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    return tokenized.input_ids

def train_model(data_string, model_name='gpt2', output_dir='./output', epochs=3, batch_size=8, max_length=512):
    """
    Fine-tune the GPT-2 model on mouse behavior token sequences.
    
    Args:
    - data_string (str): A string containing sequences of mouse behavior tokens.
    - model_name (str): The name of the model to use (default is 'gpt2').
    - output_dir (str): Directory where the trained model and outputs will be saved.
    - epochs (int): Number of training epochs.
    - batch_size (int): Batch size for training.
    - max_length (int): Maximum length of token sequences.
    """

    model, tokenizer = transformer.get_model_and_tokenizer(model_name)
    input_ids = tokenize(data_string, tokenizer, max_length=max_length)
    dataset = torch.utils.data.TensorDataset(input_ids)
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=500,
        save_total_limit=2,
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        report_to="none"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def main():
    data_file_path = './data/2ABT_logistic.txt'

    if os.path.exists(data_file_path):
        with open(data_file_path, 'r') as file:
            data_string = file.read()
        train_model(data_string, model_name='gpt2', output_dir='./output', epochs=5, batch_size=4)
    else:
        print(f"Error: File {data_file_path} not found.")

main()