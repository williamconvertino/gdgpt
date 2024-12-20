import setup_paths
import torch
from util import get_model_from_args, get_tokenizer_and_dataset_from_args, load_most_recent_checkpoint

from src.training import train_model
from src.datasets import TinyStoriesDataset
from src.tokenizers import TinyStoriesTokenizer

if __name__ == "__main__":
  
  torch.manual_seed(0)
  
  model = get_model_from_args()
  
  loaded_model, loaded_results = load_most_recent_checkpoint(model)
  
  if loaded_model is not None:
    if loaded_results is None:
      print(f"Checkpoint found for {model.name} but no results found, aborting training")
      exit()
    model = loaded_model
    
  # Load tokenizer and datasets
  tokenizer, train_dataset, val_dataset, _ = get_tokenizer_and_dataset_from_args(model.config.context_size)
  # Train the model
  train_model(model, train_dataset, val_dataset, loaded_results)