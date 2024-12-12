import setup_paths
import torch
from util import get_model_from_args

from src.training import train_model
from src.datasets import TinyStoriesDataset
from src.tokenizers import TinyStoriesTokenizer

if __name__ == "__main__":
  
  torch.manual_seed(0)
  
  model, _ = get_model_from_args()
  
  # Load tokenizer and datasets
  tokenizer = TinyStoriesTokenizer()
  train_dataset = TinyStoriesDataset(tokenizer, 'train', context_size=model.config.context_size)
  val_dataset = TinyStoriesDataset(tokenizer, 'val', context_size=model.config.context_size)
  
  # Train the model
  train_model(model, train_dataset, val_dataset)