import setup_paths
import torch
from util import get_model_from_args, get_tokenizer_and_dataset_from_args

from src.evaluation import evaluate_model_generation
from src.datasets import TinyStoriesDataset
from src.tokenizers import TinyStoriesTokenizer

if __name__ == "__main__":
  
  torch.manual_seed(0)
  
  model, _ = get_model_from_args()
  
  # Load tokenizer and datasets
  tokenizer, _, _, test_dataset = get_tokenizer_and_dataset_from_args(model.config.context_size)
  
  # Train the model
  evaluate_model_generation(model, tokenizer, test_dataset)