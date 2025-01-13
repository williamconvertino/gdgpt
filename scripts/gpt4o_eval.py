import setup_paths
import torch
from util import get_model_from_args, get_tokenizer_and_dataset_from_args, load_most_recent_checkpoint

from src.evaluation import generate_gpt4o_inputs
from src.datasets import TinyStoriesDataset
from src.tokenizers import TinyStoriesTokenizer

if __name__ == "__main__":
  
  torch.manual_seed(0)
  
  model = get_model_from_args()
  model, _ = load_most_recent_checkpoint(model)
  
  # Load tokenizer and datasets
  tokenizer, _, _, test_dataset = get_tokenizer_and_dataset_from_args(model.config.context_size)
  
  # Train the model
  generate_gpt4o_inputs(model, tokenizer, test_dataset, num_generations=10)