import setup_paths
import torch
from util import get_model_from_args, get_tokenizer_and_dataset_from_args, load_most_recent_checkpoint, get_flags_from_args

from src.evaluation import generate_gpt4o_inputs, create_batch
from src.datasets import TinyStoriesDataset
from src.tokenizers import TinyStoriesTokenizer

if __name__ == "__main__":
  
  torch.manual_seed(0)
  
  flags = get_flags_from_args()
  
  if 'input' in flags:
    print("Creating inputs")
    model = get_model_from_args()
    model, _ = load_most_recent_checkpoint(model)
    
    tokenizer, _, _, test_dataset = get_tokenizer_and_dataset_from_args(model.config.context_size)
    
    generate_gpt4o_inputs(model, tokenizer, test_dataset, num_generations=10)
    
  elif 'batch' in flags:
    print("Creating batch")
    create_batch()
  elif 'check' in flags:
    pass
  elif 'parse' in flags:
    pass
  else:
    raise ValueError("No valid flags detected. Please use --input or --batch")