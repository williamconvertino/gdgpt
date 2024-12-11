import setup_paths
import sys
import re
import torch
from util import load_most_recent_checkpoint, get_model_class
from src.training import train_model
from src.datasets import TinyStoriesDataset
from src.tokenizers import TinyStoriesTokenizer

TINYSTORIES_TOKENIZER_VOCAB_SIZE = 10002

def run_experiment(model, num_epochs_trained=0, seed=0):
  
  # Set seed
  torch.manual_seed(seed)
  
  # Load tokenizer and datasets
  tokenizer = TinyStoriesTokenizer()
  train_dataset = TinyStoriesDataset(tokenizer, 'train', context_size=config.context_size)
  val_dataset = TinyStoriesDataset(tokenizer, 'val', context_size=config.context_size)
  
  # Train the model
  train_model(model, train_dataset, val_dataset)

if __name__ == "__main__":
  
  # Extract model
  model_name = sys.argv[1]
  experiment_params = sys.argv[2]
  
  model_class, model_config_class = get_model_class(model_name)
  
  config = model_config_class(vocab_size=TINYSTORIES_TOKENIZER_VOCAB_SIZE)
  
  resume_from_checkpoint = True
  
  # Extract model and experiment parameters
  if experiment_params:
    head_regex = re.compile(r'(\d+)H')
    head_search = head_regex.search(experiment_params)
    if head_search:
      config.n_head = int(head_search.group(1))
    
    layer_regex = re.compile(r'(\d+)L')
    layer_search = layer_regex.search(experiment_params)
    if layer_search:
      config.n_layer = int(layer_search.group(1))
    
    ff_regex = re.compile(r'FF=(\w+)|ff=(\w+)')
    ff_search = ff_regex.search(experiment_params)
    if ff_search:
      config.use_ff = ff_search.group(1).lower() == 'true'
  
    resume_regex = re.compile(r'resume=(\w+)')
    resume_search = resume_regex.search(experiment_params)
    if resume_search and resume_search.group(1).lower() == 'false':
      resume_from_checkpoint = False

  model = model_class(config)
  
  if resume_from_checkpoint:
    model, num_epochs_trained = load_most_recent_checkpoint(model)
    
  # Run experiment
  run_experiment(model, num_epochs_trained)