import importlib
import sys
import re
import os
import torch
from dataclasses import fields

from src.datasets import TinyStoriesDataset, ChildrenStoriesDataset, CombinedDataset
from src.tokenizers import TinyStoriesTokenizer, ChildrenStoriesTokenizer, CombinedTokenizer

TINYSTORIES_TOKENIZER_VOCAB_SIZE = 10000
CHILDREN_STORIES_TOKENIZER_VOCAB_SIZE = 15000
COMBINED_TOKENIZER_VOCAB_SIZE = 15000

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../checkpoints')

def get_model_class(model_name):
  def _get_attr_case_insensitive(module, name):
    name = name.replace('_', '')
    for attr in dir(module):
      if attr.lower() == name.lower():
        return getattr(module, attr)
    return None
  model_module = importlib.import_module('src.models.' + model_name.lower())
  model_class = _get_attr_case_insensitive(model_module, model_name)
  model_config = _get_attr_case_insensitive(model_module, model_name + 'Config')
  return model_class, model_config

def load_most_recent_checkpoint(model):
  
  model_name = model.name
  
  if not os.path.exists(MODELS_DIR):
    return model, 0
  
  model_files = [f for f in os.listdir(MODELS_DIR) if f.startswith(f"{model_name}")]
  if not model_files:
    print(f"No existing checkpoint found for {model_name}")
    return model, 0
  
  model_files = sorted(model_files, key=lambda x: int(x.split('_')[-1][:-3]))
  latest_model_file = model_files[-1]
  latest_epoch = int(latest_model_file.split('_')[-1][:-3])
  
  model_path = os.path.join(MODELS_DIR, latest_model_file)
  
  model.load_state_dict(torch.load(model_path, weights_only=True))

  print(f"Loaded model with epoch={latest_epoch}")
  return model, latest_epoch

def get_tokenizer_and_dataset_from_args(context_size):
  
  if 'children_stories' in sys.argv[2:]:
    tokenizer = ChildrenStoriesTokenizer()
    train_dataset = ChildrenStoriesDataset(tokenizer, 'train', context_size=context_size)
    val_dataset = ChildrenStoriesDataset(tokenizer, 'val', context_size=context_size)
    test_dataset = ChildrenStoriesDataset(tokenizer, 'test', context_size=context_size)
  elif 'combined' in sys.argv[2:]:
    tokenizer = CombinedTokenizer()
    train_dataset = CombinedDataset(tokenizer, 'train', context_size=context_size)
    val_dataset = CombinedDataset(tokenizer, 'val', context_size=context_size)
    test_dataset = CombinedDataset(tokenizer, 'test', context_size=context_size)
  else:
    tokenizer = TinyStoriesTokenizer()
    train_dataset = TinyStoriesDataset(tokenizer, 'train', context_size=context_size)
    val_dataset = TinyStoriesDataset(tokenizer, 'val', context_size=context_size)
    test_dataset = TinyStoriesDataset(tokenizer, 'test', context_size=context_size)
  
  print(f"Loaded dataset {train_dataset.name} ({len(train_dataset)} batches, {len(train_dataset) * train_dataset.batch_size} samples)")
  
  return tokenizer, train_dataset, val_dataset, test_dataset

def get_model_from_args():
  # Extract model
  model_name = sys.argv[1]
  
  if 'children_stories' in sys.argv[2:]:
    vocab_size = CHILDREN_STORIES_TOKENIZER_VOCAB_SIZE
  elif 'combined' in sys.argv[2:]:
    vocab_size = COMBINED_TOKENIZER_VOCAB_SIZE
  else:
    vocab_size = TINYSTORIES_TOKENIZER_VOCAB_SIZE
  
  model_class, model_config_class = get_model_class(model_name)
  
  if len(sys.argv) < 3 or sys.argv[2] == 'tiny':
    vocab_size = TINYSTORIES_TOKENIZER_VOCAB_SIZE
  else:
    vocab_size = CHILDREN_STORIES_TOKENIZER_VOCAB_SIZE
  
  config = model_config_class(vocab_size=vocab_size)
  
  for parameter in sys.argv[2:]:
    s = parameter.split('=')
    if len(s) != 2:
      continue
    key, value = s
    if key in [f.name for f in fields(config)]:
      if type(getattr(config, key)) == int:
        value = int(value)
      elif type(getattr(config, key)) == bool:
        value = value.lower() == 'true'
      setattr(config, key, value)
      
  model = model_class(config)
  num_epochs_trained = 0
  
  model, num_epochs_trained = load_most_recent_checkpoint(model)
  
  return model, num_epochs_trained