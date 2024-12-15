import importlib
import sys
import re
import os
import torch

from src.datasets import TinyStoriesDataset, ChildrenStoriesDataset
from src.tokenizers import TinyStoriesTokenizer, ChildrenStoriesTokenizer

TINYSTORIES_TOKENIZER_VOCAB_SIZE = 10000
CHILDREN_STORIES_TOKENIZER_VOCAB_SIZE = 15000

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
  
  if len(sys.argv) < 3 or sys.argv[2] == 'tiny':
    tokenizer = TinyStoriesTokenizer()
    train_dataset = TinyStoriesDataset(tokenizer, 'train', context_size=context_size)
    val_dataset = TinyStoriesDataset(tokenizer, 'val', context_size=context_size)
    test_dataset = TinyStoriesDataset(tokenizer, 'test', context_size=context_size)
  else:
    tokenizer = ChildrenStoriesTokenizer()
    train_dataset = ChildrenStoriesDataset(tokenizer, 'train', context_size=context_size)
    val_dataset = ChildrenStoriesDataset(tokenizer, 'val', context_size=context_size)
    test_dataset = ChildrenStoriesDataset(tokenizer, 'test', context_size=context_size)
  
  print(f"Loaded dataset {train_dataset.name} ({len(train_dataset)} samples) with tokenizer {tokenizer.name}")
  
  return tokenizer, train_dataset, val_dataset, test_dataset

def get_model_from_args():
  # Extract model
  model_name = sys.argv[1]
  
  if len(sys.argv) > 2:
    experiment_params = sys.argv[2]
  else:
    experiment_params = None
  
  model_class, model_config_class = get_model_class(model_name)
  
  if len(sys.argv) < 3 or sys.argv[2] == 'tiny':
    vocab_size = TINYSTORIES_TOKENIZER_VOCAB_SIZE
  else:
    vocab_size = CHILDREN_STORIES_TOKENIZER_VOCAB_SIZE
  
  config = model_config_class(vocab_size=vocab_size)
  
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
    
    d_embed_regex = re.compile(r'(\d+)D')
    d_embed_search = d_embed_regex.search(experiment_params)
    if d_embed_search:
      config.d_embed = int(d_embed_search.group(1))
    
    ff_regex = re.compile(r'ff=(\w+)')
    ff_search = ff_regex.search(experiment_params)
    if ff_search:
      config.use_ff = ff_search.group(1).lower() == 'true'
  
    ln_out_regex = re.compile(r'ln_out=(\w+)')
    ln_out_search = ln_out_regex.search(experiment_params)
    if ln_out_search:
      config.use_ln_out = ln_out_search.group(1).lower() == 'true'
  
    attn_fn_regex = re.compile(r'attn=(\w+)')
    attn_fn_search = attn_fn_regex.search(experiment_params)
    if attn_fn_search:
      config.attn_fn = attn_fn_search.group(1)
      
    wqk_regex = re.compile(r'wqk=(\w+)')
    wqk_search = wqk_regex.search(experiment_params)
    if wqk_search:
      config.wqk = wqk_search.group(1)
  
    wv_regex = re.compile(r'wv=(\w+)')
    wv_search = wv_regex.search(experiment_params)
    if wv_search:
      config.wv = wv_search.group(1)
  
    resume_regex = re.compile(r'resume=(\w+)')
    resume_search = resume_regex.search(experiment_params)
    if resume_search and resume_search.group(1).lower() == 'false':
      resume_from_checkpoint = False

  model = model_class(config)
  num_epochs_trained = 0
  
  if resume_from_checkpoint:
    model, num_epochs_trained = load_most_recent_checkpoint(model)
  
  return model, num_epochs_trained