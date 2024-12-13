import importlib
import sys
import re
import os
import torch

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
  
  model_files = [f for f in os.listdir(MODELS_DIR) if f.startswith(f"{model_name}")]
  if not model_files:
    print(f"No existing checkpoint found for {model_name}")
    return model, 0
  
  model_files.sort(lambda x: int(x.split('_')[1]))
  latest_model_file = model_files[-1]
  latest_epoch = int(latest_model_file.split('_')[1])
  
  model_path = os.path.join(MODELS_DIR, latest_model_file)
  
  model.load_state_dict(torch.load(model_path, weights_only=True))

  print(f"Loaded model with epoch={latest_epoch}")
  return model, latest_epoch

def get_model_from_args():
  # Extract model
  model_name = sys.argv[1]
  
  if len(sys.argv) > 2:
    experiment_params = sys.argv[2]
  else:
    experiment_params = None
  
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
    
    d_embed_regex = re.compile(r'(\d+)D')
    d_embed_search = d_embed_regex.search(experiment_params)
    if d_embed_search:
      config.d_embed = int(d_embed_search.group(1))
    
    ff_regex = re.compile(r'ff=(\w+)')
    ff_search = ff_regex.search(experiment_params)
    if ff_search:
      config.use_ff = ff_search.group(1).lower() == 'true'
  
    attn_fn_regex = re.compile(r'attn=(\w+)')
    attn_fn_search = attn_fn_regex.search(experiment_params)
    if attn_fn_search:
      config.attn_fn = attn_fn_search.group(1)
      
    wqk_regex = re.compile(r'wqk=(\w+)')
    wqk_search = wqk_regex.search(experiment_params)
    if wqk_search:
      config.wqk = wqk_search.group(1)
  
    resume_regex = re.compile(r'resume=(\w+)')
    resume_search = resume_regex.search(experiment_params)
    if resume_search and resume_search.group(1).lower() == 'false':
      resume_from_checkpoint = False
      
    head_regex = re.compile(r'(\d+)He')
    head_search = head_regex.search(experiment_params)
    if head_search:
      config.n_head_e = int(head_search.group(1))
      
    head_regex = re.compile(r'(\d+)Hp')
    head_search = head_regex.search(experiment_params)
    if head_search:
      config.n_head_p = int(head_search.group(1))

  model = model_class(config)
  
  if resume_from_checkpoint:
    model, num_epochs_trained = load_most_recent_checkpoint(model)
  
  return model, num_epochs_trained