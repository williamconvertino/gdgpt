import importlib
import sys
import re

TINYSTORIES_TOKENIZER_VOCAB_SIZE = 10002

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
  return model, 0 # TODO: Implement this function

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