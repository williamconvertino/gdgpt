import setup_paths
from util import load_most_recent_checkpoint

import sys
import re
import importlib

def run_experiment(model, num_epochs_trained=0):
  pass

if __name__ == "__main__":
  # Extract model
  model_name = sys.argv[1]
  experiment_params = sys.argv[2:]
  
  model_module = importlib.import_module('src.models.' + model_name.lower())
  model_class = getattr(model_module, model_name)
  model_config = getattr(model_module, model_name + 'Config')
  
  config = model_config()
  
  # Extract model parameters
  head_regex = re.compile(r'(\d+)H')
  head_search = head_regex.search(experiment_params)
  if head_search:
    config.n_head = int(head_search.group(1))
  
  layer_regex = re.compile(r'(\d+)L')
  layer_search = layer_regex.search(experiment_params)
  if layer_search:
    config.n_layer = int(layer_search.group(1))
  
  ff_regex = re.compile(r'FF=(\w+)')
  ff_search = ff_regex.search(experiment_params)
  if ff_search:
    config.feed_forward = ff_search.group(1).lower() == 'true'
  
  model = model_class(config)
    
  # Resume training from most recent checkpoint unless otherwise specified
  resume_regex = re.compile(r'resume=(\w+)')
  resume_search = resume_regex.search(experiment_params)
  if not resume_search or resume_search.group(1).lower() == 'true':
    model, num_epochs_trained = load_most_recent_checkpoint(model)
    
  # Run experiment
  run_experiment(model, num_epochs_trained)