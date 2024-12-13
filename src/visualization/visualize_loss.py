import os
import json
from matplotlib import pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../results')
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../figures')

def visualize_loss(*args, num_epochs=None, num_epoch_steps=None, title="Losses", xlabel="Step", ylabel="Loss"):
  
  colors = ['c', 'r', 'g', 'm', 'y', 'k']
  
  plt.figure(figsize=(10, 6))
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  
  plt.ylim(0, 9.5)
  
  for i, item in enumerate(args):
    if len(item) == 3:
      losses, label, color = item
    else:
      losses, label = item
      color = colors[i]
    
    if num_epochs is not None:
      losses = [loss for loss in losses if loss[0] <= num_epochs * num_epoch_steps]
      
    x, y = zip(*losses)
    plt.plot(x, y, label=label, color=color)
  
  if num_epoch_steps is not None:
    if num_epochs is None:
      num_epochs = max([losses[-1][0] for losses, _, _ in args])
    
    for i in range(0, num_epochs * num_epoch_steps, num_epoch_steps):
      x = losses[i][0]
      plt.axvline(x=x, color='gray', linestyle='--', linewidth=0.5)
    
  plt.legend()
  save_title = title.lower().replace(" ", "_")
  os.makedirs(FIGURES_DIR, exist_ok=True)
  plt.savefig(f'{FIGURES_DIR}/{save_title}.png')
  plt.show()

def visualize_loss_from_files(*args, num_epochs=None, title="Losses", xlabel="Step", ylabel="Loss"):
  
  new_args = []
  
  for i, item in enumerate(args):
    if len(item) == 3:
      file_name, label, color = item
    else:
      file_name, label = item
      
    with open(f'{RESULTS_DIR}/{file_name}.json', 'r') as f:
      data = json.load(f)
      data = data['val_losses']
    
    if len(item) == 3:
      new_args.append((data, label, color))
    else:
      new_args.append((data, label))
    
  visualize_loss(*new_args, num_epochs=num_epochs, title=title, xlabel=xlabel, ylabel=ylabel)