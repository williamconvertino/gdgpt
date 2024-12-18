import os
import time
import torch
import json
from torch.nn import functional as F

from src.visualization import visualize_loss
from src.util import get_time_remaining

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2

CHECKPOINTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../checkpoints')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../results')

def model_forward(model, batch, device):
  sequence = batch.to(device)
  input_ids = sequence[:, :-1]
  target_ids = sequence[:, 1:]
  _, loss = model(input_ids, target_ids)
  return loss

def train_model(model, train_dataset, val_dataset, max_epochs=None):
  
  # Setup
  # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  device = None
  if torch.cuda.is_available():
    print(f'Found {torch.cuda.device_count()} GPUs')
    for i in range(torch.cuda.device_count()):
      
      gpu = torch.device(f'cuda:{i}')
      free_memory, total_memory = torch.cuda.mem_get_info(gpu)
      total_memory = int(total_memory / 1024**3)
      free_memory = int(free_memory / 1024**3)
      
      percent_used = (total_memory - free_memory) / total_memory
      
      print(f'[GPU {i}] Total memory: {total_memory}GB, Free memory: {free_memory}GB')
      
      if percent_used < 0.1:
        device = torch.device(f'cuda:{i}')
        print(f"Using GPU {i}")
        break
    if device is None:
      print("All GPUs are being used. Using CPU instead.")
      device = torch.device('cpu')
  else:
    print("No GPUs found. Using CPU.")
    device = torch.device('cpu')
  
  model.to(device)
  
  optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

  results = {
    'num_epoch_steps': len(train_dataset),
    'num_epochs': 0,
    'train_losses': [],
    'val_losses': []
  }

  record_steps = len(train_dataset) // 100 # Only validates/saves model losses a limited number of times (for performance/memory reasons)
  
  epoch = 0
  
  # Training Loop
  print(f"Training {model.name} [Device: {device}]")
  
  while True:
    
    if max_epochs is not None and epoch >= max_epochs:
      break
    
    train_loss = 0.0
    val_loss = 0.0
    
    start_time = time.time()
    
    for step, batch in enumerate(train_dataset):
      
      model.train()
      
      optimizer.zero_grad()
      
      train_loss = model_forward(model, batch, device)
      train_loss.backward()
      optimizer.step()
      
      train_loss = train_loss.item()
      
      if step % record_steps == 0 or step == len(train_dataset) - 1:
        total_val_loss = 0.0
        model.eval()
        with torch.no_grad():
          for val_batch in val_dataset:
            batch_val_loss = model_forward(model, val_batch, device)
            batch_val_loss = batch_val_loss.item()
            total_val_loss += batch_val_loss
        val_loss = total_val_loss / len(val_dataset)
        
        # Write both train and val losses at the same step
        total_step = epoch * len(train_dataset) + step
        
        results['val_losses'].append((total_step, val_loss))
        results['train_losses'].append((total_step, train_loss))

        if step == 0:
          start_time = time.time() # Reset start time to avoid time remaining being skewed by initial validation time
      
      if step <= 1000 or step % 100 == 0 or step == len(train_dataset) - 1:
        time_remaining = get_time_remaining(start_time, step, len(train_dataset))
        print(f"\r\tEpoch {epoch} | Step {step}/{len(train_dataset)} | Train Loss: {train_loss:.4f} | Most Recent Val Loss: {val_loss:.4f} | Time Remaining: {time_remaining}", end='')
    
    print(f"\nEpoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    epoch += 1
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    
    torch.save(model.state_dict(), f'{CHECKPOINTS_DIR}/{model.name}_epoch_{epoch}.pt')
    
    with open(f'{RESULTS_DIR}/{model.name}.json', 'w') as f:
      json.dump(results, f)
      
    visualize_loss((results['train_losses'], "Train"), (results['val_losses'], "Test"), title=f"{model.name} Training Losses (Epoch {epoch})")