import os
import time
import torch
import json
from torch.nn import functional as F

from src.visualization import visualize_loss

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2

CHECKPOINTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../checkpoints')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../results')

def get_time_remaining(start_time, step, num_steps):
  step = max(1, step)
  elapsed_time = time.time() - start_time
  steps_remaining = num_steps - step
  time_per_step = elapsed_time / step
  time_remaining = steps_remaining * time_per_step
  
  days = int(time_remaining // (24 * 3600))
  remaining_seconds = time_remaining % (24 * 3600)
  hours = int(remaining_seconds // 3600)
  remaining_seconds %= 3600
  minutes = int(remaining_seconds // 60)
  seconds = int(remaining_seconds % 60)
  
  formatted_time = f"{seconds}s"
  if minutes > 0:
    formatted_time = f"{minutes}m {formatted_time}"
  if hours > 0:
    formatted_time = f"{hours}h {formatted_time}"
  if days > 0:
    formatted_time = f"{days}d {formatted_time}"
  
  return formatted_time

def model_forward(model, batch, device):
  sequence = batch.to(device)
  input_ids = sequence[:, :-1]
  target_ids = sequence[:, 1:]
  _, loss = model(input_ids, target_ids)
  return loss

def train_model(model, train_dataset, val_dataset, max_epochs=None):
  
  # Setup
  # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if not torch.cuda.is_available():
    device = torch.device('cpu')
    print("CUDA not available, using CPU")
  else:
    print(f'Found {torch.cuda.device_count()} GPUs')
    for i in range(torch.cuda.device_count()):
      print(f'[GPU {i}] total memory used: {torch.cuda.memory_allocated(i) / (1024 ** 3)} GB')
      if torch.cuda.memory_allocated(i) / (1024 ** 3) < 1.0: # If GPU has less than 1GB of memory allocated, use it
        device = torch.device(f'cuda:{i}')
        print(f"Using GPU {i}")
        break
    device = torch.device('cpu')
    print("All GPUs are being used, using CPU")
  
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