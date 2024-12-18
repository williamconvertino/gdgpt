import time

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
