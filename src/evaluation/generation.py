import torch

QUICK_EVAL_SEQUENCES = [  
  "There once was a fox who lived in the forest. He was very hungry and ",
  "There once was a princess who lived in a castle. She was very lonely and ",
]

def setup_device(model):
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
  return model, device

def quick_eval(model, tokenizer):
  
  model, device = setup_device(model)
  
  for sequence in QUICK_EVAL_SEQUENCES:
    
    model_input = tokenizer.encode(sequence)
    input_size = len(model_input)
    model_input = torch.tensor(model_input).unsqueeze(0)
    

    with torch.no_grad():
      model.eval()
      
      model_input = model_input.to(device)
      
      generated_sequence = model.generate(model_input)
      generated_text = tokenizer.decode(generated_sequence[0].tolist()[input_size:])
      
      beam_search_sequence = model.beam_search(model_input)
      beam_search_text = tokenizer.decode(beam_search_sequence[0].tolist()[input_size:])
    
    input_text = tokenizer.decode(model_input[0].tolist())
    sequence = sequence.replace('\n', '')
    generated_text = generated_text.replace('\n', '')
    beam_search_text = beam_search_text.replace('\n', '')

    print("=" * 100)
    print("<Prompt:> ")
    print(sequence)
    print("=" * 100)
    print("<Generated ending:> ")
    print(f'{input_text} [{generated_text}]')
    print("-" * 100)
    print("<Beam search ending:> ")
    print(f'{input_text} [{beam_search_text}]')

def evaluate_model_generation(model, tokenizer, test_dataset, num_generations=10):
  
  model, device = setup_device(model)
  
  for i, batch in enumerate(test_dataset):
    
    if i >= num_generations:
      break
    
    sequence = batch[0]

    input_size = model.config.context_size // 2

    model_input = sequence[:input_size]
    
    with torch.no_grad():
      model.eval()
      
      true_start = tokenizer.decode(model_input.tolist())
      true_end = tokenizer.decode(sequence[input_size:].tolist())
      
      model_input = model_input.unsqueeze(0).to(device)
      
      generated_sequence = model.generate(model_input)
      generated_end = tokenizer.decode(generated_sequence[0, input_size:].tolist())
      
      beam_search_sequence = model.beam_search(model_input)
      beam_search_end = tokenizer.decode(beam_search_sequence[0, input_size:].tolist())
      
    true_start = true_start.replace('\n', '')
    true_end = true_end.replace('\n', '')
    generated_end = generated_end.replace('\n', '')
    beam_search_end = beam_search_end.replace('\n', '')

    print("=" * 100)
    print("<Prompt:>")
    print(true_start)
    print("=" * 100)
    print("<True ending:>")
    print(f'{true_start} [{true_end}]')
    print("-" * 100)    
    print("<Generated ending:>")
    print(f'{true_start} [{generated_end}]')
    print("-" * 100)
    print("<Beam search ending:>")
    print(f'{true_start} [{beam_search_end}]')
    print("=" * 100)