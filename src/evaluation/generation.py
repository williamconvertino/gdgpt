import torch

QUICK_EVAL_SEQUENCES = [  
  "There once was a fox who lived in the forest. He was very hungry and ",
  "There once was a princess who lived in a castle. She was very lonely and ",
]

def quick_eval(model, tokenizer):
  for sequence in QUICK_EVAL_SEQUENCES:
    model_input = tokenizer.encode(sequence)
    model_input = torch.tensor(model_input).unsqueeze(0)
    
    with torch.no_grad():
      model.eval()
      
      generated_sequence = model.generate(model_input)
      generated_text = tokenizer.decode(generated_sequence[0].tolist())
      
      beam_search_sequence = model.beam_search(model_input)
      beam_search_text = tokenizer.decode(beam_search_sequence[0].tolist())
      
    print("=" * 100)
    print("<Prompt:> ")
    print(sequence)
    print("=" * 100)
    print("<Generated ending:> ")
    print(generated_text)
    print("-" * 100)
    print("<Beam search ending:> ")
    print(beam_search_text)

def evaluate_model_generation(model, tokenizer, test_dataset, num_generations=10):
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
      
      generated_sequence = model.generate(model_input.unsqueeze(0))
      generated_end = tokenizer.decode(generated_sequence[0, input_size:].tolist())
      
      beam_search_sequence = model.beam_search(model_input.unsqueeze(0))
      beam_search_end = tokenizer.decode(beam_search_sequence[0, input_size:].tolist())
      
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