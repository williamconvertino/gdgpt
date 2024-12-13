import torch

def evaluate_model_generation(model, tokenizer, test_dataset, num_generations=5):
  for i, batch in enumerate(test_dataset):
    
    if i >= num_generations:
      break
    
    sequence = batch[0]

    input_size = model.config.context_size // 2

    model_input = sequence[:input_size]
    
    true_start = tokenizer.decode(model_input.tolist())
    true_end = tokenizer.decode(sequence[input_size:].tolist())
    
    generated_sequence = model.generate(model_input.unsqueeze(0))
    generated_end = tokenizer.decode(generated_sequence[0].tolist())
    
    beam_search_sequence = model.beam_search(model_input.unsqueeze(0))
    beam_search_end = tokenizer.decode(beam_search_sequence[0].tolist())
    
    print("=" * 100)
    print("<Prompt:>")
    print(true_start)
    print("=" * 100)
    print("<True ending:>")
    print(true_end)
    print("-" * 100)    
    print("<Generated ending:>")
    print(generated_end)
    print("-" * 100)
    print("<Beam search ending:>")
    print(beam_search_end)