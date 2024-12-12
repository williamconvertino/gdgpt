import torch
    
def evaluate_model_generation(model, tokenizer, test_dataset, num_generations=5):
  for i, batch in enumerate(test_dataset):
    if i >= num_generations:
      break
    sequence = batch[0]
    decoded_sequence = tokenizer.decode(sequence.tolist())
    test_sequence = sequence[:model.config.context_size // 2]