import os
from transformers import GPT2TokenizerFast
from tokenizers import ByteLevelBPETokenizer

class Tokenizer(GPT2TokenizerFast):
  
  def __init__(self, tokenizer_path):
    
    vocab_path = f'{tokenizer_path}/vocab.json'
    merges_path = f'{tokenizer_path}/merges.txt'
    
    super().__init__(vocab_file=vocab_path, merges_file=merges_path)
  
  def generate_tokenizer_files(dataset, tokenizer_path, vocab_size, min_frequency=5):
    os.makedirs(tokenizer_path, exist_ok=True)
    bpe = ByteLevelBPETokenizer()
    print('Training tokenizer...')
    bpe.train_from_iterator(dataset, vocab_size=vocab_size - 1, min_frequency=min_frequency, special_tokens=['<eos>']) # -1 to account for the <eos> token
    print('Saving tokenizer...')
    bpe.save_model(tokenizer_path)