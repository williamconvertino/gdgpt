import os
from datasets import load_dataset

from src.tokenizers import Tokenizer

HUGGINGFACE_PATH = 'roneneldan/TinyStories'

TOKENIZER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/tokenizers/ts_tokenizer_10k')
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/datasets/TinyStories')

class TinyStoriesTokenizer(Tokenizer):
  
  def __init__(self):
    
    self.name = 'ts_tokenizer_10k'
    
    if not os.path.exists(TOKENIZER_DIR):
      print('Creating TS tokenizer files...')
      dataset = load_dataset(HUGGINGFACE_PATH, cache_dir=f'{DATASET_DIR}/raw')
      Tokenizer.generate_tokenizer_files(dataset['train']['text'], TOKENIZER_DIR, vocab_size=10000)
  
    super().__init__(TOKENIZER_DIR)