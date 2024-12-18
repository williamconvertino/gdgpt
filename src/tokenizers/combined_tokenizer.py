import os
from datasets import load_dataset, concatenate_datasets

from src.tokenizers import Tokenizer

TS_HUGGINGFACE_PATH = 'roneneldan/TinyStories'
CS_HUGGINGFACE_PATH = 'ajibawa-2023/Children-Stories-Collection'

TS_DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/datasets/TinyStories')
CS_DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/datasets/ChildrenStories')

TOKENIZER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/tokenizers/combined_tokenizer_15k')

class CombinedTokenizer(Tokenizer):
  
  def __init__(self):
    
    self.name = 'combined_tokenizer_15k'
    
    if not os.path.exists(TOKENIZER_DIR):
      print('Creating combined tokenizer files...')
      ts_dataset = load_dataset(TS_HUGGINGFACE_PATH, cache_dir=f'{TS_DATASET_DIR}/raw')
      cs_dataset = load_dataset(CS_HUGGINGFACE_PATH, cache_dir=f'{CS_DATASET_DIR}/raw')
      dataset = concatenate_datasets([ts_dataset['train'], cs_dataset['train']])
      Tokenizer.generate_tokenizer_files(dataset['text'], TOKENIZER_DIR, vocab_size=15000)
  
    super().__init__(TOKENIZER_DIR)