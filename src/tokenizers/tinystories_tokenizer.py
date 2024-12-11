import os
from transformers import GPT2TokenizerFast
from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset

HUGGINGFACE_PATH = 'roneneldan/TinyStories'
TOKENIZERS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/tokenizers/')
DATASETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/datasets/')

DATASET_NAME = 'tinystories'

class TinyStoriesTokenizer(GPT2TokenizerFast):
  
  def __init__(self, vocab_size=10000):
    
    self.name = f'{DATASET_NAME}_{vocab_size // 1000}k'
    self.tokenizer_path = f'{TOKENIZERS_PATH}/{self.name}'
    self.vocab_path = f'{self.tokenizer_path}/vocab.json'
    self.merges_path = f'{self.tokenizer_path}/merges.txt'
    self._vocab_size = vocab_size

    if not os.path.exists(self.vocab_path) or not os.path.exists(self.merges_path):
      print('Generating tokenizer from {DATASET_NAME} dataset...')
      self._generate_tokenizer_files()
    
    super().__init__(vocab_file=self.vocab_path, merges_file=self.merges_path)
    
    self.add_special_tokens({'eos_token': '[EOS]'})
  
  def vocab_size(self):
    return self._vocab_size
  
  def _generate_tokenizer_files(self, min_frequency=5):
    os.makedirs(self.tokenizer_path, exist_ok=True)
    tokenizer = ByteLevelBPETokenizer()
    dataset = load_dataset(HUGGINGFACE_PATH, cache_dir=f'{DATASETS_PATH}/raw')
    tokenizer.train_from_iterator(dataset['train']['text'], vocab_size=self._vocab_size, min_frequency=min_frequency)
    tokenizer.save_model(self.tokenizer_path)