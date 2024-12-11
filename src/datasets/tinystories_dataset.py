import os
import random
import math
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, IterableDataset

HUGGINGFACE_PATH = 'roneneldan/TinyStories'

DATASET_NAME = 'tinystories'
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/datasets/')

class TinyStoriesDataset(IterableDataset):
    
  def __init__(self, tokenizer, split, context_size, batch_size=64, shuffle_buffer_size=1024):
  
    if split == 'val':
      split = 'validation'
  
    self.tokenizer = tokenizer
    self.context_size = context_size
    self.stride = context_size // 2 # Overlap of 1/2 context size
    self.batch_size = batch_size
    self.shuffle_buffer_size = shuffle_buffer_size
    self.file_path = f'{DATASET_DIR}/{DATASET_NAME}/preprocessed/{split}.bin'
    
    self.uc_translation_table = str.maketrans('', '', '�â€œ™') # Unwanted characters
     
    if not os.path.exists(self.file_path):      
      dataset = load_dataset(HUGGINGFACE_PATH, split=f'{split}', cache_dir=f'{DATASET_DIR}/{DATASET_NAME}/raw')
      dataset = dataset.map(lambda x: self._preprocess(x), batched=True, remove_columns=['text'])
      self.data_size = sum([len(example) for example in dataset['input_ids']])
      self.data = self._generate_data(dataset)
    else:
      self.data_size = os.path.getsize(self.file_path) // np.dtype('int32').itemsize
      self.data = np.memmap(self.file_path, dtype='int32', mode='r')
  
  def _preprocess(self, examples):
    texts = [text.translate(self.uc_translation_table) for text in examples['text']] # Remove unwanted characters
    tokenized_texts = self.tokenizer(texts, return_attention_mask=False) # Tokenize the texts
    examples['input_ids'] = [example + [self.tokenizer.eos_token_id] for example in tokenized_texts['input_ids']] # Add EOS token to the end of each sequence
    return examples
      
  def _generate_data(self, dataset, buffer_size=1024):
    
    os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
    memmap_array = np.memmap(self.file_path, dtype='int32', mode='w+', shape=(self.data_size,))
    
    buffer = []
    write_pointer = 0
    
    for i, sequence in tqdm(enumerate(dataset['input_ids']), desc='Generating dataset files'):
      buffer.extend(sequence)
      if len(buffer) >= buffer_size:
        memmap_array[write_pointer: write_pointer + len(buffer)] = buffer
        write_pointer += len(buffer)
        buffer = []
    
    if len(buffer) > 0:
      memmap_array[write_pointer: write_pointer + len(buffer)] = buffer
      write_pointer += len(buffer)
      buffer = []
      
    memmap_array.flush()
    return memmap_array

  def __len__(self):
    num_windows = self.data_size / self.context_size
    num_windows = math.floor((num_windows * 2) - 1) # Account for sliding window overlap
    return math.ceil(num_windows / self.batch_size)
  
  # Implements a pseudo-shuffling mechanism via the buffer and creates output batches of size self.batch_size
  def __iter__(self):
    read_pointer = 0
    buffer = []
    batch = []
    
    while read_pointer + self.context_size < len(self.data):
      chunk = self.data[read_pointer: read_pointer + self.context_size]
      if len(chunk) < self.context_size:
        break
      buffer.append(chunk)
      read_pointer += self.stride
      if len(buffer) == self.shuffle_buffer_size:
        batch.append(torch.tensor(buffer.pop(random.randint(0, len(buffer) - 1))))
      if len(batch) == self.batch_size:
        batch_tensor = torch.stack(batch)
        batch = []
        yield batch_tensor.long()
    
    while len(buffer) > 0:
      batch.append(torch.tensor(buffer.pop(random.randint(0, len(buffer) - 1))))
      if len(batch) == self.batch_size:
        batch_tensor = torch.stack(batch)
        batch = []
        yield batch_tensor.long()
        
    if len(batch) > 0:
      yield torch.stack(batch).long()