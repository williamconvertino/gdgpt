import os
import random
import math
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, IterableDataset

class Dataset(IterableDataset):
    
  def __init__(self, file_path, context_size, batch_size=64, shuffle=False, shuffle_buffer_size=1024):
    
    self.file_path = file_path
    self.context_size = context_size
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.shuffle_buffer_size = shuffle_buffer_size
      
    self.data = np.memmap(file_path, dtype='int32', mode='r')
    self.file_size = os.path.getsize(self.file_path) // np.dtype('int32').itemsize

  def __len__(self):
    num_windows = self.file_size / self.context_size
    num_windows = math.floor((num_windows * 2) - 1) # Account for sliding window overlap
    return math.ceil(num_windows / self.batch_size)
  
  def __iter__(self):
    
    read_pointer = 0
    buffer = []
    batch = []
  
    # Pseudo-shuffling mechanism that chooses random elements from the buffer 
    def pop_buffer():
      pop_index = random.randint(0, len(buffer) - 1) if self.shuffle else 0
      return torch.tensor(buffer.pop(pop_index))
    
    while read_pointer + self.context_size < len(self.data):
      chunk = self.data[read_pointer: read_pointer + self.context_size]
      if len(chunk) < self.context_size:
        break
      buffer.append(chunk)
      read_pointer += self.stride
      if len(buffer) == self.shuffle_buffer_size:
        batch.append(pop_buffer())
      if len(batch) == self.batch_size:
        batch_tensor = torch.stack(batch)
        batch = []
        yield batch_tensor.long()
    
    while len(buffer) > 0:
      batch.append(pop_buffer())
      if len(batch) == self.batch_size:
        batch_tensor = torch.stack(batch)
        batch = []
        yield batch_tensor.long()
        
    if len(batch) > 0:
      yield torch.stack(batch).long()
      
  def preprocess(examples, tokenizer):
    # Remove unwanted characters
    uc_translation_table = str.maketrans('', '', '�â€œ™')
    texts = [text.translate(uc_translation_table) for text in examples['text']]
    
    # Tokenize text and add EOS token to the end of each sequence
    tokenized_texts = tokenizer(texts, return_attention_mask=False)
    examples['input_ids'] = [example + [tokenizer.eos_token_id] for example in tokenized_texts['input_ids']]
    return examples
      
  def generate_data_file(dataset, file_path, tokenizer, buffer_size=1024):
    
    # Preprocess data
    dataset.map(lambda x: Dataset.preprocess(x, tokenizer), batched=True, remove_columns=['text'])
    file_size = sum([len(example) for example in dataset['input_ids']])
    
    # Initialize memmap array
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    memmap_array = np.memmap(file_path, dtype='int32', mode='w+', shape=(file_size,))
    
    # Write data to memmap array
    buffer = []
    write_pointer = 0
    
    for sequence in tqdm(dataset['input_ids'], desc='Generating dataset files'):
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