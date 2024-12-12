import os
import random
import math
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

from src.datasets.dataset import Dataset

HUGGINGFACE_PATH = 'roneneldan/TinyStories'

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/datasets/TinyStories')

class TinyStoriesDataset(Dataset):
    
  def __init__(self, tokenizer, split, context_size):
    
    self.name = f'TinyStories_{split}_({tokenizer.name})'
    
    file_path = f'{DATASET_DIR}/{tokenizer.name}/{split}.bin'    
    
    if not os.path.exists(file_path):
      
      dataset = load_dataset(HUGGINGFACE_PATH, cache_dir=f'{DATASET_DIR}/raw')
      train_test_splits = dataset['train'].train_test_split(test_size=20000, shuffle=True)
      
      train_dataset = train_test_splits['train']
      test_dataset = train_test_splits['test']
      val_dataset = dataset['validation']
      
      Dataset.generate_data_file(train_dataset, f'{DATASET_DIR}/{tokenizer.name}/train.bin', tokenizer)
      Dataset.generate_data_file(test_dataset, f'{DATASET_DIR}/{tokenizer.name}/test.bin', tokenizer)
      Dataset.generate_data_file(val_dataset, f'{DATASET_DIR}/{tokenizer.name}/val.bin', tokenizer)
    
    super().__init__(file_path, context_size)