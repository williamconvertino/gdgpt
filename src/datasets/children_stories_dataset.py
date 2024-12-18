import os
import random
import math
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

from src.datasets.dataset import Dataset

HUGGINGFACE_PATH = 'ajibawa-2023/Children-Stories-Collection'

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/datasets/ChildrenStories')

class ChildrenStoriesDataset(Dataset):
    
  def __init__(self, tokenizer, split, context_size, stride=0.5, batch_size=64):
    
    self.name = f'ChildrenStories_{split}_({tokenizer.name})'
    
    file_path = f'{DATASET_DIR}/{tokenizer.name}/{split}.bin'    
    
    if not os.path.exists(file_path):
      
      print(f'Creating {self.name} dataset file...')
      
      dataset = load_dataset(HUGGINGFACE_PATH, cache_dir=f'{DATASET_DIR}/raw')['train']
      
      train_test_splits = dataset.train_test_split(test_size=10000, shuffle=True)

      train_dataset = train_test_splits['train']
      test_dataset = train_test_splits['test']
      
      train_val_split = train_dataset.train_test_split(test_size=10000, shuffle=True)
      train_dataset = train_val_split['train']
      val_dataset = train_val_split['test']
      
      Dataset.generate_data_file(train_dataset, f'{DATASET_DIR}/{tokenizer.name}/train.bin', tokenizer)
      Dataset.generate_data_file(test_dataset, f'{DATASET_DIR}/{tokenizer.name}/test.bin', tokenizer)
      Dataset.generate_data_file(val_dataset, f'{DATASET_DIR}/{tokenizer.name}/val.bin', tokenizer)
    
    super().__init__(file_path, context_size)