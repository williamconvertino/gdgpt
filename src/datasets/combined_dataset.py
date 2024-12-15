import os
import random
import math
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets

from src.datasets.dataset import Dataset

CS_HUGGINGFACE_PATH = 'ajibawa-2023/Children-Stories-Collection'
TS_HUGGINGFACE_PATH = 'roneneldan/TinyStories'

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/datasets/CombinedDataset')

CS_DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/datasets/ChildrenStories')
TS_DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/datasets/TinyStories')

class CombinedDataset(Dataset):
    
  def __init__(self, tokenizer, split, context_size, stride=0.5, batch_size=64):
    
    self.name = f'CombinedDataset_{split}_({tokenizer.name})'
    
    file_path = f'{DATASET_DIR}/{tokenizer.name}/{split}.bin'    
    
    if not os.path.exists(file_path):
      
      ts_dataset = load_dataset(TS_HUGGINGFACE_PATH, cache_dir=f'{TS_DATASET_DIR}/raw')
      ts_dataset = concatenate_datasets([ts_dataset['train'], ts_dataset['validation']])
      cs_dataset = load_dataset(CS_HUGGINGFACE_PATH, cache_dir=f'{CS_DATASET_DIR}/raw')
      
      print(ts_dataset)
      print(cs_dataset)
      
      dataset = concatenate_datasets([ts_dataset, cs_dataset]).shuffle()
      
      train_test_splits = dataset['train'].train_test_split(test_size=10000, shuffle=True)

      train_dataset = train_test_splits['train']
      test_dataset = train_test_splits['test']
      
      train_val_split = train_dataset.train_test_split(test_size=10000, shuffle=True)
      train_dataset = train_val_split['train']
      val_dataset = train_val_split['test']
      
      Dataset.generate_data_file(train_dataset, f'{DATASET_DIR}/{tokenizer.name}/train.bin', tokenizer)
      Dataset.generate_data_file(test_dataset, f'{DATASET_DIR}/{tokenizer.name}/test.bin', tokenizer)
      Dataset.generate_data_file(val_dataset, f'{DATASET_DIR}/{tokenizer.name}/val.bin', tokenizer)
    
    super().__init__(file_path, context_size)