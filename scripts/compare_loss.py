import setup_paths
import os

from src.visualization import visualize_loss_from_files

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../results')

if __name__ == "__main__":
  
  visualize_loss_from_files(
    ('name', 'label', 'b'),
    ('name', 'label', 'r'),
  )