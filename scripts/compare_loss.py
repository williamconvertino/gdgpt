import setup_paths
import os

from src.visualization import visualize_loss_from_files

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../results')

if __name__ == "__main__":
  
  visualize_loss_from_files(
    ('GD_512D_1L_8H_FF=False_LN_OUT=False_ATTN=softmax_WQK=diag_shared_WV=none', 'Dec 16', 'orange'),
    ('GDGPTPlus', 'Dec 12', 'gray'),
    ('CausalGDM', 'Dec 6', 'red'),
    ('GD_512D_1L_8H_FF=True_LN_OUT=False_ATTN=softmax_WQK=diag_shared_WV=none', 'Dec 16 (FF, no ln_out)', 'green'),
    ('GDGPTPlus_ff', 'Dec 12 (FF, no LN out)', 'black'),
    ('CausalGDM_ff', 'Dec 6 (FF)', 'blue'),
    
    title="Model loss comparison",
  )