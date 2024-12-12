import math
import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class GPTMinConfig:
  vocab_size: int
  context_size: int = 256
  d_embed: int = 512
  n_head: int = 8
  n_layer: int = 1
  use_ff: bool = False
  attn_fn: str = 'softmax'
  
  def get_extension(self):
    return f'{self.d_embed}D_{self.n_layer}L_{self.n_head}H_FF={self.use_ff}_attn={self.attn_fn}'
  
  def __post_init__(self):
    assert self.attn_fn in ['softmax', 'linear', 'rbf'], 'Invalid attention function, must be "softmax", "linear" or "rbf"'

class AttentionBlock(nn.Module):
  
  def __init__(self, config):
    super().__init__()
    
    self.config = config
    
    # Attn
    self.W_q = nn.Parameter(torch.zeros(config.n_head, config.d_embed, config.d_embed))
    self.W_k = nn.Parameter(torch.zeros(config.n_head, config.d_embed, config.d_embed))
    self.W_v = nn.Parameter(torch.zeros(config.n_head, config.d_embed, config.d_embed))
    
    self.W_o = nn.Linear(config.d_embed * config.n_head, config.d_embed, bias=False)
    
    if config.attn_fn == 'rbf':
      self.gamma = nn.Parameter(torch.tensor(config.n_head, 1, 1))
    
    # FF
    if config.use_ff:
      self.ff = nn.Sequential(
        nn.LayerNorm(config.d_embed, elementwise_affine=False),
        nn.Linear(config.d_embed, 4 * config.d_embed, bias=False),
        nn.GELU(),
        nn.Linear(4 * config.d_embed, config.d_embed, bias=False),
        nn.Dropout(0.1)
      )
  
  def _init_weights(self):
    nn.init.normal_(self.W_q, mean=0, std=0.02)
    nn.init.normal_(self.W_k, mean=0, std=0.02)
    nn.init.normal_(self.W_v, mean=0, std=0.02)
    nn.init.normal_(self.W_o.weight, mean=0, std=0.02/math.sqrt(2 * self.config.n_layer))
    if self.config.use_ff:
      nn.init.normal_(self.ff[1].weight, mean=0, std=0.02)
      nn.init.normal_(self.ff[3].weight, mean=0, std=0.02)
      
  def forward(self, x):
    
    B, S, _ = x.size()
    device = x.device
    
    x = x.repeat(1, 1, self.config.n_head).view(B, S, self.config.n_head, self.config.d_embed).transpose(1, 2)
    
    # Attention  
    K = x @ self.W_k
    Q = x @ self.W_q
    V = x @ self.W_v
    
    causal_mask = torch.tril(torch.ones(S, S, device=device), diagonal=0).view(1, S, S).bool().logical_not()
    
    if self.config.attn_fn == 'softmax':
      attn = Q @ K.transpose(-1, -2)
      attn = attn / math.sqrt(self.config.d_embed)
      attn = attn.masked_fill(causal_mask, float('-inf'))
      attn = F.softmax(attn, dim=-1)
    elif self.config.attn_fn == 'linear':
      attn = Q @ K.transpose(-1, -2)
      attn = attn / math.sqrt(self.config.d_embed)
      attn = attn.masked_fill(causal_mask, 0)
    elif self.config.attn_fn == 'rbf':
      attn = -torch.cdist(Q, K, p=2).pow(2)
      attn = attn / (2 * self.gamma + 1e-6) # Add small epsilon for numerical stability
      attn = torch.exp(attn)
      attn = attn.masked_fill(causal_mask, 0)
    
    attn = attn @ V
    
    x = self.W_o(attn.transpose(1, 2).contiguous().view(B, S, -1))
    
    if self.config.use_ff:
      x = x + self.ff(x)
    
    return x
    
class GPTMin(nn.Module):
  
  def __init__(self, config):
    super().__init__()
    
    self.config = config
    self.name = 'GPTMin_' + config.get_extension()
    
    # Embeddings
    self.wte = nn.Embedding(config.vocab_size, config.d_embed)
    self.wpe = nn.Embedding(config.context_size + 1, config.d_embed)
  
    # Attention
    self.attn = nn.ModuleList([AttentionBlock(config) for _ in range(config.n_layer)])
  
    # Weight initialization
    self._init_weights()
    print(f'Initialized model {self.name} with {self.get_num_params()/1e6:.2f}M parameters')
          
  def _init_weights(self):
    nn.init.normal_(self.wte.weight, mean=0, std=0.02)
    nn.init.normal_(self.wpe.weight, mean=0, std=0.02)
    
  def get_num_params(self):
    num_parameters = sum(p.numel() for p in self.parameters())
    num_parameters -= self.wte.weight.numel() # We don't count our token embedding parameters
    return num_parameters
  
  def forward(self, x, targets=None):
    
    device = x.device
    B, S = x.size()
    
    # Embeddings
    e = self.wte(x)
    p = self.wpe(torch.arange(0, S, device=device)).repeat(B, 1, 1)
    
    # Attn
    x = p + e
    
    for attn_block in self.attn:
      x = x + attn_block(x)

    # LM Head
    logits = x @ self.wte.weight.transpose(0, 1)
    
    if targets is None:
      return logits, None
    
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1))
    return logits, loss
    
  def generate(self, x, max_new_tokens=100, eos_token=None):
    
    for _ in range(max_new_tokens):
      
      logits, _ = self(x)
      x_new = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
      x = torch.cat((x, x_new), dim=1)
      
      if eos_token is not None and x_new.item() == eos_token:
        break
    
    return x
  
  def beam_search(self, x, max_new_tokens=100, num_beams=3, eos_token=None):
    
    beams = [{'x': x, 'score': 0, 'eos': False}]  # Initial beam
    
    for _ in range(max_new_tokens):
        
        new_sequences = []
        
        for beam in beams:
          
            # If EOS is already encountered, propagate the beam without changes
            if beam['eos']:
                new_sequences.append(beam)
                continue
            
            # Generate beam candidates
            logits, _ = self(beam['x'])
            topk = torch.topk(logits[:, -1, :], num_beams, dim=-1)
            
            for i in range(num_beams):
                idx_next = topk.indices[0, i].unsqueeze(0).unsqueeze(0)
                score = topk.values[0, i].item()
                new_x = torch.cat((beam['x'], idx_next), dim=1)
                new_eos = eos_token is not None and idx_next.item() == eos_token
                new_sequences.append({
                    'x': new_x,
                    'score': beam['score'] + score,
                    'eos': new_eos
                })
        
        # Select beam based on normalized score
        new_sequences.sort(key=lambda seq: seq['score'] / (len(seq['x'][0]) + 1), reverse=True)
        beams = new_sequences[:num_beams]
        
        # Break early if all beams have encountered EOS
        if all(beam['eos'] for beam in beams):
            break
    
    most_probable_sequence = max(beams, key=lambda seq: seq['score'] / (len(seq['x'][0]) + 1))
    return most_probable_sequence['x']
