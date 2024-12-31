import math
import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
  vocab_size: int
  context_size: int = 256
  d_embed: int = 512
  n_head: int = 8
  n_layer: int = 1
  use_covariate_ff: bool = False
  use_ln_out: bool = True
  use_ff: bool = True
  use_ppe: bool = False
  attn_fn: str = 'softmax'
  wqk: str = 'full'
  wv: str = 'full'

  def get_extension(self):
    name =  f'{self.d_embed}D_{self.n_layer}L_{self.n_head}H_FF={self.use_ff}_LN_OUT={self.use_ln_out}_ATTN={self.attn_fn}_WQK={self.wqk}_WV={self.wv}'
    if self.use_covariate_ff:
      name += '_COV_FF'
    if self.use_ppe:
      name += '_PPE'
    return name
  
  def __post_init__(self):
    assert self.wqk in ['diag', 'full', 'diag_shared', 'full_shared'], 'Invalid W_qk type, must be "diag," "full," "diag_shared" or "full_shared"'
    assert self.wv in ['none', 'diag', 'full'], 'Invalid W_v type, must be "none," "diag" or "full"'
    assert self.attn_fn in ['softmax', 'linear', 'rbf'], 'Invalid attention function, must be "softmax", "linear" or "rbf"'

class Attention(nn.Module):

  def __init__(self, config):
    super().__init__()
    
    self.config = config

    if self.config.use_ppe:
      self.ln_p = nn.LayerNorm(config.d_embed, bias=False)
      self.ln_e = nn.LayerNorm(config.d_embed, bias=False)
    else:
      self.ln_x = nn.LayerNorm(config.d_embed, bias=False)

    self.W_q = nn.Linear(config.d_embed, config.n_head * config.d_embed, bias=False)
    self.W_k = nn.Linear(config.d_embed, config.n_head * config.d_embed, bias=False)
    self.W_v = nn.Linear(config.d_embed, config.n_head * config.d_embed, bias=False)
    self.W_o = nn.Linear(config.n_head * config.d_embed, config.d_embed, bias=False)

    self.dropout_attn = nn.Dropout(config.dropout)
    self.dropout_o = nn.Dropout(config.dropout)
    
    self._init_weights()

  def _init_weights(self):
    nn.init.normal_(self.W_q.weight, std=0.02)
    nn.init.normal_(self.W_k.weight, std=0.02)
    nn.init.normal_(self.W_v.weight, std=0.02)
    nn.init.normal_(self.W_o.weight, std=0.02 / math.sqrt(2 * self.config.n_layer))

  def forward(self, x, e, p):
    device = x.device
    B, S, E = x.size()
    
    if self.config.use_ppe:
      e = self.ln_e(e)
      p = self.ln_p(p)
      Q = self.W_q(p).view(B, S, self.config.n_head, self.config.d_embed).transpose(1, 2)
      K = self.W_k(p).view(B, S, self.config.n_head, self.config.d_embed).transpose(1, 2)
      V = self.W_v(e).view(B, S, self.config.n_head, self.config.d_embed).transpose(1, 2)
    else:
      x = self.ln_x(x)
      Q = self.W_q(x).view(B, S, self.config.n_head, self.config.d_embed).transpose(1, 2)
      K = self.W_k(x).view(B, S, self.config.n_head, self.config.d_embed).transpose(1, 2)
      V = self.W_v(x).view(B, S, self.config.n_head, self.config.d_embed).transpose(1, 2)
    
    causal_mask = torch.tril(torch.ones(S, S, device=device), diagonal=0).view(1, S, S).bool().logical_not()
    
    if self.config.attn_fn == 'softmax':
      attn = Q @ K.transpose(-1, -2)
      attn = attn / math.sqrt(self.config.d_embed) # Divide by sqrt(d_embed) for numerical stability, common in attention mechanisms 
      attn = attn.masked_fill(causal_mask, float('-inf'))
      attn = F.softmax(attn, dim=-1)
    elif self.config.attn_fn == 'linear':
      attn = Q @ K.transpose(-1, -2)
      attn = attn / math.sqrt(self.config.d_embed)
      attn = attn.masked_fill(causal_mask, 0)
    elif self.config.attn_fn == 'rbf':
      attn = -torch.cdist(Q, K, p=2).pow(2)
      attn = attn / (-2 * self.gamma + 1e-6) # Add small epsilon for numerical stability
      attn = attn.clamp(min=-10, max=10) # Clamp to avoid numerical instability
      attn = attn.masked_fill(causal_mask, float('-inf'))
      attn = torch.exp(attn)
    
    attn = self.drop_attn(attn)
    
    attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.config.n_head * self.config.d_embed)
    
    attn_output = self.W_o(attn_output)
    attn_output = self.dropout_o(attn_output)
    
    return attn_output
  
class TransformerBlock(nn.Module):

  def __init__(self, config):
    super().__init__()
    
    # Attention
    self.attn = Attention(config)
    
    # Feed Forward
    if config.use_ff:
      self.ff = nn.Sequential(
        nn.LayerNorm(config.d_embed, bias=False),
        nn.Linear(config.d_embed, config.d_ff, bias=False),
        nn.GELU(),
        nn.Linear(config.d_ff, config.d_embed, bias=False),
        nn.Dropout(config.dropout)
      )
    
    self._init_weights()
    
  def _init_weights(self):
    if self.config.use_ff:
      nn.init.normal_(self.ff[1].weight, std=0.02)
      nn.init.normal_(self.ff[3].weight, std=0.02)

  def forward(self, x, e, p):
    x = x + self.attn(x, e, p)	
    if self.config.use_ff:
      x = x + self.ff(x)
    return x

class GPT(nn.Module):

  def __init__(self, config):
    super().__init__()

    self.config = config
    self.name = f'GPT_{config.get_extension()}'
    
    # Embedding
    self.W_e = nn.Embedding(config.vocab_size, config.d_embed)
    self.W_p = nn.Embedding(config.context_size, config.d_embed)
    
    self.dropout_e = nn.Dropout(config.dropout)
    self.dropout_p = nn.Dropout(config.dropout)
    
    # Attention Blocks
    self.attn_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])

    # Output
    self.ln_out = nn.LayerNorm(config.d_embed, bias=False)
    self.lm_head = nn.Linear(config.d_embed, config.vocab_size, bias=False)
    self.W_e.weight = self.lm_head.weight # Weight tying
    
    # Initialize weights
    self._init_weights()

    # Parameter count
    self.n_param = sum(p.numel() for p in self.parameters()) - sum(p.numel() for p in self.W_e.parameters()) - sum(p.numel() for p in self.W_p.parameters())
    print(f'Initialized {self.name} with {self.n_param/1e6:.2f}M parameters')

  def _init_weights(self):
    nn.init.normal_(self.W_e.weight, std=0.02)
    nn.init.normal_(self.W_p.weight, std=0.02)

  def forward(self, x, targets=None):
    
    device = x.device
    B, S = x.size()
    
    e = self.W_e(x)
    p = self.W_p(torch.arange(S, device=device))
    
    e = self.dropout_e(e)
    p = self.dropout_p(p).unsqueeze(0).expand(B, -1, -1)
    
    x = e + p

    for attn_block in self.attn_blocks:
      x = attn_block(x, e, p)

    if self.config.use_ln_out:
      x = self.ln_out(x)

    if targets is None:
      logits = self.lm_head(x)
      loss = None
    elif self.config.use_nto:
      targets = targets[:, -1].contiguous()
      logits = self.lm_head(x)[:, -1,:]
      loss = F.cross_entropy(logits, targets)
    else:
      logits = self.lm_head(x)
      targets = targets.contiguous()
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
      
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
                if idx_next.item() == eos_token:
                  new_sequences.append({
                      'x': beam['x'],
                      'score': beam['score'],
                      'eos': True
                    })
                else:    
                  new_x = torch.cat((beam['x'], idx_next), dim=1)
                  new_sequences.append({
                      'x': new_x,
                      'score': beam['score'] + score,
                      'eos': False
                  })
        
        # Select beam based on normalized score
        new_sequences.sort(key=lambda seq: seq['score'] / (len(seq['x'][0]) + 1), reverse=True)
        beams = new_sequences[:num_beams]
        
        # Break early if all beams have encountered EOS
        if all(beam['eos'] for beam in beams):
            break
    
    most_probable_sequence = max(beams, key=lambda seq: seq['score'] / (len(seq['x'][0]) + 1))
    
    return most_probable_sequence['x']
