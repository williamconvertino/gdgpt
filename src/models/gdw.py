import math
import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class GDWConfig:
  vocab_size: int
  context_size: int = 256
  d_embed: int = 512
  n_head: int = 8
  n_layer: int = 1
  use_ff: bool = False
  wqk: str = 'diag'
  attn_fn: str = 'softmax'
  
  def get_extension(self):
    return f'{self.d_embed}D_{self.n_layer}L_{self.n_head}H_FF={self.use_ff}_wqk={self.wqk}_attn={self.attn_fn}'
  
  def __post_init__(self):
    assert self.wqk in ['diag', 'full'], 'Invalid W_qk type, must be "diag" or "full"'
    assert self.attn_fn in ['softmax', 'linear', 'rbf'], 'Invalid attention function, must be "softmax", "linear" or "rbf"'

class GDW(nn.Module):
  
  def __init__(self, config):
    super().__init__()
    
    self.config = config
    self.name = 'GDW_' + config.get_extension()
    
    # Embeddings
    self.wte = nn.Embedding(config.vocab_size, config.d_embed)
    self.wpe = nn.Embedding(config.context_size + 1, config.d_embed)
    
    # Normalization
    self.ln_e = nn.LayerNorm(config.d_embed, elementwise_affine=False)
    self.ln_p = nn.LayerNorm(config.d_embed, elementwise_affine=False)
    
    # Krn
    if config.wqk == 'diag':
      self.p_W_qk_diag = nn.Parameter(torch.zeros(config.n_head, config.d_embed))
      self.e_W_qk_diag = nn.Parameter(torch.zeros(config.n_head, config.d_embed))
    else:
      self.p_W_qk = nn.Parameter(torch.zeros(config.n_head, config.d_embed, config.d_embed))
      self.e_W_qk = nn.Parameter(torch.zeros(config.n_head, config.d_embed, config.d_embed))
    
    if config.attn_fn == 'rbf':
      self.gamma = nn.Parameter(torch.tensor(config.n_head, 1, 1))
    
    # GD step
    self.p_W_o_list = nn.ModuleList([nn.Linear(config.d_embed * config.n_head, config.d_embed, bias=False) for _ in range(config.n_layer)]) # Use a different projection matrix (learning rate) for each GD step
    self.e_W_o_list = nn.ModuleList([nn.Linear(config.d_embed * config.n_head, config.d_embed, bias=False) for _ in range(config.n_layer)])
    N_reg = 1.0 / torch.arange(1, config.context_size + 1, device=self.wte.weight.device).unsqueeze(1).float() # 1/N term
    self.register_buffer('N_reg', N_reg)
    
    # FF
    if config.use_ff:
      self.ff = nn.Sequential(
        nn.LayerNorm(config.d_embed, elementwise_affine=False),
        nn.Linear(config.d_embed, 4 * config.d_embed, bias=False),
        nn.GELU(),
        nn.Linear(4 * config.d_embed, config.d_embed, bias=False),
        nn.Dropout(0.1)
      )
  
    # Weight initialization
    self._init_weights()
    print(f'Initialized model {self.name} with {self.get_num_params()/1e6:.2f}M parameters')
          
  def _init_weights(self):
    nn.init.normal_(self.wte.weight, mean=0, std=0.02)
    nn.init.normal_(self.wpe.weight, mean=0, std=0.02)
    if self.config.wqk == 'diag':
      nn.init.normal_(self.p_W_qk_diag, mean=0, std=0.02)
      nn.init.normal_(self.e_W_qk_diag, mean=0, std=0.02)
    else:
      nn.init.normal_(self.p_W_qk, mean=0, std=0.02)
      nn.init.normal_(self.e_W_qk, mean=0, std=0.02)
    for k in range(self.config.n_layer):
      nn.init.normal_(self.p_W_o_list[k].weight, mean=0, std=0.02/math.sqrt(2 * self.config.n_layer))
      nn.init.normal_(self.e_W_o_list[k].weight, mean=0, std=0.02/math.sqrt(2 * self.config.n_layer))
    if self.config.use_ff:
      nn.init.normal_(self.ff[1].weight, mean=0, std=0.02)
      nn.init.normal_(self.ff[3].weight, mean=0, std=0.02)
    
  def get_num_params(self):
    num_parameters = sum(p.numel() for p in self.parameters())
    num_parameters -= self.wte.weight.numel() # We don't count our token embedding parameters
    return num_parameters
      
  def gd_step(self, e, p_krn, e_krn, f_k, k):
    B, S, _ = e.size()
    
    # Compute weighted average of token embedding vectors
    R = torch.softmax(self.wte.weight @ f_k.transpose(1, 2), dim=-1)
    avg_wte = R.transpose(-1, -2) @ self.wte.weight
    avg_wte = avg_wte / R.sum(dim=1).unsqueeze(-1)
    
    # Subtract weighted average from token embeddings
    V = e - avg_wte
    
    # Compute delta f_k
    p_delta_f_k = p_krn @ V.unsqueeze(1)
    e_delta_f_k = e_krn @ V.unsqueeze(1)
    
    p_delta_f_k = self.N_reg[:S] * p_delta_f_k
    e_delta_f_k = self.N_reg[:S] * e_delta_f_k
  
    p_delta_f_k = self.p_W_o_list[k](p_delta_f_k.transpose(1, 2).contiguous().view(B, S, -1))
    e_delta_f_k = self.e_W_o_list[k](e_delta_f_k.transpose(1, 2).contiguous().view(B, S, -1))
    
    delta_f_k = p_delta_f_k + e_delta_f_k
    
    return f_k + delta_f_k
  
  def calculate_krn(self, Q, K):
    
    S = Q.size(1)
    device = Q.device
    
    causal_mask = torch.tril(torch.ones(S, S, device=device), diagonal=0).view(1, S, S).bool().logical_not()
    
    if self.config.attn_fn == 'softmax':
      krn = Q @ K.transpose(-1, -2)
      krn = krn / math.sqrt(self.config.d_embed) # Divide by sqrt(d) for numerical stability, common in 
      krn = krn.masked_fill(causal_mask, float('-inf'))
      krn = F.softmax(krn, dim=-1)
    elif self.config.attn_fn == 'linear':
      krn = Q @ K.transpose(-1, -2)
      krn = krn / math.sqrt(self.config.d_embed)
      krn = krn.masked_fill(causal_mask, 0)
    elif self.config.attn_fn == 'rbf':
      krn = -torch.cdist(Q, K, p=2).pow(2)
      krn = krn / (2 * self.gamma + 1e-6) # Add small epsilon for numerical stability
      krn = torch.exp(krn)
      krn = krn.masked_fill(causal_mask, 0)
    return krn
  
  def forward(self, x, targets=None):
    
    device = x.device
    B, S = x.size()
    
    # Embeddings
    e = self.wte(x)
    p = self.wpe(torch.arange(0, S + 1, device=device)).repeat(B, 1, 1)
    
    # Normalization
    e = self.ln_e(e)
    p = self.ln_p(p)
    
    # Krn
    p_x_i = p[:, :-1, :] # x_i only uses tokens 1-N
    p_x_j = p[:, 1:, :] # x_j only uses tokens 2-N+1
    
    e_x_i = e[:, :-1, :] # Both e_x_i and e_x_j use tokens 1-N
    e_x_j = e[:, :-1, :] 
    
    p_x_i = p_x_i.repeat(1, 1, self.config.n_head).view(B, S, self.config.n_head, self.config.d_embed).transpose(1, 2)
    p_x_j = p_x_j.repeat(1, 1, self.config.n_head).view(B, S, self.config.n_head, self.config.d_embed).transpose(1, 2)
    
    e_x_i = e_x_i.repeat(1, 1, self.config.n_head).view(B, S, self.config.n_head, self.config.d_embed).transpose(1, 2)
    e_x_j = e_x_j.repeat(1, 1, self.config.n_head).view(B, S, self.config.n_head, self.config.d_embed).transpose(1, 2)
    
    if self.config.wqk == 'diag':
      p_W_qk = torch.diag_embed(self.p_W_qk_diag)
      e_W_qk = torch.diag_embed(self.e_W_qk_diag)
    else:
      p_W_qk = self.p_W_qk
      e_W_qk = self.e_W_qk
      
    p_K = p_x_i @ p_W_qk
    p_Q = p_x_j @ p_W_qk
    e_K = e_x_i @ e_W_qk
    e_Q = e_x_j @ e_W_qk
    
    p_krn = self.calculate_krn(p_Q, p_K)
    e_krn = self.calculate_krn(e_Q, e_K)
    
    # GD steps
    f_k = torch.zeros_like(e, device=device)
    for k in range(self.config.n_layer):
      f_k = self.gd_step(e, p_krn, e_krn, f_k, k)
      
    # FF
    if self.config.use_ff:
      f_k = f_k + self.ff(f_k)
    
    # LM Head
    logits = f_k @ self.ln_e(self.wte.weight).transpose(0, 1) # Need to use the same normalization as the embeddings for consistency
    
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
