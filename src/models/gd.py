import math
import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class GDConfig:
  vocab_size: int
  context_size: int = 256
  d_embed: int = 512
  n_head: int = 8
  n_layer: int = 1
  use_covariate_ff: bool = False
  use_ff: bool = False
  use_ln_out: bool = False
  attn_fn: str = 'softmax'
  wqk: str = 'diag_shared'
  wv: str = 'none'
  
  def get_extension(self):
    return f'{self.d_embed}D_{self.n_layer}L_{self.n_head}H_FF={self.use_ff}_LN_OUT={self.use_ln_out}_COV_FF={self.use_covariate_ff}_ATTN={self.attn_fn}_WQK={self.wqk}_WV={self.wv}'
  
  def __post_init__(self):
    assert self.wqk in ['diag', 'full', 'diag_shared', 'full_shared'], 'Invalid W_qk type, must be "diag," "full," "diag_shared" or "full_shared"'
    assert self.wv in ['none', 'diag', 'full'], 'Invalid W_v type, must be "none," "diag" or "full"'
    assert self.attn_fn in ['softmax', 'linear', 'rbf'], 'Invalid attention function, must be "softmax", "linear" or "rbf"'

class GD(nn.Module):
  
  def __init__(self, config):
    super().__init__()
    
    self.config = config
    self.name = 'GD_' + config.get_extension()
    
    # Embeddings
    self.wte = nn.Embedding(config.vocab_size, config.d_embed)
    self.wpe = nn.Embedding(config.context_size + 1, config.d_embed)
    
    self.ln_e = nn.LayerNorm(config.d_embed, bias=False)
    self.ln_p = nn.LayerNorm(config.d_embed, bias=False)
    
    self.drop_e = nn.Dropout(0.1)
    self.drop_p = nn.Dropout(0.1)
    
    # Krn
    if config.wqk == 'diag':
      self.W_q_diag = nn.Parameter(torch.zeros(config.d_embed))
      self.W_k_diag = nn.Parameter(torch.zeros(config.d_embed))
    elif config.wqk == 'diag_shared':
      self.W_q_diag = self.W_k_diag = nn.Parameter(torch.zeros(config.d_embed))
    elif config.wqk == 'full':
      self.W_q = nn.Parameter(torch.zeros(config.d_embed, config.d_embed))
      self.W_k = nn.Parameter(torch.zeros(config.d_embed, config.d_embed))
    elif config.wqk == 'full_shared':
      self.W_q = self.W_k = nn.Parameter(torch.zeros(config.d_embed, config.d_embed))
      
    if config.attn_fn == 'rbf':
      self.gamma = nn.Parameter(torch.ones(config.n_head, 1, 1))
    
    self.drop_krn = nn.Dropout(0.1)
    
    # GD step
    if config.wv == 'diag':
      self.W_v_diag = nn.Parameter(torch.zeros(config.d_embed))
    elif config.wv == 'full':
      self.W_v = nn.Parameter(torch.zeros(config.d_embed, config.d_embed))
      
    self.W_o_list = nn.ModuleList([nn.Linear(config.d_embed * config.n_head, config.d_embed, bias=False) for _ in range(config.n_layer)]) # Use a different projection matrix (learning rate) for each GD step
    N_reg = 1.0 / torch.arange(1, config.context_size + 1, device=self.wte.weight.device).unsqueeze(1).float() # 1/N term
    self.register_buffer('N_reg', N_reg)
    
    self.drop_gd = nn.Dropout(0.1)
    
    # FF
    if config.use_ff:
      self.ff = nn.Sequential(
        nn.LayerNorm(config.d_embed, bias=False),
        nn.Linear(config.d_embed, 4 * config.d_embed, bias=False),
        nn.GELU(),
        nn.Linear(4 * config.d_embed, config.d_embed, bias=False),
        nn.Dropout(0.1)
      )
      
    if config.use_covariate_ff:
      self.ff_cov = nn.Sequential(
        nn.Linear(config.context_size + 1, 4 * config.d_embed, bias=False),
        nn.GELU(),
        nn.Linear(4 * config.d_embed, config.context_size + 1, bias=False),
        nn.Dropout(0.1)
      )
      
    # Output
    if config.use_ln_out:
      self.ln_out = nn.LayerNorm(config.d_embed, bias=False)
  
    # Weight initialization
    self._init_weights()
    print(f'Initialized model {self.name} with {self.get_num_params()/1e6:.2f}M parameters')
          
  def _init_weights(self):
    nn.init.normal_(self.wte.weight, mean=0, std=0.02)
    nn.init.normal_(self.wpe.weight, mean=0, std=0.02)
    if self.config.wqk == 'diag' or self.config.wqk == 'diag_shared':
      nn.init.normal_(self.W_q_diag, mean=0, std=0.02)
      nn.init.normal_(self.W_k_diag, mean=0, std=0.02)
    elif self.config.wqk == 'full' or self.config.wqk == 'full_shared':
      nn.init.normal_(self.W_q, mean=0, std=0.02)
      nn.init.normal_(self.W_k, mean=0, std=0.02)
    for k in range(self.config.n_layer):
      nn.init.normal_(self.W_o_list[k].weight, mean=0, std=0.02/math.sqrt(2 * self.config.n_layer))
    if self.config.use_ff:
      nn.init.normal_(self.ff[1].weight, mean=0, std=0.02)
      nn.init.normal_(self.ff[3].weight, mean=0, std=0.02)
    if self.config.use_covariate_ff:
      nn.init.normal_(self.ff_cov[0].weight, mean=0, std=0.02)
      nn.init.normal_(self.ff_cov[2].weight, mean=0, std=0.02)
    
  def get_num_params(self):
    num_parameters = sum(p.numel() for p in self.parameters())
    num_parameters -= self.wte.weight.numel() # We don't count our token embedding parameters
    return num_parameters
      
  def gd_step(self, e, krn, f_k, k):
    B, S, _ = e.size()
    
    # Compute weighted average of token embedding vectors
    R = torch.softmax(self.wte.weight @ f_k.transpose(1, 2), dim=-1)
    avg_wte = R.transpose(-1, -2) @ self.wte.weight
    avg_wte = avg_wte / R.sum(dim=1).unsqueeze(-1)
    
    # Subtract weighted average from token embeddings
    V = e - avg_wte
    
    if self.config.wv == 'diag':
      V = V @ torch.diag_embed(self.W_v_diag)
    elif self.config.wv == 'full':
      V = V @ self.W_v
    
    # Compute delta f_k
    delta_f_k = krn @ V.unsqueeze(1)
    delta_f_k = self.N_reg[:S] * delta_f_k
    delta_f_k = self.W_o_list[k](delta_f_k.transpose(1, 2).contiguous().view(B, S, -1))
    
    delta_f_k = self.drop_gd(delta_f_k)
    
    return delta_f_k
  
  def forward(self, x, targets=None):
    
    device = x.device
    B, S = x.size()
    
    # Embeddings
    e = self.wte(x)
    p = self.wpe(torch.arange(0, S + 1, device=device)).repeat(B, 1, 1)
    
    e = self.ln_e(e)
    p = self.ln_p(p)
    
    e = self.drop_e(e)
    p = self.drop_p(p)
    
    if self.config.use_covariate_ff:
      p = p + self.ff_cov(p)
    
    # Krn
    x_i = p[:, :-1, :] # x_i only uses positional embeddings from tokens 1, ..., N
    x_j = p[:, 1:, :] # x_j only uses positional embeddings from tokens 2, ..., N+1
    
    x_i = x_i.repeat(1, 1, self.config.n_head).view(B, S, self.config.n_head, self.config.d_embed).transpose(1, 2)
    x_j = x_j.repeat(1, 1, self.config.n_head).view(B, S, self.config.n_head, self.config.d_embed).transpose(1, 2)
    
    if self.config.wqk == 'diag' or self.config.wqk == 'diag_shared':
      W_q = torch.diag_embed(self.W_q_diag)
      W_k = torch.diag_embed(self.W_k_diag)
    else:
      W_q = self.W_q
      W_k = self.W_k
      
    K = x_i @ W_k
    Q = x_j @ W_q
    
    causal_mask = torch.tril(torch.ones(S, S, device=device), diagonal=0).view(1, S, S).bool().logical_not()
    
    if self.config.attn_fn == 'softmax':
      krn = Q @ K.transpose(-1, -2)
      krn = krn / math.sqrt(self.config.d_embed) # Divide by sqrt(d_embed) for numerical stability, common in attention mechanisms 
      krn = krn.masked_fill(causal_mask, float('-inf'))
      krn = F.softmax(krn, dim=-1)
    elif self.config.attn_fn == 'linear':
      krn = Q @ K.transpose(-1, -2)
      krn = krn / math.sqrt(self.config.d_embed)
      krn = krn.masked_fill(causal_mask, 0)
    elif self.config.attn_fn == 'rbf':
      krn = -torch.cdist(Q, K, p=2).pow(2)
      krn = krn / (-2 * self.gamma + 1e-6) # Add small epsilon for numerical stability
      krn = krn.masked_fill(causal_mask, float('-inf'))
      krn = torch.exp(krn)
    
    krn = self.drop_krn(krn)
    
    # GD steps
    f_k = torch.zeros_like(e, device=device)
    for k in range(self.config.n_layer):
      f_k = f_k + self.gd_step(e, krn, f_k, k)
      
    if targets is None:
      f_k = f_k[:, [-1], :] # Inference-time optimization, only consider the last token
    
    # FF
    if self.config.use_ff:
      f_k = f_k + self.ff(f_k)
    
    # LM Head
    if self.config.use_ln_out:
      f_k = self.ln_out(f_k)
      
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
