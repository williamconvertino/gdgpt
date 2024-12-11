import math
import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class GDGPTUltraConfig:
  vocab_size: int
  context_size: int = 256
  d_embed: int = 512
  n_head: int = 8
  n_layer: int = 1
  use_ff: bool = False
  attn_fn: str = 'softmax'
  
  def get_extension(self):
    return f'{self.n_layer}L_{self.n_head}H_FF={self.use_ff}_attn={self.attn_fn}'
  
  def __post_init__(self):
    assert self.attn_fn in ['softmax', 'linear', 'rbf'], 'Invalid attention function'

class GDGPTUltra(nn.Module):
  
  def __init__(self, config):
    super().__init__()
    
    self.config = config
    self.name = 'GDGPTUltra_' + config.get_extension()
    
    # Embeddings
    self.wte = nn.Embedding(config.vocab_size, config.d_embed)
    self.wpe = nn.Embedding(config.context_size + 1, config.d_embed)
    
    # Regularization
    self.drop_e = nn.Dropout(0.1)
    self.drop_p = nn.Dropout(0.1)
    self.ln_e = nn.LayerNorm(config.d_embed, bias=False)
    self.ln_p = nn.LayerNorm(config.d_embed, bias=False)
    self.ln_f = nn.LayerNorm(config.d_embed, bias=False)
    
    # Krn
    self.W_q_diag = nn.Parameter(torch.zeros(config.n_head, config.d_embed))
    self.W_k_diag = nn.Parameter(torch.zeros(config.n_head, config.d_embed))
    
    # GD step
    self.W_o_list = nn.ModuleList([nn.Linear(config.d_embed * config.n_head, config.d_embed, bias=False) for _ in range(config.n_layer)]) # Use a different projection matrix (learning rate) for each GD step
    N_reg = 1.0 / torch.arange(1, config.context_size + 1, device=self.wte.weight.device).unsqueeze(1).float() # 1/N term
    self.register_buffer('N_reg', N_reg)
    
    # FF
    if config.use_ff:
      self.ff = nn.Sequential(
        nn.Linear(config.d_embed, 4 * config.d_embed),
        nn.GELU(),
        nn.Linear(4 * config.d_embed, config.d_embed)
      )
    
    # LM Head
    self.lm_head = nn.Linear(config.d_embed, config.vocab_size, bias=False)
    self.wte.weight = self.lm_head.weight # Weight tying, crucial for GD
    
    # Weight initialization
    self._init_weights()
    print(f'Initialized model {self.name} with {self.get_num_params()/1e6:.2f}M parameters')
          
  def _init_weights(self):
    nn.init.normal_(self.wte.weight, mean=0, std=0.02)
    nn.init.normal_(self.wpe.weight, mean=0, std=0.02)
    nn.init.normal_(self.W_q_diag, mean=0, std=0.02)
    nn.init.normal_(self.W_k_diag, mean=0, std=0.02)
    nn.init.normal_(self.lm_head.weight, mean=0, std=0.02)
    for k in range(self.config.n_layer):
      nn.init.normal_(self.W_o_list[k].weight, mean=0, std=0.02/math.sqrt(2 * self.config.n_layer))
    
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
    V = (e - avg_wte)
    
    # Compute delta f_k
    delta_f_k = krn @ V.unsqueeze(1)
    delta_f_k = self.N_reg[:S] * delta_f_k
    delta_f_k = self.W_o_list[k](delta_f_k.transpose(1, 2).contiguous().view(B, S, -1))
    
    return f_k + delta_f_k
  
  def forward(self, x, targets=None):
    
    device = x.device
    B, S = x.size()
    
    # Embeddings
    e = self.wte(x)
    p = self.wpe(torch.arange(0, S + 1, device=device)).repeat(B, 1, 1)
    
    # Regularization
    e = self.drop_e(e)
    p = self.drop_p(p)
    e = self.ln_e(e)
    p = self.ln_p(p)
    
    # Krn
    x_i = p[:, :-1, :] # x_i only uses tokens 1-N
    x_j = p[:, 1:, :] # x_j only uses tokens 2-N+1
    
    x_i = x_i.repeat(1, 1, self.config.n_head).view(B, S, self.config.n_head, self.config.d_embed).transpose(1, 2)
    x_j = x_j.repeat(1, 1, self.config.n_head).view(B, S, self.config.n_head, self.config.d_embed).transpose(1, 2)
    
    W_q = torch.diag_embed(self.W_q_diag)
    W_k = torch.diag_embed(self.W_k_diag)
    
    K = x_i @ W_k
    Q = x_j @ W_q
    
    causal_mask = torch.tril(torch.ones(S, S, device=e.device), diagonal=0).view(1, S, S).bool().logical_not()
    
    krn = Q @ K.transpose(-1, -2) / math.sqrt(self.config.d_embed) # Divide by sqrt(d) for numerical stability, common in GPT
    krn = krn.masked_fill(causal_mask, float('-inf'))
    krn = F.softmax(krn, dim=-1)
    
    # GD steps
    f_k = torch.zeros_like(e, device=device)
    for k in range(self.config.n_layer):
      f_k = self.gd_step(e, krn, f_k, k)
      
    # FF
    if self.config.use_ff:
      f_k = self.ff(f_k)
    
    # LM Head
    f_k = self.ln_f(f_k)
    logits = self.lm_head(f_k)
    
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
