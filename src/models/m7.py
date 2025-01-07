import math
import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class M7Config:
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
  use_skip: bool = True

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

  def __init__(self, config, wte):
    super().__init__()
    
    self.config = config
    self.wte = wte

    if self.config.use_ppe:
      self.ln_p = nn.LayerNorm(config.d_embed, bias=False)
      self.ln_e = nn.LayerNorm(config.d_embed, bias=False)
    else:
      self.ln_x = nn.LayerNorm(config.d_embed, bias=False)

    # self.W_q = nn.Linear(config.d_embed, config.n_head * config.d_embed, bias=False)
    # self.W_k = nn.Linear(config.d_embed, config.n_head * config.d_embed, bias=False)
    # self.W_v = nn.Linear(config.d_embed, config.n_head * config.d_embed, bias=False)
    
    self.W_q = self.W_k = nn.Parameter(torch.zeros(config.n_head, config.d_embed))

    self.W_o = nn.Linear(config.n_head * config.d_embed, config.d_embed, bias=False)
    
    N_reg = 1.0 / torch.arange(1, config.context_size + 1, device=self.wte.weight.device).unsqueeze(1).float() # 1/N term
    self.register_buffer('N_reg', N_reg)
    

    if config.attn_fn == 'rbf':
      self.gamma = nn.Parameter(torch.tensor(1.0))

    self.dropout_attn = nn.Dropout(0.1)
    self.dropout_o = nn.Dropout(0.1)
    
    self._init_weights()

  def _init_weights(self):
    nn.init.normal_(self.W_q, std=0.02)
    nn.init.normal_(self.W_k, std=0.02)
    if self.config.attn_fn == 'rbf':
      nn.init.normal_(self.gamma, std=0.02)
    nn.init.normal_(self.W_o.weight, std=0.02 / math.sqrt(2 * self.config.n_layer))

  def E_wte(self, x):
    f_k = torch.zeros_like(x, device=x.device)
    R = torch.softmax(self.wte.weight @ f_k.transpose(1, 2), dim=-1)
    avg_wte = R.transpose(-1, -2) @ self.wte.weight
    avg_wte = avg_wte / R.sum(dim=1).unsqueeze(-1)
    return avg_wte

  def forward(self, x, e, p):
    device = x.device
    B, S, E = x.size()
    
    x = self.ln_x(x)
    E_wte = self.E_wte(x)
    
    x = x.repeat(1, 1, self.config.n_head).view(B, S, self.config.n_head, self.config.d_embed).transpose(1, 2)
    E_wte = E_wte.repeat(1, 1, self.config.n_head).view(B, S, self.config.n_head, self.config.d_embed).transpose(1, 2)
    
    Q = x @ torch.diag_embed(self.W_q)
    K = x @ torch.diag_embed(self.W_k)
    V = x - E_wte

    # Q = self.W_q(x).view(B, S, self.config.n_head, self.config.d_embed).transpose(1, 2)
    # K = self.W_k(x).view(B, S, self.config.n_head, self.config.d_embed).transpose(1, 2)
    # V = self.W_v(x).view(B, S, self.config.n_head, self.config.d_embed).transpose(1, 2)
    
    # attn_output = F.scaled_dot_product_attention(Q, K, V, is_causal=True, dropout_p=0.1 if self.training else 0)

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
      attn = self.gamma * (attn / -2) # Add small epsilon for numerical stability
      attn = attn.clamp(min=-10, max=10) # Clamp to avoid numerical instability
      attn = attn.masked_fill(causal_mask, float('-inf'))
      attn = torch.exp(attn)
    
    attn = self.dropout_attn(attn)
    
    attn_output = attn @ V

    attn_output = self.N_reg[:S] * attn_output

    attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.config.n_head * self.config.d_embed)
    
    attn_output = self.W_o(attn_output)
    attn_output = self.dropout_o(attn_output)
    
    return attn_output
  
class TransformerBlock(nn.Module):

  def __init__(self, config, wte):
    super().__init__()
    
    self.config = config

    # Attention
    self.attn = Attention(config, wte)
    
    # Feed Forward
    if config.use_ff:
      self.ff = nn.Sequential(
        nn.LayerNorm(config.d_embed, bias=False),
        nn.Linear(config.d_embed, 4 * config.d_embed, bias=False),
        nn.GELU(),
        nn.Linear(4 * config.d_embed, config.d_embed, bias=False),
        nn.Dropout(0.1)
      )
    
    self._init_weights()
    
  def _init_weights(self):
    if self.config.use_ff:
      nn.init.normal_(self.ff[1].weight, std=0.02)
      nn.init.normal_(self.ff[3].weight, std=0.02)

  def forward(self, x, e, p):
    if self.config.use_skip:
      x = x + self.attn(x, e, p)	
    else:
      x = self.attn(x, e, p)
    if self.config.use_ff:
      x = x + self.ff(x)
    return x

class M7(nn.Module):

  def __init__(self, config):
    super().__init__()

    self.config = config
    self.name = f'M7_{config.get_extension()}'
    
    # Embedding
    self.W_e = nn.Embedding(config.vocab_size, config.d_embed)
    self.W_p = nn.Embedding(config.context_size, config.d_embed)
    
    self.dropout_e = nn.Dropout(0.1)
    self.dropout_p = nn.Dropout(0.1)
    
    # Attention Blocks
    self.attn_blocks = nn.ModuleList([TransformerBlock(config, self.W_e) for _ in range(config.n_layer)])

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
