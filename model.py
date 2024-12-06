from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import LayerNorm, functional as F
import math
import inspect

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Why do we need this? We're gonna divide the embedding dimensions by the heads and concatenate at the end
        assert config.n_embd % config.n_head == 0
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x) # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2) # 3 * (B, T, C)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, H, T, C/H)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, H, T, C/H)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, H, T, C/H)
        
        # (B, H, T, C/H) @ (B, H, C/H, T) = (B, H, T, T)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) 
        # att = att.masked_fill(self.bias[:,:,:T,:T]==0, float("-inf"))
        # att = F.softmax(att, dim=-1)
        # # (B, H, T, T) @ (B, H, T, C/H) = (B, H, T, C/H)
        # y = att @ v
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # (B, T, H, C/H) -> Contiguous -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        #  (B, T, C) * (C, C) = (B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x) # (B, T, 4 * C)
        x = self.gelu(x) # (B, T, 4 * C)
        x = self.c_proj(x) # (B, T, C)
        return x

class Block(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd)
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        std = 0.02
        if hasattr(module, 'NANOGPT_SCALE_INIT'):
            std *= (2 * self.config.n_layer) ** -0.5
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)


    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype = torch.long, device = idx.device) #(T)
        pos_emb = self.transformer.wpe(pos) #(T, n_embd)
        tok_emb = self.transformer.wte(idx) #(B, T, n_embd)
        x = pos_emb + tok_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) #(B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    def configure_optimizer(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"Num of decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"Num of non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"Using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr = learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=fused_available)
        return optimizer

    @torch.no_grad()
    def generate(self, idx, tokenizer, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate sequence starting from <SOT> and stopping at <EOT>.
        
        Args:
            idx: LongTensor of shape (b,t), the conditioning sequence of indices.
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature to scale the logits.
            top_k: If specified, top-k sampling is applied to the logits.
        """
        # Ensure the tokenizer has <SOT> and <EOT> token IDs
        sot_token_id = tokenizer.token_to_id['<SOT>']
        eot_token_id = tokenizer.token_to_id['<EOT>']

        # Initialize sequence with <SOT> if not already provided
        if idx is None:
            idx = torch.tensor([[sot_token_id]], dtype=torch.long).to(next(self.parameters()).device)  # Start from <SOT>

        for _ in range(max_new_tokens):
            # If the sequence context is growing too long, crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward the model to get the logits for the last token in the sequence
            logits, _ = self(idx_cond)
            
            # Get the logits for the last predicted token and scale by temperature
            logits = logits[:, -1, :] / temperature
            
            # Optionally apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Sample the next token from the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append the new token to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
            # If <EOT> token is generated, stop the generation
            if idx_next.item() == eot_token_id:
                break

        return idx