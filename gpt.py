# gpt.py
import torch


class Head(torch.nn.Module):
    """Une tête d'attention causale."""

    def __init__(self, n_embd, head_size, block_size , dropout = 0.0):
        super().__init__()
        # 1. Les trois projections W_Q, W_K, W_V : C -> hs, sans biais
        self.key = torch.nn.Linear(in_features= n_embd , out_features= head_size , bias = False)
        self.query = torch.nn.Linear(in_features= n_embd , out_features= head_size , bias = False)
        self.value = torch.nn.Linear(in_features= n_embd , out_features= head_size , bias = False)
        self.dropout = torch.nn.Dropout(dropout) 
        # Le masque triangulaire [block_size, block_size]
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        # c'est exactement l'equivalent de self.tril = torch.tril(torch.ones(block_size, block_size)) mais juste buffer : suit le modèle (GPU, checkpoints) mais n'est pas un paramètre entraîné

    def forward(self, x):            # x : [B, T, C]
        B, T, C = x.shape
        k = self.key(x)                     # [B, T, hs]
        q = self.query(x)                      # [B, T, hs]
        v = self.value(x)                      # [B, T, hs]
        wei = q @ k.transpose(-2,-1) / k.shape[-1]**0.5                   # 2. scores : [B, T, T]
        wei = wei.masked_fill(self.tril[:T,:T] == 0 , float("-inf"))                    # 3a. masque causal
        # le mask créé de booléen fait changer ces True en valeur : wei.masked_fill(mask,val)
        wei = torch.nn.functional.softmax(wei,dim = -1)                    # 3b. softmax sur chaque ligne
        wei = self.dropout(wei)
        out = wei @ v                   # 4. moyenne pondérée : [B, T, hs]
        return out
    

class MultiHeadAttention(torch.nn.Module):
    """n_head têtes d'attention en parallèle, concaténées puis projetées."""

    def __init__(self, n_embd, n_head, block_size ,dropout = 0.0):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd doit être divisible par n_head"
        head_size = n_embd // n_head                     # taille de chaque tête
        self.heads = torch.nn.ModuleList([Head(n_embd,head_size,block_size , dropout = dropout) for _ in range(n_head)])                    # n_head objets Head, dans un torch.nn.ModuleList
        self.proj = torch.nn.Linear(n_head * head_size,n_embd)                     # W_O : C -> C
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):                   # [B, T, C]
        out = torch.cat([head(x) for head in self.heads] , dim = -1)                           # concaténer les sorties des têtes : [B, T, C]
        out = self.proj(out)                           # projeter avec W_O : [B, T, C]
        out = self.dropout(out)
        return out
    

class FeedForward(torch.nn.Module):
    """MLP appliqué à chaque position séparément : C -> 4C -> C."""

    def __init__(self, n_embd , n_hidden = None , dropout = 0.0):
        super().__init__()
        if n_hidden is None :
            n_hidden = 4 * n_embd                              # Convention chatGPT 2 
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_embd , n_hidden ),                 # C -> 4C
            torch.nn.GELU(),                 # GELU
            torch.nn.Linear( n_hidden , n_embd),                 # 4C -> C
            torch.nn.Dropout(dropout)
        )

    def forward(self, x):        # [B, T, C]
        return self.net(x)       # [B, T, C]
    

class Block(torch.nn.Module):
    """Un bloc transformer en pré-norme : communiquer (attention), puis calculer (FeedForward)."""

    def __init__(self, n_embd, n_head, block_size , dropout = 0.0):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(n_embd)                 
        self.attn = MultiHeadAttention(n_embd,n_head,block_size , dropout = dropout)                 
        self.ln2 = torch.nn.LayerNorm(n_embd)         
        self.ffwd = FeedForward(n_embd , dropout = dropout)

    def forward(self, x):               # [B, T, C]
        x = x + self.attn(self.ln1(x))                         # x + attention(LN1(x))
        x = x + self.ffwd(self.ln2(x))                         # x + feedforward(LN2(x))
        return x                        # [B, T, C]


class GPT(torch.nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer , dropout = 0.0):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = torch.nn.Embedding(vocab_size , n_embd)                  # Embedding : V -> C
        self.pos_emb = torch.nn.Embedding(block_size , n_embd )                  # Embedding : block_size -> C
        self.blocks = torch.nn.Sequential(
            *[Block(n_embd, n_head, block_size,dropout) for _ in range(n_layer)]
        )
        self.ln_f = torch.nn.LayerNorm(n_embd)                     # LayerNorm finale
        self.lm_head = torch.nn.Linear(n_embd , vocab_size)                  # Linear : C -> V
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, idx, targets=None):   # idx : [B, T]
        B, T = idx.shape
        tok = self.tok_emb(idx)                                                  # [B, T, C]
        pos = self.pos_emb(torch.arange(T, device=idx.device))     # [T, C]
        x = tok + pos                                                    # [B, T, C]
        x = self.dropout(x)                                                # dropout pour bruiter 
        x = self.blocks(x)                                                    # les blocs
        x = self.ln_f(x)                                                    # LayerNorm finale
        logits = self.lm_head(x)                                               # [B, T, V]
        if targets is None:
            return logits, None
        B, T, V = logits.shape
        loss = torch.nn.functional.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx : [B, t]  ->  [B, t + max_new_tokens]
        for _ in range(max_new_tokens):
            idx_cond = idx[:,-self.block_size:]      # 1. les block_size derniers tokens de idx : [B, min(t, block_size)]
            logits, _ = self(idx_cond)               # 2. appeler le modèle sur idx_cond : [B, T, V]
            logits = logits[:,-1,:]                  # 3. garder la dernière position : [B, V]
            probs = torch.nn.functional.softmax(logits , dim = -1 )         # 4. softmax sur le vocabulaire : [B, V]
            idx_next = torch.multinomial(probs , num_samples = 1)                       # 5. tirer un token par ligne : [B, 1]
            idx = torch.cat([idx , idx_next] ,dim = -1)                                            # 6. coller idx_next à la fin de idx : [B, t+1]
        return idx