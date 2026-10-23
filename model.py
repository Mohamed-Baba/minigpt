# model.py
import torch


class BigramLM(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # La matrice W du PDF : ligne a = logits du caractère qui suit a
        self.table = torch.nn.Embedding(vocab_size , vocab_size)

    def forward(self, idx, targets=None):
        # idx : [B, T]  ->  logits : [B, T, V]
        logits = self.table(idx)
        if targets is None:
            return logits, None
        B, T, V = logits.shape
        loss = torch.nn.functional.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx : [B, T]  ->  [B, T + max_new_tokens]
        for _ in range(max_new_tokens):
            logits, _ = self(idx)   # [B, T, V]
            logits = logits[:,-1,:]            # [B, V]    dernière position
            probs = torch.nn.functional.softmax(logits , dim = 1)             # [B, V]    softmax sur les 65 scores
            idx_next = torch.multinomial(probs,num_samples = 1)          # [B, 1]    tirage avec multinomial
            idx = torch.cat([idx , idx_next] , dim = 1)               # [B, T+1]  concaténation sur le temps
        return idx