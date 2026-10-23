# train_bigram.py
# Entraîne le BigramLM sur notre dataset.


import torch
from dataset import CharTokenizer, TextDataset
from model import BigramLM

torch.manual_seed(1337)

# ---------- données ----------
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()
tok = CharTokenizer(text)
train_ds = TextDataset(text, tok, split="train", block_size=8)
valid_ds = TextDataset(text, tok, split="valid", block_size=8)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=32, shuffle=True)

# ---------- modèle et optimiseur ----------
model = BigramLM(tok.vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)


@torch.no_grad()
def estimate_loss(loader, n_batches=200):
    """Perte moyenne sur n_batches batches : une estimation, pas tout le split."""
    model.eval()     
    # Le changement de mode n'affecte pas notre code simple mais c'est une généralisation si forward dépend de l'attribut self.training ou non
    losses = []
    for k, (x, y) in enumerate(loader):
        if k == n_batches:
            break
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# ---------- avant l'entraînement ----------
print(f"init       : train {estimate_loss(train_loader):.4f} | valid {estimate_loss(valid_loader):.4f}")
context = torch.zeros((1, 1), dtype=torch.long)          # '\n'
print("Texte AVANT :", tok.decode(model.generate(context, 200)[0].tolist()), "\n")

# ---------- entraînement ----------
max_steps, eval_every = 5000, 500
step = 0
while step < max_steps:
    for x, y in train_loader:
        # À TOI : les 4 lignes de ton TP
        # 1. remettre les gradients à zéro
        optimizer.zero_grad()
        # 2. forward : logits, loss = ...
        _,loss = model(x,y)
        # 3. backward
        loss.backward()
        # 4. mise à jour des paramètres
        optimizer.step()

        step += 1
        if step % eval_every == 0:
            print(f"step {step:5d} : train {estimate_loss(train_loader):.4f} | valid {estimate_loss(valid_loader):.4f}")
        if step == max_steps:
            break

# ---------- après l'entraînement ----------
print("\nTexte APRÈS :", tok.decode(model.generate(context, 500)[0].tolist()))