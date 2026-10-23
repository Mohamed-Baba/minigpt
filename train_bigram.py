# train_bigram.py
# Entraîne le BigramLM sur notre dataset.


import torch
import csv
import os
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
estimate_history = []
step_history = []
N_BATCHES = 200
train_loss,valid_loss = estimate_loss(train_loader, n_batches = N_BATCHES) , estimate_loss(valid_loader, n_batches = N_BATCHES)
estimate_history.append({"step":0 , "train":train_loss , "valid":valid_loss})

print(f"init       : train {train_loss:.4f} | valid {valid_loss:.4f}")
context = torch.zeros((1, 1), dtype=torch.long)          # '\n'
print("Texte AVANT :", tok.decode(model.generate(context, 200)[0].tolist()), "\n")

# ---------- entraînement ----------
max_steps, eval_every = 5000, 500
step = 0
while step < max_steps:
    for x, y in train_loader:

        # 1. remettre les gradients à zéro
        optimizer.zero_grad()
        # 2. forward : logits, loss = ...
        _,loss = model(x,y)
        # 3. backward
        loss.backward()
        # 4. mise à jour des paramètres
        optimizer.step()

        # Enregistrer la loss :
        step_history.append({"step" : step + 1 , "loss": loss.item()})
        step += 1
        if step % eval_every == 0:
            train_loss,valid_loss = estimate_loss(train_loader, n_batches = N_BATCHES) , estimate_loss(valid_loader, n_batches = N_BATCHES)
            estimate_history.append({"step": step , "train":train_loss , "valid":valid_loss})    
            print(f"step {step:5d} : train {train_loss:.4f} | valid {valid_loss:.4f}")
        if step == max_steps:
            break

# ---------- après l'entraînement ----------
run_name = "bigram"   # On met ça pour généraliser notre fichier
os.makedirs("results", exist_ok=True)   # On crée le dossier results s'il n'existe pas

with open(f"results/{run_name}_eval.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["step", "train", "valid"])  # On va convertir notre dictionnaire de clées données en csv
    writer.writeheader()
    writer.writerows(estimate_history)

with open(f"results/{run_name}_steps.csv", "w", newline = "") as f:
    writer = csv.DictWriter(f, fieldnames = ["step" , "loss"])
    writer.writeheader()
    writer.writerows(step_history)

print("\nTexte APRÈS :", tok.decode(model.generate(context, 500)[0].tolist()))