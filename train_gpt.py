# train_gpt.py
# Entraîne le GPT avec la recette de nanoGPT : warmup + cosinus, clipping, AdamW à deux groupes.
# Sauvegarde le MEILLEUR modèle (plus basse validation) : early stopping.
# Lancer depuis le dossier minigpt (bouton ▶ de VS Code)

import csv
import math
import os
import time
import torch
from dataset import CharTokenizer, TextDataset
from gpt import GPT

# ---------- configuration ----------
run_name     = "gpt_ref"     # un nom différent pour chaque run
block_size   = 256         # longueur du contexte
batch_size   = 64
n_embd       = 384         # C
n_head       = 6
n_layer      = 6
dropout      = 0.2
lr           = 1e-3        # learning rate maximal
min_lr       = lr / 10     # learning rate final
warmup       = 100         # pas de montée
max_steps    = 5000
weight_decay = 0.1
grad_clip    = 1.0
eval_every   = 500
N_BATCHES    = 200         # batches par estimation de la perte
seed         = 1337

torch.manual_seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Appareil :", device)


# ---------- planning du learning rate ----------
def get_lr(step):
    """Learning rate au pas step : montée linéaire, puis descente en cosinus."""
    if step < warmup:
        return lr * (step + 1) / warmup                       # montée linéaire jusqu'à lr
    r = (step - warmup) / (max_steps - warmup)                # avancée de la descente, de 0 à 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * r))               # de 1 (début) à 0 (fin)
    return min_lr + coeff * (lr - min_lr)                     # de lr à min_lr


print("Test lr :", get_lr(0), get_lr(warmup - 1), get_lr(warmup), get_lr(max_steps // 2), get_lr(max_steps))

# ---------- données ----------
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()
tok = CharTokenizer(text)
train_ds = TextDataset(text, tok, split="train", block_size=block_size)
valid_ds = TextDataset(text, tok, split="valid", block_size=block_size)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=True)

# ---------- modèle ----------
model = GPT(tok.vocab_size, block_size, n_embd, n_head, n_layer, dropout=dropout).to(device)
print(f"Paramètres : {sum(p.numel() for p in model.parameters()):,}")

# ---------- optimiseur : weight decay sur les matrices seulement ----------
decay    = [p for p in model.parameters() if p.dim() >= 2]   # matrices et embeddings
no_decay = [p for p in model.parameters() if p.dim() < 2]    # biais et LayerNorm
optimizer = torch.optim.AdamW(
    [{"params": decay, "weight_decay": weight_decay},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=lr, betas=(0.9, 0.99),
)


@torch.no_grad()
def estimate_loss(loader, n_batches=N_BATCHES):
    """Perte moyenne sur n_batches batches, en mode eval (dropout éteint)."""
    model.eval()
    losses = []
    for k, (x, y) in enumerate(loader):
        if k == n_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# ---------- avant l'entraînement ----------
estimate_history, step_history = [], []
train_loss, valid_loss = estimate_loss(train_loader), estimate_loss(valid_loader)
estimate_history.append({"step": 0, "train": train_loss, "valid": valid_loss})
print(f"init       : train {train_loss:.4f} | valid {valid_loss:.4f}")

# ---------- checkpoint ----------
def save_checkpoint(step, valid_loss):
    """Sauve les poids, la configuration et le tokenizer : tout ce qu'il faut pour sample.py."""
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "config": {"vocab_size": tok.vocab_size, "block_size": block_size, "n_embd": n_embd,
                   "n_head": n_head, "n_layer": n_layer, "dropout": dropout},
        "stoi": tok.stoi,
        "step": step,
        "valid": valid_loss,
    }, f"checkpoints/{run_name}.pt")


# ---------- entraînement ----------
best_valid, best_step = float("inf"), 0     # meilleure validation vue jusqu'ici
model.train()
step, t0 = 0, time.time()
while step < max_steps:
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        lr_step = get_lr(step)                         # learning rate de ce pas
        for group in optimizer.param_groups:
            group["lr"] = lr_step

        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)   # norme du gradient <= 1
        optimizer.step()

        step_history.append({"step": step + 1, "loss": loss.item(), "lr": lr_step})
        step += 1
        if step % eval_every == 0:
            train_loss, valid_loss = estimate_loss(train_loader), estimate_loss(valid_loader)
            estimate_history.append({"step": step, "train": train_loss, "valid": valid_loss})
            print(f"step {step:5d} : train {train_loss:.4f} | valid {valid_loss:.4f} "
                  f"| lr {lr_step:.2e} | {time.time() - t0:.0f} s")
            if valid_loss < best_valid:                     # le modèle n'a jamais été aussi bon
                best_valid, best_step = valid_loss, step
                save_checkpoint(step, valid_loss)
                print(f"           -> meilleur modèle sauvegardé (valid {valid_loss:.4f})")
        if step == max_steps:
            break

print(f"\nMeilleure validation : {best_valid:.4f} au pas {best_step}  ->  checkpoints/{run_name}.pt")

# ---------- texte généré (par le modèle FINAL ; sample.py utilise le MEILLEUR) ----------
model.eval()                                        # dropout éteint pour générer
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("\nTexte APRÈS :\n", tok.decode(model.generate(context, 500)[0].tolist()))

# ---------- enregistrement des pertes ----------
os.makedirs("results", exist_ok=True)
with open(f"results/{run_name}_eval.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["step", "train", "valid"])
    writer.writeheader()
    writer.writerows(estimate_history)
with open(f"results/{run_name}_steps.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["step", "loss", "lr"])
    writer.writeheader()
    writer.writerows(step_history)
print(f"\nPertes enregistrées : results/{run_name}_eval.csv et _steps.csv")