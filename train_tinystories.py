# train_tinystories.py
# Entraîne le GPT sur TinyStories-Instruct (fichiers .bin produits par prepare_data.py).
# Même recette que train_gpt.py ; ce qui change : les données (memmap), le tirage des fenêtres,
# pas de dropout (2,66 milliards de tokens : pas de surapprentissage), bf16 et une sauvegarde régulière.
# Lancer depuis le dossier minigpt (bouton ▶ de VS Code)

import csv
import json
import math
import os
import time
import torch
from bin_dataset import BinDataset
from gpt import GPT

# ---------- configuration ----------
run_name     = "ts_ref"    # "ts_check" pour le run court de vérification, puis "ts_ref" pour le vrai
data_dir     = "data/tinystories_instruct"
block_size   = 256
batch_size   = 64
n_embd       = 384
n_head       = 6
n_layer      = 6
dropout      = 0.0          # corpus énorme : chaque token n'est vu qu'une fois, pas besoin de dropout
lr           = 1e-3
min_lr       = lr / 10
warmup       = 200
max_steps    = 20000         # 1000 pour la vérification ; ~20000 pour le vrai run
weight_decay = 0.1
grad_clip    = 1.0
eval_every   = 1000          # 250 pour la vérification ; 1000 pour le vrai run
N_BATCHES    = 100
use_bf16     = True         # calculs en bfloat16 : plus rapide, moins de mémoire
seed         = 1337

torch.manual_seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
use_bf16 = use_bf16 and device == "cuda"
print("Appareil :", device, "| bf16 :", use_bf16)


def get_lr(step):
    """Learning rate au pas step : montée linéaire, puis descente en cosinus."""
    if step < warmup:
        return lr * (step + 1) / warmup
    r = (step - warmup) / (max_steps - warmup)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * r)) * (lr - min_lr)


# ---------- données ----------
with open(os.path.join(data_dir, "meta.json")) as f:
    meta = json.load(f)
stoi = meta["stoi"]
itos = {i: c for c, i in stoi.items()}
vocab_size = meta["vocab_size"]

train_ds = BinDataset(os.path.join(data_dir, "train.bin"), block_size)
valid_ds = BinDataset(os.path.join(data_dir, "val.bin"), block_size)
print(f"Train : {len(train_ds):,} fenêtres | Valid : {len(valid_ds):,} fenêtres")

# Tirage AVEC remise : shuffle=True créerait une permutation de 2,66 milliards d'indices (20 Go).
train_sampler = torch.utils.data.RandomSampler(train_ds, replacement=True, num_samples=max_steps * batch_size)
valid_sampler = torch.utils.data.RandomSampler(valid_ds, replacement=True, num_samples=10**8)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, sampler=valid_sampler)

# ---------- modèle et optimiseur ----------
model = GPT(vocab_size, block_size, n_embd, n_head, n_layer, dropout=dropout).to(device)
print(f"Paramètres : {sum(p.numel() for p in model.parameters()):,}")

decay    = [p for p in model.parameters() if p.dim() >= 2]
no_decay = [p for p in model.parameters() if p.dim() < 2]
optimizer = torch.optim.AdamW(
    [{"params": decay, "weight_decay": weight_decay},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=lr, betas=(0.9, 0.99),
)


def forward(x, y):
    """Le forward, en bf16 si activé (les poids restent en float32)."""
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
        return model(x, y)


@torch.no_grad()
def estimate_loss(loader, n_batches=N_BATCHES):
    model.eval()
    losses = []
    for k, (x, y) in enumerate(loader):
        if k == n_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss = forward(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(step, valid_loss):
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "config": {"vocab_size": vocab_size, "block_size": block_size, "n_embd": n_embd,
                   "n_head": n_head, "n_layer": n_layer, "dropout": dropout},
        "stoi": stoi,
        "step": step,
        "valid": valid_loss,
    }, f"checkpoints/{run_name}.pt")


def save_csv(estimate_history, step_history):
    """Sauvegarde les pertes. Appelée à chaque évaluation : un run interrompu garde ses courbes."""
    os.makedirs("results", exist_ok=True)
    with open(f"results/{run_name}_eval.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "tokens", "train", "valid"])
        writer.writeheader()
        writer.writerows(estimate_history)
    with open(f"results/{run_name}_steps.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "loss", "lr"])
        writer.writeheader()
        writer.writerows(step_history)


@torch.no_grad()
def story(prompt, n=400):
    """Génère la suite d'un prompt et s'arrête à <|endoftext|>."""
    model.eval()
    idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)
    out = "".join(itos[i] for i in model.generate(idx, n)[0].tolist())
    model.train()
    return out.split("<|endoftext|>")[0]


# ---------- entraînement ----------
tokens_per_step = batch_size * block_size
estimate_history, step_history = [], []
train_loss, valid_loss = estimate_loss(train_loader), estimate_loss(valid_loader)
estimate_history.append({"step": 0, "tokens": 0, "train": train_loss, "valid": valid_loss})
print(f"init       : train {train_loss:.4f} | valid {valid_loss:.4f}")

best_valid, best_step = float("inf"), 0
model.train()
step, t0 = 0, time.time()
for x, y in train_loader:                     # le sampler donne exactement max_steps batches
    x, y = x.to(device), y.to(device)

    lr_step = get_lr(step)
    for group in optimizer.param_groups:
        group["lr"] = lr_step

    optimizer.zero_grad()
    _, loss = forward(x, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    step_history.append({"step": step + 1, "loss": loss.item(), "lr": lr_step})
    step += 1

    if step % eval_every == 0 or step == max_steps:
        train_loss, valid_loss = estimate_loss(train_loader), estimate_loss(valid_loader)
        elapsed = time.time() - t0
        estimate_history.append({"step": step, "tokens": step * tokens_per_step,
                                 "train": train_loss, "valid": valid_loss})
        print(f"step {step:6d} : train {train_loss:.4f} | valid {valid_loss:.4f} | lr {lr_step:.2e} "
              f"| {elapsed:.0f} s | {step * tokens_per_step / elapsed:,.0f} tokens/s")
        if valid_loss < best_valid:
            best_valid, best_step = valid_loss, step
            save_checkpoint(step, valid_loss)
        save_csv(estimate_history, step_history)

print(f"\nMeilleure validation : {best_valid:.4f} au pas {best_step}  ->  checkpoints/{run_name}.pt")
print(f"Tokens vus : {step * tokens_per_step / 1e6:,.0f} millions "
      f"({100 * step * tokens_per_step / len(train_ds):.2f} % du corpus)")

print("\n--- Exemple : Words: cat, rocket, cake ---")
print(story("Words: cat, rocket, cake\nStory: \n"))