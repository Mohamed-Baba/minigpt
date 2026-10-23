# check_model.py
# Vérifie model.py avant tout entraînement.

import math
import torch
import torch.nn.functional as F
from dataset import CharTokenizer, TextDataset
from model import BigramLM

torch.manual_seed(1337)

# ---------- 1. Rappel : .view et cross_entropy sur un mini-exemple ----------
logits_toy = torch.arange(12).view(2, 3, 2)
print("logits_toy [2, 3, 2] ->\n", logits_toy.view(6, 2))
assert logits_toy.view(6, 2)[4].tolist() == [8, 9]     # ligne 4 = (fenêtre 1, position 1)

z = torch.tensor([[2., 0., -1.], [0., 0., 0.]])
y = torch.tensor([0, 2])
ce = F.cross_entropy(z, y).item()
print(f"\ncross_entropy mini-exemple : {ce:.4f}  (calcul à la main : 0.634)")
assert abs(ce - 0.6344) < 1e-3

# ---------- 2. Données ----------
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()
tok = CharTokenizer(text)
train_ds = TextDataset(text, tok, split="train", block_size=8)
loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
x, y = next(iter(loader))

# ---------- 3. Le modèle ----------
model = BigramLM(tok.vocab_size)
n_params = sum(p.numel() for p in model.parameters())
print("\nNombre de paramètres :", n_params)
assert n_params == 65 * 65, "la table doit être exactement W de taille 65 x 65"

# ---------- 4. Forward sans cibles (mode génération) ----------
logits, loss = model(x)
print("Sans cibles -> logits", tuple(logits.shape), "| loss =", loss)
assert logits.shape == (32, 8, 65)
assert loss is None

# ---------- 5. Forward avec cibles (mode entraînement) ----------
logits, loss = model(x, y)
print("Avec cibles -> logits", tuple(logits.shape), f"| loss = {loss.item():.4f}")
print(f"ln(65) = {math.log(65):.4f}")
assert loss.dim() == 0, "la perte doit être un seul nombre (scalaire)"
assert 4.17 < loss.item() < 5.5, "perte initiale anormale : regarde forward"

# ---------- 6. La perte du modèle = calcul à la main ----------
# On recalcule la perte nous-mêmes avec la formule du PDF et on compare.
z_flat = logits.view(-1, 65)                    # [256, 65]
y_flat = y.view(-1)                             # [256]
log_probs = z_flat - torch.logsumexp(z_flat, dim=1, keepdim=True)   # log softmax
manual = -log_probs[torch.arange(len(y_flat)), y_flat].mean()
print(f"Perte recalculée à la main : {manual.item():.4f}")
assert torch.allclose(loss, manual), "forward ne calcule pas la bonne perte"

# ---------- 7. Le gradient ne touche que les lignes vues ----------
model.zero_grad()
loss.backward()
grad = model.table.weight.grad                  # [65, 65]
rows_seen = set(x.view(-1).tolist())
rows_with_grad = set(torch.nonzero(grad.abs().sum(dim=1)).view(-1).tolist())
print(f"Caractères présents dans le batch : {len(rows_seen)} | lignes de W avec un gradient : {len(rows_with_grad)}")
assert rows_seen == rows_with_grad, "seules les lignes des caractères du batch doivent bouger"

print("\nTous les tests sont passés.")
