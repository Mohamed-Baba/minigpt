# check_gpt.py
# Vérifie le GPT complet avant l'entraînement.
# Lancer depuis le dossier minigpt :  python check_gpt.py

import math
import torch
from gpt import GPT

torch.manual_seed(1337)
V, block_size, n_embd, n_head, n_layer = 65, 32, 64, 4, 4
model = GPT(V, block_size, n_embd, n_head, n_layer)

# ---------- 1. Formes ----------
x = torch.randint(0, V, (2, block_size))
y = torch.randint(0, V, (2, block_size))
logits, loss = model(x, y)
assert logits.shape == (2, block_size, V)
print("Logits :", tuple(logits.shape))

logits_only, no_loss = model(x)
assert no_loss is None
print("Sans cibles : loss = None, OK")

# ---------- 2. Paramètres ----------
n_params = sum(p.numel() for p in model.parameters())
n_blocks = sum(p.numel() for p in model.blocks.parameters())
print(f"Paramètres : {n_params:,}  dont blocs : {n_blocks:,}  "
      f"(≈ 12·C²·L = {12 * n_embd**2 * n_layer:,})")

# ---------- 3. Perte initiale proche de ln(65) ----------
print(f"Perte initiale : {loss.item():.4f}   ln(65) = {math.log(V):.4f}")
assert abs(loss.item() - math.log(V)) < 0.5, "perte initiale anormale"

# ---------- 4. Causalité du modèle complet ----------
t = 10
x2 = x.clone()
x2[:, t + 1:] = torch.randint(0, V, (2, block_size - t - 1))
logits2, _ = model(x2)
assert torch.allclose(logits[:, : t + 1], logits2[:, : t + 1], atol=1e-5), "FUITE DU FUTUR"
assert not torch.allclose(logits[:, t + 1:], logits2[:, t + 1:]), "le modèle ignore son entrée ?"
print("Causalité : OK (le futur ne fuit pas vers le passé)")

# ---------- 5. Génération plus longue que block_size ----------
out = model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)
assert out.shape == (1, 101)
assert out.min() >= 0 and out.max() < V
print("Génération au-delà de block_size : OK")

# ---------- 6. Le gradient atteint toutes les couches ----------
model.zero_grad()
loss.backward()
sans_gradient = [name for name, p in model.named_parameters() if p.grad is None]
assert not sans_gradient, f"paramètres sans gradient : {sans_gradient}"
print("Gradient présent dans toutes les couches : OK")

# ---------- 7. Le GPU (si disponible) ----------
if torch.cuda.is_available():
    model_gpu = GPT(V, block_size, n_embd, n_head, n_layer).to("cuda")
    lg, ls = model_gpu(x.to("cuda"), y.to("cuda"))
    out = model_gpu.generate(torch.zeros((1, 1), dtype=torch.long, device="cuda"), 20)
    print(f"GPU : OK sur {torch.cuda.get_device_name(0)} (perte {ls.item():.4f})")
else:
    print("GPU : non disponible, tests faits sur CPU")

print("\nTous les tests sont passés.")