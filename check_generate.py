# check_generate.py
# Vérifie BigramLM.generate avant l'entraînement.


import torch
from dataset import CharTokenizer
from model import BigramLM

with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()
tok = CharTokenizer(text)

torch.manual_seed(1337)
model = BigramLM(tok.vocab_size)

# ---------- 1. Formes ----------
context = torch.zeros((1, 1), dtype=torch.long)          # le caractère '\n' (id 0)
out = model.generate(context, max_new_tokens=100)
print("Forme de sortie :", tuple(out.shape))
assert out.shape == (1, 101), "il faut t + max_new_tokens tokens"
assert out.dtype == torch.long
assert out[0, 0] == 0, "le contexte de départ doit être conservé au début"
assert out.min() >= 0 and out.max() < tok.vocab_size, "ids hors vocabulaire"

# ---------- 2. Plusieurs séquences en parallèle ----------
out4 = model.generate(torch.zeros((4, 1), dtype=torch.long), max_new_tokens=20)
assert out4.shape == (4, 21)
print("4 séquences en parallèle : OK")

# ---------- 3. Un contexte plus long est prolongé, pas modifié ----------
start = torch.tensor([tok.encode("First Citizen:")], dtype=torch.long)   # [1, 14]
out_long = model.generate(start, max_new_tokens=30)
assert out_long.shape == (1, 44)
assert torch.equal(out_long[:, :14], start), "le début ne doit pas changer (bug d'addition ?)"
print("Contexte prolongé :", repr(tok.decode(out_long[0].tolist())))

# ---------- 4. Même graine -> même texte ----------
torch.manual_seed(0); a = model.generate(context, 50)
torch.manual_seed(0); b = model.generate(context, 50)
assert torch.equal(a, b)
print("Reproductible avec la même graine : OK")

# ---------- 5. Le tirage suit bien les probabilités du modèle ----------
# Après '\n', on génère 1 token 20 000 fois et on compare les fréquences
# aux probabilités softmax(W[0]) : elles doivent être proches.
torch.manual_seed(1)
many = model.generate(torch.zeros((20000, 1), dtype=torch.long), max_new_tokens=1)[:, 1]
freq = torch.bincount(many, minlength=tok.vocab_size).float() / 20000
p = torch.softmax(model.table.weight[0], dim=-1).detach()
ecart = (freq - p).abs().max().item()
print(f"Écart max entre fréquences et probabilités : {ecart:.4f}")
assert ecart < 0.02, "le tirage ne suit pas softmax(W[0])"

# ---------- 6. Pas de gradient pendant la génération ----------
assert not out.requires_grad, "@torch.no_grad() manquant ?"

# ---------- 7. Le texte d'un modèle NON entraîné ----------
print("\nTexte généré (modèle NON entraîné) :")
print(tok.decode(out[0].tolist()))
print("\nTous les tests sont passés.")
