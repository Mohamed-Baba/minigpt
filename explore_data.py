# explore_data.py
# Vérifie dataset.py : tokenizer, datasets train/valid, DataLoader.

import torch
from dataset import CharTokenizer, TextDataset

BLOCK_SIZE = 8

# ---------- 1. Lecture du fichier ----------
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()
print("Nombre de caractères :", len(text))
assert len(text) == 1_115_394, "fichier tronqué ou mauvais encodage"

# ---------- 2. Tokenizer sur un mini-exemple ----------
tok = CharTokenizer("banana")
print("\n[banana] stoi :", tok.stoi)
assert tok.vocab_size == 3
assert tok.stoi == {"a": 0, "b": 1, "n": 2}        # ordre trié
assert tok.encode("nab") == [2, 0, 1]
assert tok.decode([2, 0, 1]) == "nab"

# ---------- 3. Tokenizer sur Shakespeare ----------
tok = CharTokenizer(text)
print("\nTaille du vocabulaire :", tok.vocab_size)
print("Vocabulaire :", repr("".join(tok.itos[i] for i in range(tok.vocab_size))))
assert tok.vocab_size == 65
assert tok.decode(tok.encode(text[:1000])) == text[:1000]   # aller-retour

try:
    tok.encode("é")
except KeyError as e:
    print("Caractère inconnu -> KeyError :", e)

# ---------- 4. Datasets train / valid ----------
train_ds = TextDataset(text, tok, split="train", block_size=BLOCK_SIZE)
valid_ds = TextDataset(text, tok, split="valid", block_size=BLOCK_SIZE)
print("\nlen(train_ds) =", len(train_ds))
print("len(valid_ds) =", len(valid_ds))
assert len(train_ds) == 1_003_854 - BLOCK_SIZE
assert len(valid_ds) == 111_540 - BLOCK_SIZE

# premier élément
x, y = train_ds[0]
print("\nx =", x)
print("y =", y)
print(repr(tok.decode(x.tolist())), "->", repr(tok.decode(y.tolist())))
assert tok.decode(x.tolist()) == "First Ci"
assert tok.decode(y.tolist()) == "irst Cit"
assert torch.equal(x[1:], y[:-1])                  # décalage d'un cran
assert x.dtype == torch.long and y.dtype == torch.long

# dernier élément : ne doit pas être tronqué
for ds, name in [(train_ds, "train"), (valid_ds, "valid")]:
    x, y = ds[len(ds) - 1]
    assert len(x) == len(y) == BLOCK_SIZE, (name, len(x), len(y))
    assert y[-1] == ds.data[-1]                    # y finit sur le dernier token
print("Dernier élément OK (train et valid)")

# split invalide
try:
    TextDataset(text, tok, split="test")
except ValueError as e:
    print("Split invalide -> ValueError :", e)

# ---------- 5. DataLoader ----------
torch.manual_seed(0)
loader = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True)
xb, yb = next(iter(loader))
print("\nBatch x :", tuple(xb.shape), " batch y :", tuple(yb.shape))
assert xb.shape == yb.shape == (4, BLOCK_SIZE)
for b in range(4):
    print(f"  {tok.decode(xb[b].tolist())!r} -> {tok.decode(yb[b].tolist())!r}")

print("\nTous les tests sont passés.")