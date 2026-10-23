# sample.py
# Charge un modèle entraîné et écrit la suite d'un texte que tu donnes (un « prompt »).
# Lancer depuis le dossier minigpt (bouton ▶ de VS Code).

import torch
from gpt import GPT

CHECKPOINT     = "checkpoints/gpt_ref.pt"   # le meilleur modèle sauvé par train_gpt.py
MAX_NEW_TOKENS = 500                        # nombre de caractères à écrire
SEED           = None                       # un entier (ex. 0) pour un texte reproductible

device = "cuda" if torch.cuda.is_available() else "cpu"
if SEED is not None:
    torch.manual_seed(SEED)

# ---------- recharger le modèle ----------
ckpt = torch.load(CHECKPOINT, map_location=device)   # relit le fichier, tenseurs sur le bon appareil
model = GPT(**ckpt["config"]).to(device)             # même architecture que pendant l'entraînement
model.load_state_dict(ckpt["model"])                 # on y remet les poids appris
model.eval()                                         # dropout éteint
print(f"Modèle chargé : {CHECKPOINT} (pas {ckpt.get('step', '?')}, "
      f"validation {ckpt.get('valid', float('nan')):.4f})")

# ---------- recharger le tokenizer ----------
stoi = ckpt["stoi"]                                  # caractère -> numéro, celui de l'entraînement
itos = {i: c for c, i in stoi.items()}               # numéro -> caractère


def encode(s):
    return [stoi[c] for c in s]


def decode(ids):
    return "".join(itos[i] for i in ids)


# ---------- la boucle : un prompt, une suite ----------
print("Écris le début d'un texte, puis Entrée. Tape \\n pour un retour à la ligne. Entrée vide pour quitter.")
while True:
    prompt = input("\nPrompt : ")
    if prompt == "":
        break
    prompt = prompt.replace("\\n", "\n")             # permet d'écrire ROMEO:\n dans le terminal

    inconnus = set(prompt) - set(stoi)
    if inconnus:
        print("Caractères inconnus du modèle :", inconnus)
        continue

    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)   # [1, t]
    out = model.generate(idx, max_new_tokens=MAX_NEW_TOKENS)               # [1, t + MAX_NEW_TOKENS]
    print("-" * 60)
    print(decode(out[0].tolist()))
    print("-" * 60)