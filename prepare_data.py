# prepare_data.py
# Convertit TinyStories-Instruct (texte) en fichiers binaires prêts pour l'entraînement.
#
#   data/tinystories_instruct/train.txt  ->  train.bin
#   data/tinystories_instruct/valid.txt  ->  val.bin
#   + meta.json (le vocabulaire : stoi)
#
# Le texte est lu PAR MORCEAUX : la mémoire utilisée reste petite, même pour 2,5 Go.
# Chaque caractère devient un entier sur 2 octets (uint16) dans le fichier .bin.
#
# Lancer depuis le dossier minigpt :  python prepare_data.py

import json
import os
import string
import time
import unicodedata
from collections import Counter

import numpy as np

DATA_DIR = "data/tinystories_instruct"
CHUNK_CHARS = 16_000_000          # caractères lus à la fois (~16 Mo)

# ---------- 1. Le vocabulaire, fixé à l'avance ----------
# Les caractères ASCII imprimables (lettres, chiffres, ponctuation, espace) + le retour à la ligne.
CHARS = sorted(set(string.printable) - set("\t\r\x0b\x0c"))
STOI = {c: i for i, c in enumerate(CHARS)}
VOCAB_SIZE = len(CHARS)

# Caractères non ASCII fréquents, remplacés par leur équivalent ASCII.
REPLACE = {
    "’": "'", "‘": "'",              # ’ ‘
    "“": '"', "”": '"',              # “ ”
    "–": "-", "—": "-",              # – —
    "…": "...",                           # …
    " ": " ",                             # espace insécable
    "\t": " ",
    "\r": "",
}
TRANSLATE = str.maketrans(REPLACE)

# Table de conversion octet ASCII -> numéro de token (pour encoder très vite avec NumPy).
LOOKUP = np.zeros(128, dtype=np.uint16)
for c, i in STOI.items():
    LOOKUP[ord(c)] = i


def normalize(text, dropped):
    """Remplace les caractères connus, supprime les autres caractères hors vocabulaire."""
    text = text.translate(TRANSLATE)
    if text.isascii() and all(c in STOI for c in set(text)):
        return text                                    # cas rapide : rien à supprimer
    kept = []
    for c in text:
        if c in STOI:
            kept.append(c)
            continue
        # lettre accentuée : on garde la lettre sans l'accent (é -> e, ç -> c)
        base = unicodedata.normalize("NFKD", c).encode("ascii", "ignore").decode()
        if base and all(b in STOI for b in base):
            kept.append(base)
        else:
            dropped[c] += 1                            # emoji, caractère étranger : supprimé
    return "".join(kept)


def encode_ascii(text):
    """Texte (déjà normalisé, donc ASCII) -> tableau numpy de numéros de tokens."""
    raw = np.frombuffer(text.encode("ascii"), dtype=np.uint8)     # un octet par caractère
    return LOOKUP[raw]                                            # octet -> numéro


def convert(txt_path, bin_path):
    """Lit txt_path par morceaux, encode, et écrit les numéros dans bin_path."""
    dropped = Counter()
    n_tokens = 0
    t0 = time.time()
    with open(txt_path, "r", encoding="utf-8", errors="replace") as fin, open(bin_path, "wb") as fout:
        while True:
            chunk = fin.read(CHUNK_CHARS)
            if not chunk:
                break
            ids = encode_ascii(normalize(chunk, dropped))
            ids.tofile(fout)                           # ajoute les numéros à la fin du fichier
            n_tokens += len(ids)
            print(f"  {txt_path} : {n_tokens / 1e6:,.0f} M caractères ({time.time() - t0:.0f} s)", end="\r")
    print()
    n_dropped = sum(dropped.values())
    print(f"  -> {bin_path} : {n_tokens:,} tokens, {n_dropped:,} caractères supprimés "
          f"({100 * n_dropped / max(n_tokens, 1):.4f} %)")
    if dropped:
        print("     les plus fréquents :", dropped.most_common(10))
    return n_tokens


if __name__ == "__main__":
    print(f"Vocabulaire : {VOCAB_SIZE} caractères")
    n_val = convert(os.path.join(DATA_DIR, "valid.txt"), os.path.join(DATA_DIR, "val.bin"))
    n_train = convert(os.path.join(DATA_DIR, "train.txt"), os.path.join(DATA_DIR, "train.bin"))

    with open(os.path.join(DATA_DIR, "meta.json"), "w") as f:
        json.dump({"vocab_size": VOCAB_SIZE, "stoi": STOI,
                   "n_train": n_train, "n_val": n_val}, f)
    print(f"Vocabulaire enregistré : {DATA_DIR}/meta.json")

    # Vérification : relire le début de val.bin et le décoder
    itos = {i: c for c, i in STOI.items()}
    head = np.memmap(os.path.join(DATA_DIR, "val.bin"), dtype=np.uint16, mode="r")[:300]
    print("\nDébut de val.bin, décodé :\n" + "".join(itos[int(i)] for i in head))