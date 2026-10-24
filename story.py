# story.py
# Le conteur : tu donnes des mots, un thème ou des caractéristiques, le modèle écrit une histoire.
# Les fonctions ci-dessous serviront aussi à l'interface graphique (app.py).
# Le modèle n'est pas modifié : on utilise seulement son forward et son generate.
# Lancer depuis le dossier minigpt (bouton ▶ de VS Code).

from collections import Counter

import torch
from gpt import GPT
from prepare_data import normalize          # le même nettoyage que pour les données d'entraînement

CHECKPOINT = "checkpoints/ts_ref.pt"
END = "<|endoftext|>"

# Les caractéristiques que le modèle a vues pendant l'entraînement
FEATURES = ["Dialogue", "BadEnding", "MoralValue", "Twist", "Foreshadowing", "Conflict"]


def load_model(path=CHECKPOINT, device=None):
    """Recharge le modèle, son vocabulaire, et le met en mode évaluation."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device)
    model = GPT(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    stoi = ckpt["stoi"]
    itos = {i: c for c, i in stoi.items()}
    return model, stoi, itos, device


def build_prompt(words="", features=(), summary="", sentence=""):
    """Construit l'en-tête au format de TinyStories-Instruct, puis « Story: »."""
    lines = []
    if features:
        lines.append("Features: " + ", ".join(features))
    if words.strip():
        lines.append("Words: " + ", ".join(w.strip() for w in words.split(",") if w.strip()))
    if summary.strip():
        lines.append("Summary: " + summary.strip())
    if sentence.strip():
        lines.append("Random sentence: " + sentence.strip())
    lines.append("Story: ")
    prompt = "\n".join(lines) + "\n"
    return normalize(prompt, Counter())        # accents, guillemets... -> caractères du vocabulaire


@torch.no_grad()
def write_story(model, stoi, itos, device, prompt, max_chars=1500, chunk=64):
    """Génère l'histoire avec model.generate, par morceaux de 64 caractères,
    et s'arrête dès que le modèle écrit <|endoftext|>."""
    idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)
    n_prompt = idx.shape[1]
    text = ""
    while idx.shape[1] - n_prompt < max_chars:
        idx = model.generate(idx, chunk)                       # ton generate, inchangé
        text = "".join(itos[i] for i in idx[0, n_prompt:].tolist())
        if END in text:
            break
    story = text.split(END)[0]
    # le modèle écrit parfois un résumé APRÈS l'histoire (comme dans les données) : on le coupe
    for marker in ("\nSummary:", "\nFeatures:", "\nWords:", "\nRandom sentence:"):
        story = story.split(marker)[0]
    return story.strip()


if __name__ == "__main__":
    model, stoi, itos, device = load_model()
    print(f"Conteur prêt ({CHECKPOINT}). Entrée vide pour passer un champ.")
    print("Caractéristiques possibles :", ", ".join(FEATURES))
    while True:
        words = input("\nMots (séparés par des virgules) : ")
        features = [f.strip() for f in input("Caractéristiques (séparées par des virgules) : ").split(",") if f.strip()]
        summary = input("Thème / résumé en une phrase : ")
        if not (words or features or summary):
            break
        prompt = build_prompt(words, features, summary)
        print("\n--- en-tête donné au modèle ---\n" + prompt + "-------------------------------")
        print(write_story(model, stoi, itos, device, prompt))