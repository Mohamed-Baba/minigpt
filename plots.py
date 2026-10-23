# plots.py
# Dessine les courbes de tous les runs enregistrés dans results/.
#
# Convention de nommage (deux fichiers par run, dans results/) :
#   results/<nom_du_run>_eval.csv    colonnes : step, train, valid   (pertes estimées)
#   results/<nom_du_run>_steps.csv   colonnes : step, loss           (perte réelle de chaque pas)
# Tout nouveau modèle qui écrit ces deux fichiers est tracé automatiquement.
#
# Lancer depuis le dossier minigpt :  python plots.py
# Produit :
#   figures/<nom_du_run>.png         une figure par run (2 panneaux)
#   figures/comparaison_valid.png    la validation de tous les runs superposée

import csv
import math
import os
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
FIGURES_DIR = "figures"
EVAL_SUFFIX = "_eval.csv"      # pertes estimées (train et validation)
STEPS_SUFFIX = "_steps.csv"    # perte réelle de chaque pas

# Lignes de référence dessinées sur toutes les figures
REFERENCES = {
    "prédiction uniforme, ln 65": math.log(65),
    "optimum du bigramme": 2.452,
}


# ---------- lecture ----------

def read_csv(path):
    """Lit un CSV et renvoie un dict colonne -> liste de nombres."""
    columns = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for key, value in row.items():
                columns.setdefault(key, []).append(float(value))
    return columns


def eval_path(run):
    return os.path.join(RESULTS_DIR, run + EVAL_SUFFIX)


def steps_path(run):
    return os.path.join(RESULTS_DIR, run + STEPS_SUFFIX)


def list_runs():
    """Renvoie la liste triée des noms de runs trouvés dans results/.
    'bigram_eval.csv' et 'bigram_steps.csv' donnent tous les deux le run 'bigram'."""
    if not os.path.isdir(RESULTS_DIR):
        return []
    runs = set()
    for filename in os.listdir(RESULTS_DIR):
        for suffix in (EVAL_SUFFIX, STEPS_SUFFIX):
            if filename.endswith(suffix):
                runs.add(filename[: -len(suffix)])
    return sorted(runs)


# ---------- outils ----------

def smooth(values, beta=0.98):
    """Moyenne mobile exponentielle, avec la même correction de biais qu'Adam."""
    out, s = [], 0.0
    for t, v in enumerate(values, start=1):
        s = beta * s + (1 - beta) * v
        out.append(s / (1 - beta ** t))
    return out


def draw_references(ax):
    """Lignes horizontales grises, nom écrit sous la ligne (les courbes passent au-dessus)."""
    x_right = ax.get_xlim()[1]
    for name, value in REFERENCES.items():
        ax.axhline(value, color="0.6", linestyle=":", linewidth=1)
        ax.text(x_right, value, f"{name} ({value:.3f}) ",
                ha="right", va="top", fontsize=8, color="0.35")


def style(ax, title):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("pas d'optimisation")
    ax.set_ylabel("perte (nats par token)")
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, filename):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Figure enregistrée :", path)


# ---------- figure d'un run ----------

def plot_run(run):
    """Deux panneaux : à gauche la perte réelle de chaque pas, à droite train vs valid estimées."""
    fig, (ax_steps, ax_eval) = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)

    # Panneau gauche : la perte réelle, pas par pas
    if os.path.exists(steps_path(run)):
        s = read_csv(steps_path(run))
        ax_steps.plot(s["step"], s["loss"], color="C0", linewidth=0.6, alpha=0.3,
                      label="perte du batch (brute)")
        ax_steps.plot(s["step"], smooth(s["loss"]), color="C0", linewidth=2,
                      label="moyenne mobile exponentielle")
        ax_steps.legend(frameon=False, fontsize=9)
    else:
        ax_steps.text(0.5, 0.5, f"pas de {run}{STEPS_SUFFIX}", ha="center", transform=ax_steps.transAxes)
    style(ax_steps, f"{run} : perte réelle à chaque pas")

    # Panneau droit : pertes estimées train et validation
    if os.path.exists(eval_path(run)):
        e = read_csv(eval_path(run))
        ax_eval.plot(e["step"], e["valid"], color="C1", linewidth=2, marker="o", markersize=4,
                     label="validation (estimée)")
        ax_eval.plot(e["step"], e["train"], color="C1", linewidth=1.2, linestyle="--",
                     label="train (estimée)")
        ax_eval.legend(frameon=False, fontsize=9)
    else:
        ax_eval.text(0.5, 0.5, f"pas de {run}{EVAL_SUFFIX}", ha="center", transform=ax_eval.transAxes)
    style(ax_eval, f"{run} : train vs validation")
    ax_eval.set_ylabel("")

    for ax in (ax_steps, ax_eval):
        draw_references(ax)

    save(fig, f"{run}.png")


# ---------- comparaison de tous les runs ----------

def plot_comparison(runs):
    """La perte de validation de tous les runs sur le même graphique."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    n = 0
    for i, run in enumerate(runs):
        if not os.path.exists(eval_path(run)):
            continue
        e = read_csv(eval_path(run))
        ax.plot(e["step"], e["valid"], color=f"C{i}", linewidth=2, marker="o", markersize=4,
                label=f"{run} (dernière : {e['valid'][-1]:.3f})")
        n += 1
    if n == 0:
        plt.close(fig)
        return
    style(ax, "Perte de validation : comparaison des runs")
    draw_references(ax)
    ax.legend(frameon=False, fontsize=9)
    save(fig, "comparaison_valid.png")


if __name__ == "__main__":
    runs = list_runs()
    if not runs:
        print(f"Aucun run trouvé dans {RESULTS_DIR}/. Lance d'abord un entraînement.")
    for run in runs:
        plot_run(run)
    plot_comparison(runs)