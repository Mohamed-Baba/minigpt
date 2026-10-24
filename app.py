# app.py
# La page web du conteur, pour la famille.
# Lancer depuis le dossier minigpt (bouton ▶ de VS Code), puis ouvrir l'adresse affichée dans le terminal.
#   SHARE = False : la page n'est visible que sur ton PC  (http://127.0.0.1:7860)
#   SHARE = True  : Gradio crée aussi un lien public temporaire (environ 72 h) à envoyer à ta famille.
#                   Ton PC doit rester allumé, et ce script lancé, pendant qu'ils l'utilisent.

import gradio as gr
from story import FEATURES, build_prompt, load_model, write_story

SHARE = True

# Le modèle est chargé UNE seule fois, au lancement de la page.
model, stoi, itos, device = load_model()

FEATURE_LABELS = {          # ce que voit la famille -> ce que le modèle connaît
    "Des dialogues": "Dialogue",
    "Une fin triste": "BadEnding",
    "Une morale": "MoralValue",
    "Un rebondissement": "Twist",
    "Un présage": "Foreshadowing",
    "Un conflit": "Conflict",
}
assert set(FEATURE_LABELS.values()) <= set(FEATURES)


def tell_story(words, features, summary):
    """Appelée à chaque clic sur « Raconte » : construit l'en-tête, génère, renvoie l'histoire."""
    if not (words.strip() or features or summary.strip()):
        return "Donne au moins quelques mots, une idée ou une caractéristique !"
    prompt = build_prompt(words, [FEATURE_LABELS[f] for f in features], summary)
    return write_story(model, stoi, itos, device, prompt)


with gr.Blocks(title="Le conteur de Mohamed") as demo:
    gr.Markdown(
        "# Le conteur de Mohamed\n"
        "Un petit modèle de langage que j'ai écrit et entraîné moi-même, de zéro. "
        "Il invente de courtes histoires **en anglais**, pour enfants. "
        "Donne-lui quelques mots, une idée, ou les deux."
    )
    with gr.Row():
        with gr.Column():
            words = gr.Textbox(label="Des mots à placer dans l'histoire (en anglais, séparés par des virgules)",
                               placeholder="dog, ball, rain")
            summary = gr.Textbox(label="L'idée de l'histoire, en une phrase (facultatif, en anglais)",
                                 placeholder="A little dog is afraid of the rain.")
            features = gr.CheckboxGroup(list(FEATURE_LABELS), label="Ce que l'histoire doit contenir (facultatif)")
            button = gr.Button("Raconte !", variant="primary")
        with gr.Column():
            output = gr.Textbox(label="L'histoire", lines=18)

    button.click(tell_story, inputs=[words, features, summary], outputs=output)
    gr.Examples(
        examples=[["dog, ball, rain", ["Des dialogues"], ""],
                  ["cat, rocket, cake", [], "A cat wants to fly to the moon."],
                  ["tree, bird, sad", ["Une morale"], ""]],
        inputs=[words, features, summary],
    )

demo.launch(share=SHARE)