import os
import tempfile
from pathlib import Path
from threading import Lock

import gradio as gr
import spaces

MODEL_NAME = os.getenv("TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
_model = None
_lock = Lock()


def get_model(device="cpu"):
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from TTS.api import TTS
                _model = TTS(MODEL_NAME).to("cpu")
    return _model.to(device)


@spaces.GPU(duration=120)
def clone_voice(text, voice_path, language):
    if not text or not text.strip():
        raise gr.Error("Escribe el texto que quieres generar.")
    if not voice_path:
        raise gr.Error("Sube una muestra de voz autorizada.")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(device)

    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out.close()
    model.tts_to_file(
        text=text.strip(),
        speaker_wav=str(voice_path),
        language=language,
        file_path=out.name,
        split_sentences=True,
    )

    if device == "cuda":
        model.to("cpu")
        torch.cuda.empty_cache()

    return out.name


with gr.Blocks(title="SpotIA Voice Engine") as demo:
    gr.Markdown("# 🎙️ SpotIA Voice Engine\nClonación de voz autorizada con XTTS v2.")
    gr.Markdown("Usa únicamente una voz propia o una voz para la que tengas autorización explícita.")

    text = gr.Textbox(
        label="Texto a generar",
        lines=5,
        placeholder="Escribe aquí el texto del spot...",
    )
    voice = gr.Audio(
        label="Muestra de voz autorizada",
        type="filepath",
        sources=["upload", "microphone"],
    )
    language = gr.Dropdown(
        label="Idioma",
        choices=[("Español", "es"), ("English", "en"), ("Português", "pt")],
        value="es",
    )
    consent = gr.Checkbox(label="Confirmo que tengo autorización para clonar esta voz")
    generate = gr.Button("Generar voz", variant="primary")
    output = gr.Audio(label="Voz generada", type="filepath")

    def guarded_clone(text_value, voice_value, lang_value, consent_value):
        if not consent_value:
            raise gr.Error("Debes confirmar que tienes autorización para usar esta voz.")
        return clone_voice(text_value, voice_value, lang_value)

    generate.click(
        guarded_clone,
        inputs=[text, voice, language, consent],
        outputs=output,
        api_name="clone_voice",
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
