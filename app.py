from flask import Flask, request, jsonify, send_from_directory, send_file
from pathlib import Path
from threading import Lock
import subprocess, os, uuid, torch
from TTS.api import TTS

BASE = Path(__file__).resolve().parent
TMP = BASE / "tmp"
TMP.mkdir(exist_ok=True)

app = Flask(__name__, static_folder="static", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
MODEL_NAME = os.getenv("TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_model = None
_model_lock = Lock()
_generation_lock = Lock()

def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = TTS(MODEL_NAME).to(DEVICE)
    return _model

def build_script(brand, spot_type, brief, style, duration):
    intro = {
        "Publicidad comercial": "¡Atención!",
        "Promoción de programación": "Prepárate para disfrutar nuestra programación.",
        "Identificación de emisora": "Estás escuchando",
        "Inicio de espacio publicitario": "Iniciamos nuestro espacio publicitario.",
        "Fin de espacio publicitario": "Finaliza nuestro espacio publicitario.",
        "Promo de película o serie": "Muy pronto, una historia que no te puedes perder."
    }.get(spot_type, "¡Atención!")
    closing = {
        "Inicio de espacio publicitario": f"{brand}. Volvemos en unos instantes.",
        "Fin de espacio publicitario": f"{brand}. Continuamos con nuestra programación.",
        "Identificación de emisora": f"{brand}. Siempre contigo."
    }.get(spot_type, f"{brand}. Siempre contigo.")
    words_target = {"15 segundos":35,"30 segundos":70,"45 segundos":105,"60 segundos":140}.get(duration,70)
    core = f"{intro} {brand}. {brief.strip()} {closing}"
    words = core.split()
    return " ".join(words[:words_target]) if len(words) > words_target else core

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-3000:])

@app.get("/")
def index():
    return send_from_directory(BASE / "static", "index.html")

@app.get("/health")
def health():
    return jsonify({"ok": True, "device": DEVICE, "model_loaded": _model is not None})

@app.post("/api/generate")
def generate():
    if request.form.get("consent") != "true":
        return jsonify({"error": "Debes confirmar que tienes autorización para usar esta voz."}), 400
    if "voice" not in request.files:
        return jsonify({"error": "Falta una muestra de voz."}), 400
    brand = (request.form.get("brand") or "Tu marca").strip()
    spot_type = request.form.get("type") or "Publicidad comercial"
    brief = (request.form.get("brief") or "").strip()
    style = request.form.get("style") or "Comercial dinámico"
    duration = request.form.get("duration") or "30 segundos"
    language = request.form.get("language") or "es"
    if not brief:
        return jsonify({"error": "Describe qué quieres comunicar."}), 400
    script = build_script(brand, spot_type, brief, style, duration)
    voice_file = request.files["voice"]
    voice_suffix = Path(voice_file.filename or "voice.wav").suffix.lower() or ".wav"
    voice_path = TMP / f"{uuid.uuid4().hex}{voice_suffix}"
    narration_path = TMP / f"{uuid.uuid4().hex}_voice.wav"
    final_path = TMP / f"{uuid.uuid4().hex}_spot.mp3"
    music_path = None
    try:
        voice_file.save(voice_path)
        with _generation_lock:
            get_model().tts_to_file(text=script, speaker_wav=str(voice_path), language=language, file_path=str(narration_path), split_sentences=True)
        music = request.files.get("music")
        if music and music.filename:
            music_suffix = Path(music.filename).suffix.lower() or ".mp3"
            music_path = TMP / f"{uuid.uuid4().hex}{music_suffix}"
            music.save(music_path)
            run(["ffmpeg","-y","-i",str(narration_path),"-stream_loop","-1","-i",str(music_path),"-filter_complex","[1:a]volume=0.12[m];[0:a]volume=1.0[v];[v][m]amix=inputs=2:duration=first:dropout_transition=2[a]","-map","[a]","-c:a","libmp3lame","-b:a","192k",str(final_path)])
        else:
            run(["ffmpeg","-y","-i",str(narration_path),"-c:a","libmp3lame","-b:a","192k",str(final_path)])
        response = send_file(final_path, mimetype="audio/mpeg", as_attachment=False, download_name="spotia.mp3")
        response.call_on_close(lambda: final_path.unlink(missing_ok=True))
        return response
    except Exception as e:
        final_path.unlink(missing_ok=True)
        return jsonify({"error": str(e)}), 500
    finally:
        voice_path.unlink(missing_ok=True)
        narration_path.unlink(missing_ok=True)
        if music_path:
            music_path.unlink(missing_ok=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT","5000")), threaded=True)
