from flask import Flask, request, jsonify, send_from_directory, send_file
from pathlib import Path
from threading import Lock
import subprocess, os, uuid, wave, gc

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

BASE = Path(__file__).resolve().parent
TMP = BASE / "tmp"
TMP.mkdir(exist_ok=True)
MODEL = BASE / "models" / "es_ES-davefx-medium.onnx"
CONFIG = BASE / "models" / "es_ES-davefx-medium.onnx.json"
OV_CONFIG = BASE / "openvoice_ckpt" / "config.json"
OV_CKPT = BASE / "openvoice_ckpt" / "checkpoint.pth"

app = Flask(__name__, static_folder="static", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
_generation_lock = Lock()


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-3000:])


def build_script(brand, spot_type, brief, duration):
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


def synthesize_base(text, output_path):
    from piper import PiperVoice
    voice = PiperVoice.load(str(MODEL), config_path=str(CONFIG))
    with wave.open(str(output_path), "wb") as wf:
        voice.synthesize_wav(text, wf)
    del voice
    gc.collect()


def clone_tone(src_wav, ref_wav, cloned_wav):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from openvoice.api import ToneColorConverter

    conv = ToneColorConverter(str(OV_CONFIG), device="cpu")
    conv.watermark_model = None
    conv.load_ckpt(str(OV_CKPT))

    src_se = conv.extract_se(str(src_wav))
    gc.collect()
    tgt_se = conv.extract_se(str(ref_wav))
    gc.collect()

    conv.convert(
        audio_src_path=str(src_wav),
        src_se=src_se,
        tgt_se=tgt_se,
        output_path=str(cloned_wav),
        message="SpotIA"
    )

    del src_se, tgt_se, conv
    gc.collect()


@app.get("/")
def index():
    return send_from_directory(BASE / "static", "index.html")


@app.get("/health")
def health():
    return jsonify({"ok": True, "device": "cpu", "engine": "openvoice+piper-lowmem", "model_loaded": False})


@app.post("/api/generate")
def generate():
    if request.form.get("consent") != "true":
        return jsonify({"error": "Debes confirmar que tienes autorización para usar y clonar esta voz."}), 400

    voice_file = request.files.get("voice")
    if not voice_file or not voice_file.filename:
        return jsonify({"error": "Sube una muestra de voz para clonarla."}), 400

    brand = (request.form.get("brand") or "Tu marca").strip()
    spot_type = request.form.get("type") or "Publicidad comercial"
    brief = (request.form.get("brief") or "").strip()
    duration = request.form.get("duration") or "30 segundos"
    if not brief:
        return jsonify({"error": "Describe qué quieres comunicar."}), 400

    script = build_script(brand, spot_type, brief, duration)
    src_wav = TMP / f"{uuid.uuid4().hex}_src.wav"
    ref_raw = TMP / f"{uuid.uuid4().hex}{Path(voice_file.filename).suffix or '.wav'}"
    ref_wav = TMP / f"{uuid.uuid4().hex}_ref.wav"
    cloned_wav = TMP / f"{uuid.uuid4().hex}_clone.wav"
    final_path = TMP / f"{uuid.uuid4().hex}_spot.mp3"
    music_path = None

    try:
        voice_file.save(ref_raw)
        run([
            "ffmpeg","-y","-i",str(ref_raw),
            "-ac","1","-ar","16000","-t","8",
            "-af","highpass=f=70,lowpass=f=7600,loudnorm",
            str(ref_wav)
        ])

        with _generation_lock:
            synthesize_base(script, src_wav)
            gc.collect()
            clone_tone(src_wav, ref_wav, cloned_wav)
            gc.collect()

        music = request.files.get("music")
        if music and music.filename:
            music_path = TMP / f"{uuid.uuid4().hex}{Path(music.filename).suffix or '.mp3'}"
            music.save(music_path)
            run([
                "ffmpeg","-y","-i",str(cloned_wav),
                "-stream_loop","-1","-i",str(music_path),
                "-filter_complex",
                "[1:a]volume=0.12[m];[0:a]volume=1[v];[v][m]amix=inputs=2:duration=first:dropout_transition=2[a]",
                "-map","[a]","-c:a","libmp3lame","-b:a","192k",str(final_path)
            ])
        else:
            run(["ffmpeg","-y","-i",str(cloned_wav),"-c:a","libmp3lame","-b:a","192k",str(final_path)])

        response = send_file(final_path, mimetype="audio/mpeg", as_attachment=False, download_name="spotia.mp3")
        response.call_on_close(lambda: final_path.unlink(missing_ok=True))
        return response
    except Exception as e:
        final_path.unlink(missing_ok=True)
        return jsonify({"error": str(e)}), 500
    finally:
        for p in (src_wav, ref_raw, ref_wav, cloned_wav):
            p.unlink(missing_ok=True)
        if music_path:
            music_path.unlink(missing_ok=True)
        gc.collect()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT","10000")), threaded=True)
