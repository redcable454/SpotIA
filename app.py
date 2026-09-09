from flask import Flask, request, jsonify, send_from_directory, send_file
from pathlib import Path
from threading import Lock
import subprocess, os, uuid, wave, json, gc

BASE = Path(__file__).resolve().parent
TMP = BASE / "tmp"
TMP.mkdir(exist_ok=True)
MODELS = BASE / "models"

app = Flask(__name__, static_folder="static", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 80 * 1024 * 1024
_generation_lock = Lock()

VOICE_PRESETS = {
    "studio_mx": {"name":"Locutor Latino Studio","model":"es_MX-ald-medium","speaker":None,"pitch":1.00,"noise":0.62},
    "radio_mx": {"name":"Locutor Radio Latino","model":"es_MX-ald-medium","speaker":None,"pitch":0.94,"noise":0.58},
    "promo_mx": {"name":"Promo Enérgica Latina","model":"es_MX-ald-medium","speaker":None,"pitch":1.06,"noise":0.72},
    "dj_mx": {"name":"DJ Latino Grave","model":"es_MX-ald-medium","speaker":None,"pitch":0.88,"noise":0.55},
    "natural_es": {"name":"Natural Español","model":"es_ES-davefx-medium","speaker":None,"pitch":1.00,"noise":0.66},
    "serio_es": {"name":"Corporativo Serio","model":"es_ES-davefx-medium","speaker":None,"pitch":0.92,"noise":0.50},
    "brillante_es": {"name":"Comercial Brillante","model":"es_ES-davefx-medium","speaker":None,"pitch":1.08,"noise":0.72},
    "cine_es": {"name":"Trailer Profundo","model":"es_ES-davefx-medium","speaker":None,"pitch":0.84,"noise":0.52},
    "shar_1": {"name":"Narrador Uno","model":"es_ES-sharvard-medium","speaker":0,"pitch":1.00,"noise":0.62},
    "shar_2": {"name":"Narrador Dos","model":"es_ES-sharvard-medium","speaker":1,"pitch":1.00,"noise":0.62},
    "shar_radio": {"name":"Radio Clásica","model":"es_ES-sharvard-medium","speaker":0,"pitch":0.91,"noise":0.55},
    "shar_fresh": {"name":"Juvenil Fresca","model":"es_ES-sharvard-medium","speaker":1,"pitch":1.09,"noise":0.72},
    "shar_promo": {"name":"Promo Impacto","model":"es_ES-sharvard-medium","speaker":1,"pitch":0.96,"noise":0.68},
}

TONE_FILTERS = {
    "Natural": "anull",
    "Contento": "equalizer=f=3500:t=q:w=1:g=2,equalizer=f=180:t=q:w=1:g=1",
    "Festivo": "equalizer=f=4500:t=q:w=1:g=3,bass=g=2",
    "Serio": "bass=g=3,treble=g=-1,acompressor=threshold=-18dB:ratio=2.5:attack=12:release=160",
    "Cuña DJ": "bass=g=5,treble=g=3,acompressor=threshold=-20dB:ratio=4:attack=5:release=100",
}

FX_FILTERS = {
    "clean": "anull",
    "impacto": "bass=g=5,treble=g=2,aecho=0.8:0.45:65|130:0.22|0.10,acompressor=threshold=-18dB:ratio=3",
    "robot": "tremolo=f=28:d=0.32,aecho=0.8:0.55:45:0.22",
    "radio": "highpass=f=110,lowpass=f=9000,bass=g=4,treble=g=2,acompressor=threshold=-20dB:ratio=4,alimiter=limit=0.92",
    "chopper": "tremolo=f=8:d=0.78,aecho=0.8:0.35:90:0.12",
}


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-2500:])


def piper_paths(model_key):
    return MODELS / f"{model_key}.onnx", MODELS / f"{model_key}.onnx.json"


def synthesize_clip(text, voice_id, tone, speed, out_wav):
    from piper import PiperVoice
    preset = VOICE_PRESETS.get(voice_id) or VOICE_PRESETS["studio_mx"]
    model_path, config_path = piper_paths(preset["model"])
    voice = PiperVoice.load(str(model_path), config_path=str(config_path))
    raw = TMP / f"{uuid.uuid4().hex}_raw.wav"
    try:
        with wave.open(str(raw), "wb") as wf:
            kwargs = {
                "length_scale": max(0.65, min(1.55, 1.0 / max(0.65, min(1.45, speed)))),
                "noise_scale": preset["noise"],
            }
            if preset["speaker"] is not None:
                kwargs["speaker_id"] = preset["speaker"]
            voice.synthesize_wav(text, wf, **kwargs)
        pitch = preset["pitch"]
        pitch_filter = "anull" if abs(pitch - 1.0) < 0.001 else f"asetrate=22050*{pitch},aresample=22050,atempo={1.0/pitch:.5f}"
        tone_filter = TONE_FILTERS.get(tone, "anull")
        filters = ",".join(x for x in (pitch_filter, tone_filter, "loudnorm=I=-16:LRA=8:TP=-1.5") if x != "anull") or "anull"
        run(["ffmpeg","-y","-i",str(raw),"-af",filters,"-ac","1","-ar","44100",str(out_wav)])
    finally:
        raw.unlink(missing_ok=True)
        del voice
        gc.collect()


def concat_wavs(paths, output_path):
    if len(paths) == 1:
        run(["ffmpeg","-y","-i",str(paths[0]),"-c:a","pcm_s16le",str(output_path)])
        return
    list_path = TMP / f"{uuid.uuid4().hex}.txt"
    try:
        list_path.write_text("\n".join(f"file '{str(p).replace(chr(39), '')}'" for p in paths), encoding="utf-8")
        run(["ffmpeg","-y","-f","concat","-safe","0","-i",str(list_path),"-c:a","pcm_s16le",str(output_path)])
    finally:
        list_path.unlink(missing_ok=True)


def render_final(source_wav, music_path, fx, fmt, output_path):
    fx_filter = FX_FILTERS.get(fx, "anull")
    if music_path:
        voice_chain = fx_filter if fx_filter != "anull" else "anull"
        run([
            "ffmpeg","-y","-i",str(source_wav),"-stream_loop","-1","-i",str(music_path),
            "-filter_complex",
            f"[0:a]{voice_chain},volume=1.0[v];[1:a]volume=0.14[m];[v][m]amix=inputs=2:duration=first:dropout_transition=2,alimiter=limit=0.95[a]",
            "-map","[a]",*( ["-c:a","libmp3lame","-b:a","192k"] if fmt == "mp3" else ["-c:a","pcm_s16le"] ),str(output_path)
        ])
    else:
        run(["ffmpeg","-y","-i",str(source_wav),"-af",fx_filter,*(["-c:a","libmp3lame","-b:a","192k"] if fmt == "mp3" else ["-c:a","pcm_s16le"]),str(output_path)])


@app.get("/")
def index():
    return send_from_directory(BASE / "static", "index.html")


@app.get("/health")
def health():
    return jsonify({"ok":True,"engine":"piper-studio","voices":len(VOICE_PRESETS)})


@app.get("/api/voices")
def voices():
    return jsonify([{"id":k,"name":v["name"]} for k,v in VOICE_PRESETS.items()])


@app.post("/api/generate")
def generate():
    text = (request.form.get("text") or "").strip()
    voice_id = request.form.get("voice") or "studio_mx"
    tone = request.form.get("tone") or "Natural"
    speed = float(request.form.get("speed") or 1.0)
    fmt = (request.form.get("format") or "mp3").lower()
    fx = request.form.get("fx") or "clean"
    clips_json = request.form.get("clips")
    fmt = "wav" if fmt == "wav" else "mp3"

    clips = []
    if clips_json:
        try:
            raw_clips = json.loads(clips_json)
            for c in raw_clips[:8]:
                t = str(c.get("text") or "").strip()
                if t:
                    clips.append({
                        "text": t[:1000],
                        "voice": c.get("voice") or voice_id,
                        "tone": c.get("tone") or tone,
                        "speed": float(c.get("speed") or speed),
                    })
        except Exception:
            return jsonify({"error":"La línea de tiempo no es válida."}), 400
    elif text:
        clips = [{"text":text[:1000],"voice":voice_id,"tone":tone,"speed":speed}]

    if not clips:
        return jsonify({"error":"Escribe el texto del spot o agrega clips a la línea de tiempo."}), 400

    clip_paths = []
    joined = TMP / f"{uuid.uuid4().hex}_joined.wav"
    music_path = None
    ext = ".wav" if fmt == "wav" else ".mp3"
    final_path = TMP / f"{uuid.uuid4().hex}_spot{ext}"

    try:
        music = request.files.get("music")
        if music and music.filename:
            music_path = TMP / f"{uuid.uuid4().hex}{Path(music.filename).suffix or '.mp3'}"
            music.save(music_path)

        with _generation_lock:
            for clip in clips:
                cp = TMP / f"{uuid.uuid4().hex}.wav"
                synthesize_clip(clip["text"], clip["voice"], clip["tone"], clip["speed"], cp)
                clip_paths.append(cp)
            concat_wavs(clip_paths, joined)
            render_final(joined, music_path, fx, fmt, final_path)

        mimetype = "audio/wav" if fmt == "wav" else "audio/mpeg"
        response = send_file(final_path, mimetype=mimetype, as_attachment=False, download_name=f"spotia.{fmt}")
        response.call_on_close(lambda: final_path.unlink(missing_ok=True))
        return response
    except Exception as e:
        final_path.unlink(missing_ok=True)
        return jsonify({"error":str(e)}), 500
    finally:
        joined.unlink(missing_ok=True)
        for p in clip_paths:
            p.unlink(missing_ok=True)
        if music_path:
            music_path.unlink(missing_ok=True)
        gc.collect()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT","10000")), threaded=True)
