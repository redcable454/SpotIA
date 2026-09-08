FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 espeak-ng curl ca-certificates git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2 && \
    pip install -r requirements.txt && \
    pip install --no-deps git+https://github.com/myshell-ai/OpenVoice.git && \
    python - <<'PY'
import openvoice.api as api, inspect, pathlib
p = pathlib.Path(inspect.getfile(api))
s = p.read_text()
s = s.replace("if kwargs.get('enable_watermark', True):", "if False:")
p.write_text(s)
print('Patched OpenVoice watermark loading:', p)
PY
RUN mkdir -p /app/models /app/openvoice_ckpt && \
    curl -L --fail --retry 3 -o /app/models/es_ES-davefx-medium.onnx 'https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx?download=true' && \
    curl -L --fail --retry 3 -o /app/models/es_ES-davefx-medium.onnx.json 'https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx.json?download=true' && \
    curl -L --fail --retry 3 -o /app/openvoice_ckpt/config.json 'https://huggingface.co/myshell-ai/OpenVoiceV2/resolve/main/converter/config.json?download=true' && \
    curl -L --fail --retry 3 -o /app/openvoice_ckpt/checkpoint.pth 'https://huggingface.co/myshell-ai/OpenVoiceV2/resolve/main/converter/checkpoint.pth?download=true'
COPY . .
EXPOSE 10000
CMD ["/bin/sh","-c","gunicorn -b 0.0.0.0:10000 --workers 1 --threads 1 --timeout 600 app:app"]
