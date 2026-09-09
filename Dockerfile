FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 espeak-ng curl ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
RUN mkdir -p /app/models && \
    curl -L --fail --retry 3 -o /app/models/es_ES-davefx-medium.onnx 'https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx?download=true' && \
    curl -L --fail --retry 3 -o /app/models/es_ES-davefx-medium.onnx.json 'https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx.json?download=true' && \
    curl -L --fail --retry 3 -o /app/models/es_MX-ald-medium.onnx 'https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/ald/medium/es_MX-ald-medium.onnx?download=true' && \
    curl -L --fail --retry 3 -o /app/models/es_MX-ald-medium.onnx.json 'https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/ald/medium/es_MX-ald-medium.onnx.json?download=true' && \
    curl -L --fail --retry 3 -o /app/models/es_ES-sharvard-medium.onnx 'https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/sharvard/medium/es_ES-sharvard-medium.onnx?download=true' && \
    curl -L --fail --retry 3 -o /app/models/es_ES-sharvard-medium.onnx.json 'https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/sharvard/medium/es_ES-sharvard-medium.onnx.json?download=true'
COPY . .
EXPOSE 10000
CMD ["/bin/sh","-c","gunicorn -b 0.0.0.0:10000 --workers 1 --threads 2 --timeout 300 app:app"]
