# RegTrust demo — container image for Cloud Run / Render (or any host that
# runs a plain Docker container and injects a $PORT env var at startup).
# Does NOT modify app.py: Gradio is configured entirely via the
# GRADIO_SERVER_NAME/GRADIO_SERVER_PORT env vars set by entrypoint.sh.

FROM python:3.11-slim

# libgomp1: required at runtime by torch's CPU kernels on slim/Debian images.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch wheel — the app only ever uses torch.device("cpu")
# (see app.py), and the default PyPI wheel pulls in ~2GB of unused CUDA
# libraries. --extra-index-url falls back to PyPI for every other package.
COPY requirements.txt .
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# App code and assets (checkpoints, JASPAR panel, embeddings, calibration —
# see DEPLOY.md for what's expected under assets/). Excludes DEPLOY.md,
# hf_token.txt, __pycache__ via .dockerignore.
COPY . .

RUN chmod +x entrypoint.sh

# Cloud Run injects PORT=8080 by default; Render injects its own PORT value.
# entrypoint.sh reads whichever is set (falls back to 7860 for local
# `docker run`) and exports it as GRADIO_SERVER_PORT before launching.
EXPOSE 7860

ENTRYPOINT ["./entrypoint.sh"]
