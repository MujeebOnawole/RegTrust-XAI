#!/usr/bin/env bash
# Binds Gradio to 0.0.0.0 and whatever port the host injects (Cloud Run sets
# PORT=8080, Render sets its own PORT; falls back to 7860 for local
# `docker run -p 7860:7860 <image>` testing). app.py itself is untouched —
# these two env vars are read internally by gradio's demo.launch().
export GRADIO_SERVER_NAME="0.0.0.0"
export GRADIO_SERVER_PORT="${PORT:-7860}"

exec python app.py
