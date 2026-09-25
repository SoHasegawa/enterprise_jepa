FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=12072 \
    # World-model backend defaults to the vLLM method; point it at your server.
    EWM_WORLD_MODEL_METHOD=vllm/gymops_world_model \
    EWM_VLLM_SERVER_PORT=9000 \
    WM_STATE=binary_error

WORKDIR /app

# Build context is the repo root (see docker-compose.yaml).
COPY src/ejepa_wm/server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The pluggable WM package: the EWM engine (_ewm_runtime/_ewm_finetuning), the
# generator factory, and the server. stdlib-only at runtime for the vLLM backend.
COPY src/ejepa_wm/ ./src/ejepa_wm/

EXPOSE 12072

# NOTE: the world-model (vLLM) server is reached over the network and is not part of
# this image. For a local HuggingFace checkpoint backend, mount it and set
# EWM_WORLD_MODEL_PATH (and add torch/transformers to requirements.txt).
CMD ["python", "-m", "ejepa_wm.server.ewm_predict"]
