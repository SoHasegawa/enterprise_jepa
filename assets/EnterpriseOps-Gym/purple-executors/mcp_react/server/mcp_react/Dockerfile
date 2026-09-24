FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=12072 \
    # The gym repo is mounted/copied here (benchmark/ + seed DBs).
    ENTERPRISEOPS_GYM_REPO_PATH=/gym/EnterpriseOps-Gym \
    # Default orchestrator; set ENTERPRISEOPS_ORCHESTRATOR=wm_react + WM_* for the World Model.
    ENTERPRISEOPS_ORCHESTRATOR=react

WORKDIR /app

COPY server/mcp_react/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The executor package: executor.py + the orchestrator modules + the server/.
COPY executor.py wm_react.py wm_ewm.py k_controller.py version.toml ./
COPY server/ ./server/

EXPOSE 12072

# NOTE: mount the gym repo at $ENTERPRISEOPS_GYM_REPO_PATH (it carries the seed
# DBs and the gym BenchmarkExecutor); the tool MCP servers and (for imagined
# mode) the world-model vLLM server are reached over the network and are not
# part of this image. Provide ENTERPRISEOPS_LLM_* for the agent backend.
CMD ["python", "server/mcp_react/mcp_server.py"]
