#!/usr/bin/env bash
set -euo pipefail

output="$(docker info 2>&1)" || status=$?
status="${status:-0}"

if [[ "$status" -eq 0 ]]; then
  version="$(docker info --format '{{.ServerVersion}}' 2>/dev/null || echo unknown)"
  echo "Docker OK (server ${version})"
  exit 0
fi

echo "Docker is not usable for user ${USER:-unknown}." >&2
if [[ "$output" == *"permission denied"* ]]; then
  cat >&2 <<'EOF'
Fix: add your account to the docker group, then start a new login shell.

  sudo usermod -aG docker "$USER"
  # log out/in, or: newgrp docker
  docker ps

Terminal-Bench-2.0 requires Docker to pull task images (e.g. alexgshaw/fix-git:20251031).
EOF
else
  printf '%s\n' "$output" >&2
fi
exit 1
