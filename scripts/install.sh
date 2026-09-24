#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_PYTHON_BIN="${PYTHON_BIN:-}"

usage() {
  cat <<'EOF'
Usage:
  scripts/install.sh [component...]

Components:
  common                 Install the root environment (.venv) with the `ejepa` CLI.
  enterpriseops-gym      Install EnterpriseOps-Gym environments (also installs common).
  crmarenapro            Install CRMArena-Pro environments (also installs common).
  workbench              Install WorkBench environments (also installs common).
  automationbench        Install AutomationBench environments (also installs common).
  terminal-bench-2.0     Install Terminal-Bench-2.0 environments (also installs common).
  all                    Install all components. This is the default.

Options:
  --python <bin>         Python executable (>= 3.11) used for every environment.
  -h, --help             Show this help.

Environment variables:
  PYTHON_BIN         Default value for --python.

Examples:
  scripts/install.sh
  scripts/install.sh enterpriseops-gym
  scripts/install.sh workbench automationbench
EOF
}

python_satisfies() {
  local python_bin="$1"
  local required_major="$2"
  local required_minor="$3"

  "$python_bin" -c "import sys; raise SystemExit(0 if sys.version_info >= (${required_major}, ${required_minor}) else 1)"
}

choose_python() {
  local required_major="$1"
  local required_minor="$2"
  shift 2

  local candidate=""
  for candidate in "$@"; do
    if [[ -z "$candidate" ]]; then
      continue
    fi
    if ! command -v "$candidate" >/dev/null 2>&1; then
      continue
    fi
    if python_satisfies "$candidate" "$required_major" "$required_minor"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

resolve_default_python_bin() {
  if [[ -n "$DEFAULT_PYTHON_BIN" ]]; then
    printf '%s\n' "$DEFAULT_PYTHON_BIN"
    return 0
  fi
  choose_python 3 11 python3.13 python3.12 python3.11 python3 python
}

venv_python_path() {
  local venv_dir="$1"
  if [[ -x "${venv_dir}/bin/python" ]]; then
    printf '%s\n' "${venv_dir}/bin/python"
    return 0
  fi
  if [[ -x "${venv_dir}/Scripts/python.exe" ]]; then
    printf '%s\n' "${venv_dir}/Scripts/python.exe"
    return 0
  fi
  return 1
}

venv_uv_path() {
  local venv_dir="$1"
  if [[ -x "${venv_dir}/bin/uv" ]]; then
    printf '%s\n' "${venv_dir}/bin/uv"
    return 0
  fi
  if [[ -x "${venv_dir}/Scripts/uv.exe" ]]; then
    printf '%s\n' "${venv_dir}/Scripts/uv.exe"
    return 0
  fi
  return 1
}

install_ejepa_shim() {
  local venv_dir="$1"
  local shim_dir="${venv_dir}/bin"
  local shim_path="${shim_dir}/ejepa"

  mkdir -p "$shim_dir"
  cat > "$shim_path" <<EOF
#!/usr/bin/env bash
exec "${REPO_ROOT}/ejepa" "\$@"
EOF
  chmod +x "$shim_path"
}

install_crmarenapro_databases() {
  local data_dir="${REPO_ROOT}/assets/crmarenapro/purple-executors/baseline_crm_agent/data"
  local b2b_db="${data_dir}/crmarenapro_b2b_data.db"
  local b2c_db="${data_dir}/crmarenapro_b2c_data.db"
  local image="ghcr.io/rkstu/baseline-crm-agent:latest"
  local container_id=""

  if [[ -f "$b2b_db" ]]; then
    echo "[install] crmarenapro database already present: ${b2b_db}"
    return 0
  fi

  if ! command -v docker >/dev/null 2>&1; then
    echo "[install] WARNING: docker not found; skipping CRMArenaPro database extraction." >&2
    echo "[install] WARNING: baseline_crm_agent needs crmarenapro_b2b_data.db for real scoring." >&2
    return 0
  fi

  mkdir -p "$data_dir"
  echo "[install] Pulling CRMArenaPro baseline image for SQLite databases: ${image}"
  if ! docker pull "$image"; then
    echo "[install] WARNING: could not pull ${image}; skipping CRMArenaPro database extraction." >&2
    return 0
  fi

  if ! container_id="$(docker create "$image")"; then
    echo "[install] WARNING: could not create temporary container from ${image}." >&2
    return 0
  fi

  echo "[install] Extracting CRMArenaPro SQLite databases into ${data_dir}"
  docker cp "${container_id}:/home/agent/data/crmarenapro_b2b_data.db" "$b2b_db" || true
  docker cp "${container_id}:/home/agent/data/crmarenapro_b2c_data.db" "$b2c_db" || true
  docker rm "$container_id" >/dev/null

  if [[ ! -f "$b2b_db" ]]; then
    echo "[install] WARNING: ${image} did not provide crmarenapro_b2b_data.db." >&2
    return 0
  fi
  echo "[install] Installed CRMArenaPro database: ${b2b_db}"
}

ensure_python_bin() {
  local python_bin="$1"
  local label="$2"
  local required_major="$3"
  local required_minor="$4"

  if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "[install] ${label}: python executable not found: ${python_bin}" >&2
    exit 1
  fi
  if ! python_satisfies "$python_bin" "$required_major" "$required_minor"; then
    echo "[install] ${label}: ${python_bin} does not satisfy Python >= ${required_major}.${required_minor}" >&2
    exit 1
  fi
}

sync_project() {
  local label="$1"
  local project_dir="$2"
  local python_bin="$3"
  local required_major="$4"
  local required_minor="$5"

  local venv_dir="${project_dir}/.venv"
  local venv_python=""
  local venv_uv=""

  ensure_python_bin "$python_bin" "$label" "$required_major" "$required_minor"

  if [[ ! -f "${project_dir}/pyproject.toml" ]]; then
    echo "[install] ${label}: pyproject.toml not found: ${project_dir}" >&2
    exit 1
  fi

  if ! venv_python="$(venv_python_path "$venv_dir")"; then
    echo "[install] Creating ${label} virtualenv with ${python_bin}: ${venv_dir}"
    "$python_bin" -m venv "$venv_dir"
    venv_python="$(venv_python_path "$venv_dir")"
  fi

  if ! "$venv_python" -m pip --version >/dev/null 2>&1; then
    echo "[install] Bootstrapping pip in ${venv_dir}"
    "$venv_python" -m ensurepip --upgrade
  fi

  echo "[install] Bootstrapping uv in ${venv_dir}"
  "$venv_python" -m pip install --upgrade --no-deps --requirement "${REPO_ROOT}/requirements/bootstrap.txt"
  venv_uv="$(venv_uv_path "$venv_dir")"

  echo "[install] Syncing ${label}: ${project_dir}"
  (
    cd "$project_dir"
    # When the caller has another venv active, uv warns about VIRTUAL_ENV mismatch.
    # We always target the project-local venv (`$venv_uv`), so clear it here.
    unset VIRTUAL_ENV
    if [[ ! -f uv.lock ]]; then
      echo "[install] ${label}: uv.lock not found; running uv lock (commit uv.lock for reproducible installs)"
      "$venv_uv" lock
    fi
    "$venv_uv" sync --frozen
  )
}

SELECTED_COMPONENTS=""

select_component() {
  case " ${SELECTED_COMPONENTS} " in
    *" $1 "*) return 0 ;;
  esac
  SELECTED_COMPONENTS="${SELECTED_COMPONENTS:+${SELECTED_COMPONENTS} }$1"
}

_component_selected() {
  case " ${SELECTED_COMPONENTS} " in
    *" $1 "*) return 0 ;;
    *) return 1 ;;
  esac
}

expand_component() {
  local component="$1"
  case "$component" in
    common)
      select_component "common"
      ;;
    crmarenapro|enterpriseops-gym|terminal-bench-2.0|workbench|automationbench)
      select_component "common"
      select_component "$component"
      ;;
    all)
      select_component "common"
      select_component "enterpriseops-gym"
      select_component "crmarenapro"
      select_component "workbench"
      select_component "automationbench"
      select_component "terminal-bench-2.0"
      ;;
    *)
      echo "Unknown component: ${component}" >&2
      usage >&2
      exit 2
      ;;
  esac
}

PYTHON_BIN=""
declare -a REQUESTED_COMPONENTS=()

while (($#)); do
  case "$1" in
    --python)
      if (($# < 2)); then
        echo "--python requires a value." >&2
        exit 2
      fi
      PYTHON_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      REQUESTED_COMPONENTS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$PYTHON_BIN" ]]; then
  if ! PYTHON_BIN="$(resolve_default_python_bin)"; then
    echo "No Python >= 3.11 interpreter found. Set --python or PYTHON_BIN." >&2
    exit 1
  fi
fi

if ((${#REQUESTED_COMPONENTS[@]} == 0)); then
  REQUESTED_COMPONENTS=("all")
fi

for component in "${REQUESTED_COMPONENTS[@]}"; do
  expand_component "$component"
done

echo "[install] Repository root: ${REPO_ROOT}"
echo "[install] Python >=3.11: ${PYTHON_BIN}"

if _component_selected "common"; then
  sync_project "common" "$REPO_ROOT" "$PYTHON_BIN" 3 11
fi

if _component_selected "crmarenapro"; then
  sync_project "crmarenapro green" "$REPO_ROOT/assets/crmarenapro/green" "$PYTHON_BIN" 3 11
  sync_project "crmarenapro purple" "$REPO_ROOT/assets/crmarenapro/purple" "$PYTHON_BIN" 3 11
  install_ejepa_shim "$REPO_ROOT/assets/crmarenapro/green/.venv"
  install_ejepa_shim "$REPO_ROOT/assets/crmarenapro/purple/.venv"
  install_crmarenapro_databases
fi

if _component_selected "enterpriseops-gym"; then
  sync_project "EnterpriseOps-Gym green" "$REPO_ROOT/assets/EnterpriseOps-Gym/green" "$PYTHON_BIN" 3 11
  sync_project "EnterpriseOps-Gym purple" "$REPO_ROOT/assets/EnterpriseOps-Gym/purple" "$PYTHON_BIN" 3 11
fi

if _component_selected "terminal-bench-2.0"; then
  sync_project "Terminal-Bench-2.0 green" "$REPO_ROOT/assets/Terminal-Bench-2.0/green" "$PYTHON_BIN" 3 11
  sync_project "Terminal-Bench-2.0 purple" "$REPO_ROOT/assets/Terminal-Bench-2.0/purple" "$PYTHON_BIN" 3 11
  echo "[install] Terminal-Bench-2.0 purple extras (llm_shell, terminus_2)..."
  (cd "$REPO_ROOT/assets/Terminal-Bench-2.0/purple" && uv sync --extra llm_shell --extra terminus_2)
fi

if _component_selected "workbench"; then
  sync_project "WorkBench green" "$REPO_ROOT/assets/WorkBench/green" "$PYTHON_BIN" 3 11
  sync_project "WorkBench purple" "$REPO_ROOT/assets/WorkBench/purple" "$PYTHON_BIN" 3 11
  echo "[install] WorkBench purple extras (mcp_react)..."
  (cd "$REPO_ROOT/assets/WorkBench/purple" && uv sync --extra mcp_react)
  echo "[install] WorkBench needs an external WorkBench checkout (WORKBENCH_REPO_PATH) and an"
  echo "[install] OPENROUTER_API_KEY (no Docker, no database). Optional JEPA world model needs"
  echo "[install] 'uv sync --project assets/WorkBench/purple --extra jepa' (torch/transformers)."
  echo "[install] See assets/WorkBench/README.md."
fi

if _component_selected "automationbench"; then
  sync_project "AutomationBench green" "$REPO_ROOT/assets/AutomationBench/green" "$PYTHON_BIN" 3 11
  sync_project "AutomationBench purple" "$REPO_ROOT/assets/AutomationBench/purple" "$PYTHON_BIN" 3 11
  echo "[install] AutomationBench purple extras (mcp_react)..."
  (cd "$REPO_ROOT/assets/AutomationBench/purple" && uv sync --extra mcp_react)
  echo "[install] AutomationBench green extras (upstream task loading/scoring)..."
  (cd "$REPO_ROOT/assets/AutomationBench/green" && uv sync --extra upstream)
  echo "[install] The bundled 'sample' target runs offline. Domain targets need an external"
  echo "[install] AutomationBench checkout (AUTOMATIONBENCH_REPO_PATH) and an OpenAI-compatible"
  echo "[install] policy endpoint (AUTOMATIONBENCH_LLM_MODEL / _API_ENDPOINT). Optional JEPA world"
  echo "[install] model needs 'uv sync --project assets/AutomationBench/purple --extra jepa'."
  echo "[install] See assets/AutomationBench/README.md."
fi

echo "[install] Done."
