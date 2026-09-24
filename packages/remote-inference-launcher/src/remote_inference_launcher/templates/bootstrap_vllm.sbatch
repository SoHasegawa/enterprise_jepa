#!/usr/bin/env bash
@sbatch_directives

set -euo pipefail
OUT_DIR=@out_dir
ENVIRONMENT_NAME=@environment_name
VENV_PATH=@venv_path
VLLM_PACKAGE=@vllm_package
RAY_PACKAGE=@ray_package
BACKEND=@backend
INSTALL_UV_IF_MISSING=@install_uv_if_missing
FORCE_BOOTSTRAP=@force_bootstrap
HELPER_PATH=@helper_path

expand_home_path() {
  if [ "$1" = "~" ]; then
    printf "%s\n" "${HOME}"
  elif [ "${1#\~/}" != "$1" ]; then
    printf "%s/%s\n" "${HOME}" "${1#\~/}"
  else
    printf "%s\n" "$1"
  fi
}

prepend_venv_library_paths() {
  local venv_path="$1"
  local site_packages_dir
  local library_dir
  for site_packages_dir in "${venv_path}"/lib/python*/site-packages; do
    [ -d "${site_packages_dir}" ] || continue
    for library_dir in "${site_packages_dir}"/nvidia/*/lib "${site_packages_dir}"/torch/lib; do
      if [ -d "${library_dir}" ]; then
        export LD_LIBRARY_PATH="${library_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
      fi
    done
  done
}

OUT_DIR="$(expand_home_path "${OUT_DIR}")"
VENV_PATH="$(expand_home_path "${VENV_PATH}")"
VENV_PATH="${VENV_PATH%/}"
HELPER_PATH="$(expand_home_path "${HELPER_PATH}")"
mkdir -p "${OUT_DIR}"
exec > >(tee -a "${OUT_DIR}/bootstrap.log") 2>&1

echo "Bootstrapping ${VLLM_PACKAGE} into ${VENV_PATH}"
if [ -n "${RAY_PACKAGE}" ]; then
  echo "Ray package: ${RAY_PACKAGE}"
else
  echo "Ray package: disabled"
fi
echo "Environment name: ${ENVIRONMENT_NAME}"
echo "Install backend: ${BACKEND}"
echo "Running on $(hostname) with Slurm job ${SLURM_JOB_ID:-unknown}"
export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"

if [ ! -f "${HELPER_PATH}" ]; then
  echo "Missing vLLM bootstrap helper: ${HELPER_PATH}" >&2
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  if [ "${INSTALL_UV_IF_MISSING}" = "1" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
  else
    echo "uv is not installed on the allocated node." >&2
    echo "Rerun with --install-uv-if-missing to install uv in this Slurm job." >&2
    exit 2
  fi
fi

UV_BIN="$(command -v uv)"
UV_VERSION="$("${UV_BIN}" --version)"
echo "Using ${UV_VERSION} at ${UV_BIN}"
REUSED_EXISTING_ENV="0"

if [ -e "${VENV_PATH}" ] || [ -L "${VENV_PATH}" ]; then
  if [ "${FORCE_BOOTSTRAP}" = "1" ]; then
    if [ ! -f "${VENV_PATH}/pyvenv.cfg" ]; then
      echo "Refusing to remove remote bootstrap path because it is not a Python virtual environment: ${VENV_PATH}" >&2
      exit 2
    fi
    rm -rf -- "${VENV_PATH}"
  else
    echo "Remote bootstrap environment already exists: ${VENV_PATH}"
    if [ ! -x "${VENV_PATH}/bin/python" ]; then
      echo "Existing remote bootstrap environment is missing bin/python." >&2
      exit 2
    fi
    REUSED_EXISTING_ENV="1"
  fi
fi

if [ "${REUSED_EXISTING_ENV}" = "0" ]; then
  "${UV_BIN}" venv "${VENV_PATH}" --python 3.12 --seed --managed-python

  case "${BACKEND}" in
    rocm)
      "${UV_BIN}" pip install --python "${VENV_PATH}/bin/python" "${VLLM_PACKAGE}" --index https://wheels.vllm.ai/rocm/ --default-index https://pypi.org/simple --index-strategy unsafe-best-match --upgrade
      ;;
    auto)
      "${UV_BIN}" pip install --python "${VENV_PATH}/bin/python" "${VLLM_PACKAGE}" --torch-backend=auto
      ;;
    *)
      echo "Unsupported vLLM bootstrap backend: ${BACKEND}" >&2
      exit 2
      ;;
  esac
fi

if [ -n "${RAY_PACKAGE}" ]; then
  "${UV_BIN}" pip install --python "${VENV_PATH}/bin/python" "${RAY_PACKAGE}"
fi

MANIFEST_PATH="${VENV_PATH}/remote-inference-launcher-bootstrap-manifest.json"
SUCCESS_PATH="${OUT_DIR}/@success_manifest_name"
export RIL_BOOTSTRAP_BACKEND="${BACKEND}"
export RIL_BOOTSTRAP_ENVIRONMENT_NAME="${ENVIRONMENT_NAME}"
export RIL_BOOTSTRAP_FORCE="${FORCE_BOOTSTRAP}"
export RIL_BOOTSTRAP_INSTALL_UV_IF_MISSING="${INSTALL_UV_IF_MISSING}"
export RIL_BOOTSTRAP_REUSED_EXISTING_ENV="${REUSED_EXISTING_ENV}"
export RIL_BOOTSTRAP_UV_VERSION="${UV_VERSION}"
export RIL_BOOTSTRAP_VENV_PATH="${VENV_PATH}"
export RIL_BOOTSTRAP_VLLM_PACKAGE="${VLLM_PACKAGE}"
export RIL_BOOTSTRAP_RAY_PACKAGE="${RAY_PACKAGE}"
prepend_venv_library_paths "${VENV_PATH}"
"${VENV_PATH}/bin/python" "${HELPER_PATH}" "${MANIFEST_PATH}" "${SUCCESS_PATH}"

echo "Remote vLLM bootstrap manifest: ${MANIFEST_PATH}"
