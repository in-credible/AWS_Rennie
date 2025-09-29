#!/usr/bin/env bash
# Tear down AWS CDK resources that were created from Gitpod.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CDK_APP_DIR="${REPO_ROOT}/CDK/AWS_CDK"
VENV_DIR="${CDK_APP_DIR}/.venv"
REQUIREMENTS_FILE="${CDK_APP_DIR}/requirements.txt"
REQUIREMENTS_HASH_FILE="${VENV_DIR}/.requirements.sha256"
STACKS=("automation-rocks-ohio" "automation-rocks-nova")
EXPECTED_CDK_VERSION="1.76.0"

cleanup() {
  if [[ -n "${VENV_ACTIVE:-}" ]]; then
    deactivate || true
  fi
}
trap cleanup EXIT

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: Required command '$1' is not available." >&2
    exit 1
  fi
}

ensure_cdk_cli() {
  if command -v cdk >/dev/null 2>&1; then
    local current
    current="$(cdk --version 2>/dev/null | awk '{print $1}')"
    if [[ "$current" == "$EXPECTED_CDK_VERSION" ]]; then
      return
    fi
    echo "Updating AWS CDK CLI to ${EXPECTED_CDK_VERSION} (current: ${current:-unknown})" >&2
  else
    echo "Installing AWS CDK CLI (aws-cdk@${EXPECTED_CDK_VERSION})..."
  fi
  npm install -g "aws-cdk@${EXPECTED_CDK_VERSION}" >/dev/null
}

ensure_virtualenv() {
  if [[ ! -d "${VENV_DIR}" ]]; then
    python3 -m venv "${VENV_DIR}"
  fi
  # shellcheck source=/dev/null
  source "${VENV_DIR}/bin/activate"
  VENV_ACTIVE=1
}

install_requirements() {
  local desired_hash current_hash
  desired_hash="$(python3 -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "${REQUIREMENTS_FILE}")"
  if [[ -f "${REQUIREMENTS_HASH_FILE}" ]]; then
    current_hash="$(<"${REQUIREMENTS_HASH_FILE}")"
  else
    current_hash=""
  fi
  if [[ "${desired_hash}" != "${current_hash}" ]]; then
    echo "Installing Python dependencies..."
    pip install --upgrade pip >/dev/null
    pip install -r "${REQUIREMENTS_FILE}" >/dev/null
    echo "${desired_hash}" > "${REQUIREMENTS_HASH_FILE}"
  fi
}

require_command aws
require_command python3
require_command npm

ensure_cdk_cli
ensure_virtualenv
install_requirements

pushd "${CDK_APP_DIR}" >/dev/null

for stack in "${STACKS[@]}"; do
  echo "Destroying ${stack}..."
  cdk destroy "${stack}" --force
  echo "${stack} removed."
done

popd >/dev/null
