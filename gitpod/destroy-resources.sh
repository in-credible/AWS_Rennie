#!/usr/bin/env bash
# Tear down AWS CDK resources that were created from Gitpod.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CDK_APP_DIR="${REPO_ROOT}/CDK/AWS_CDK"
VENV_DIR="${CDK_APP_DIR}/.venv"
STACKS=("automation-rocks-ohio" "automation-rocks-nova")

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: Required command '$1' is not available." >&2
    exit 1
  fi
}

require_command aws
require_command python3
require_command npm

if ! command -v cdk >/dev/null 2>&1; then
  echo "Installing AWS CDK CLI (aws-cdk@1.76.0)..."
  npm install -g aws-cdk@1.76.0
fi

if [ ! -d "${VENV_DIR}" ]; then
  python3 -m venv "${VENV_DIR}"
fi
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

pushd "${CDK_APP_DIR}" >/dev/null
pip install --upgrade pip
pip install -r requirements.txt

for stack in "${STACKS[@]}"; do
  echo "Destroying ${stack}..."
  cdk destroy "${stack}" --force
  echo "${stack} removed."
done

popd >/dev/null
