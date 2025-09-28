#!/usr/bin/env bash
# Provision AWS CDK resources from a fresh Gitpod workspace.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CDK_APP_DIR="${REPO_ROOT}/CDK/AWS_CDK"
VENV_DIR="${CDK_APP_DIR}/.venv"
STACKS=("automation-rocks-nova" "automation-rocks-ohio")
REGIONS=("us-east-1" "us-east-2")

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

if ! ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null); then
  echo "ERROR: Unable to determine AWS account. Configure credentials before running this script." >&2
  exit 1
fi
ACCOUNT_ID="${ACCOUNT_ID//[$'\r\n']/}"

python3 -m venv "${VENV_DIR}"
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

pushd "${CDK_APP_DIR}" >/dev/null
pip install --upgrade pip
pip install -r requirements.txt

echo "Using AWS account ${ACCOUNT_ID}."
if [ "${SKIP_BOOTSTRAP:-0}" != "1" ]; then
  for region in "${REGIONS[@]}"; do
    echo "Bootstrapping ${region}..."
    cdk bootstrap "aws://${ACCOUNT_ID}/${region}"
  done
else
  echo "Skipping CDK bootstrap as requested (SKIP_BOOTSTRAP=${SKIP_BOOTSTRAP})."
fi

for stack in "${STACKS[@]}"; do
  echo "Deploying ${stack}..."
  cdk deploy "${stack}" --require-approval never
  echo "${stack} deployed."
done

popd >/dev/null
