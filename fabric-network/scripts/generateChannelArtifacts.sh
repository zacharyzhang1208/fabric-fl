#!/usr/bin/env bash

set -e

ROOTDIR=$(cd "$(dirname "$0")/.." && pwd)
export FABRIC_CFG_PATH="${ROOTDIR}/configtx"
ARTIFACTS_DIR="${ROOTDIR}/channel-artifacts"
ORGANIZATIONS_DIR="${ROOTDIR}/organizations"
FABRIC_TOOL="${ROOTDIR}/scripts/utils/fabricTool.sh"

TRAINING_CHANNEL="trainingchannel"
TRAINING_PROFILE="TrainingChannel"

if [ ! -d "${ORGANIZATIONS_DIR}" ]; then
  echo "Error: organizations directory is missing."
  echo "Run ./scripts/createOrgs.sh before generating channel artifacts."
  exit 1
fi

mkdir -p "${ARTIFACTS_DIR}"

echo "Generating channel artifacts..."
"${FABRIC_TOOL}" configtxgen -profile "${TRAINING_PROFILE}" -channelID "${TRAINING_CHANNEL}" -outputBlock "${ARTIFACTS_DIR}/${TRAINING_CHANNEL}.block"

echo "Channel block ${TRAINING_CHANNEL}.block generated successfully"
