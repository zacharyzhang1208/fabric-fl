#!/usr/bin/env bash

set -e

ROOTDIR=$(cd "$(dirname "$0")/.." && pwd)
export FABRIC_CFG_PATH="${ROOTDIR}/configtx"
ARTIFACTS_DIR="${ROOTDIR}/channel-artifacts"
ORGANIZATIONS_DIR="${ROOTDIR}/organizations"
FABRIC_BIN_DIR="${FABRIC_BIN_DIR:-${ROOTDIR}/bin}"
CONFIGTXGEN="${CONFIGTXGEN:-${FABRIC_BIN_DIR}/configtxgen}"

TRAINING_CHANNEL="trainingchannel"
TRAINING_PROFILE="TrainingChannel"

if [ ! -d "${ORGANIZATIONS_DIR}" ]; then
  echo "Error: organizations directory is missing."
  echo "Run ./scripts/createOrgs.sh before generating channel artifacts."
  exit 1
fi

if [ ! -x "${CONFIGTXGEN}" ]; then
  echo "Error: configtxgen not found or not executable: ${CONFIGTXGEN}"
  exit 1
fi

mkdir -p "${ARTIFACTS_DIR}"

echo "Generating channel artifacts..."
"${CONFIGTXGEN}" -profile "${TRAINING_PROFILE}" -channelID "${TRAINING_CHANNEL}" -outputBlock "${ARTIFACTS_DIR}/${TRAINING_CHANNEL}.block"

echo "Channel block ${TRAINING_CHANNEL}.block generated successfully"
