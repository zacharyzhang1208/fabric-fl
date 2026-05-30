#!/usr/bin/env bash

set -e

ROOTDIR=$(cd "$(dirname "$0")/.." && pwd)
CRYPTO_CONFIG_DIR="${ROOTDIR}/crypto-config"
OUTPUT_DIR="${ROOTDIR}/organizations"
FABRIC_BIN_DIR="${FABRIC_BIN_DIR:-${ROOTDIR}/bin}"
CRYPTOGEN="${CRYPTOGEN:-${FABRIC_BIN_DIR}/cryptogen}"

if [ ! -x "${CRYPTOGEN}" ]; then
  echo "Error: cryptogen not found or not executable: ${CRYPTOGEN}"
  exit 1
fi

if [ -d "${OUTPUT_DIR}" ]; then
  if ! rm -rf "${OUTPUT_DIR}"; then
    echo "Error: failed to remove existing organizations directory: ${OUTPUT_DIR}"
    echo "Please remove it manually, then rerun this script."
    exit 1
  fi
fi

mkdir -p "${OUTPUT_DIR}"

echo "Generating crypto material..."
"${CRYPTOGEN}" generate --config="${CRYPTO_CONFIG_DIR}/crypto-config.yaml" --output="${OUTPUT_DIR}"

echo "Done! Organizations created in: ${OUTPUT_DIR}"
