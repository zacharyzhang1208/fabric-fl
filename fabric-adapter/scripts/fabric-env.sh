#!/usr/bin/env bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ADAPTER_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
ROOT_DIR=$(cd "${ADAPTER_DIR}/.." && pwd)
LOCAL_ENV="${ADAPTER_DIR}/fabric.env"

if [ -f "${LOCAL_ENV}" ]; then
  # shellcheck source=/dev/null
  source "${LOCAL_ENV}"
fi

resolve_path() {
  local path=${1:-}
  if [ -z "${path}" ] || [[ "${path}" = /* ]]; then
    echo "${path}"
  else
    echo "${ROOT_DIR}/${path}"
  fi
}

detect_key() {
  local key_dir="${ROOT_DIR}/fabric-network/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp/keystore"

  if [ -n "${FABRIC_KEY_PATH:-}" ]; then
    resolve_path "${FABRIC_KEY_PATH}"
    return
  fi

  if [ ! -d "${key_dir}" ]; then
    echo ""
    return
  fi

  find "${key_dir}" -maxdepth 1 -type f | sort | head -n 1
}

detect_cert() {
  local cert_dir="${ROOT_DIR}/fabric-network/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp/signcerts"

  if [ -n "${FABRIC_CERT_PATH:-}" ]; then
    resolve_path "${FABRIC_CERT_PATH}"
    return
  fi

  if [ ! -d "${cert_dir}" ]; then
    echo ""
    return
  fi

  find "${cert_dir}" -maxdepth 1 -type f -name "*.pem" | sort | head -n 1
}

export FABRIC_MSP_ID="${FABRIC_MSP_ID:-Org1MSP}"
export FABRIC_CERT_PATH
FABRIC_CERT_PATH=$(detect_cert)
export FABRIC_KEY_PATH
FABRIC_KEY_PATH=$(detect_key)
export FABRIC_TLS_CERT_PATH
FABRIC_TLS_CERT_PATH=$(resolve_path "${FABRIC_TLS_CERT_PATH:-fabric-network/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt}")
export FABRIC_PEER_ENDPOINT="${FABRIC_PEER_ENDPOINT:-localhost:7051}"
export FABRIC_PEER_HOST="${FABRIC_PEER_HOST:-peer0.org1.example.com}"
export FABRIC_CHANNEL="${FABRIC_CHANNEL:-trainingchannel}"
export FABRIC_CHAINCODE="${FABRIC_CHAINCODE:-contracts}"
export FABRIC_TIMEOUT="${FABRIC_TIMEOUT:-10s}"
export FABRIC_ADAPTER_ADDRESS="${FABRIC_ADAPTER_ADDRESS:-127.0.0.1:18080}"

GO_BIN="${GO_BIN:-go}"
if ! command -v "${GO_BIN}" >/dev/null 2>&1; then
  sdk_go=$(find "${HOME}/sdk" -mindepth 3 -maxdepth 3 -path "*/bin/go" -type f 2>/dev/null | sort -V | tail -n 1 || true)
  if [ -n "${sdk_go}" ]; then
    GO_BIN="${sdk_go}"
  fi
fi

if ! command -v "${GO_BIN}" >/dev/null 2>&1; then
  echo "Go is required to run the Fabric adapter, but it was not found in PATH." >&2
  echo "Install Go 1.20 or newer, or set GO_BIN=/path/to/go before retrying." >&2
  exit 127
fi

export ADAPTER_DIR ROOT_DIR GO_BIN
