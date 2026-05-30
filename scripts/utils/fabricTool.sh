#!/usr/bin/env bash

set -euo pipefail

ROOTDIR=$(cd "$(dirname "$0")/../.." && pwd)
FABRIC_TOOLS_IMAGE="${FABRIC_TOOLS_IMAGE:-hyperledger/fabric-tools:2.5.15}"
FABRIC_TOOL_NETWORK="${FABRIC_TOOL_NETWORK:-}"

DOCKER_ARGS=(--rm --user "$(id -u):$(id -g)")
if [ -n "${FABRIC_TOOL_NETWORK}" ]; then
  DOCKER_ARGS+=(--network "${FABRIC_TOOL_NETWORK}")
fi

ENV_ARGS=()
for name in \
  FABRIC_CFG_PATH \
  CORE_PEER_LOCALMSPID \
  CORE_PEER_MSPCONFIGPATH \
  CORE_PEER_ADDRESS \
  CORE_PEER_TLS_ENABLED \
  CORE_PEER_TLS_ROOTCERT_FILE; do
  if [ -n "${!name:-}" ]; then
    ENV_ARGS+=(-e "${name}=${!name}")
  fi
done

docker run "${DOCKER_ARGS[@]}" \
  -v "${ROOTDIR}:${ROOTDIR}" \
  -w "${ROOTDIR}" \
  "${ENV_ARGS[@]}" \
  "${FABRIC_TOOLS_IMAGE}" \
  "$@"
