#!/usr/bin/env bash

set -euo pipefail

ROOTDIR=$(cd "$(dirname "$0")/.." && pwd)
export FABRIC_CFG_PATH="${ROOTDIR}/config"

FABRIC_BIN_DIR="${FABRIC_BIN_DIR:-${ROOTDIR}/bin}"
PEER_BIN="${PEER_BIN:-${FABRIC_BIN_DIR}/peer}"

CC_NAME="contracts"
CC_SRC_PATH="${ROOTDIR}/chaincode"
CC_VERSION="1.0"
CC_LABEL="${CC_NAME}_${CC_VERSION}"
CC_PACKAGE="${ROOTDIR}/${CC_LABEL}.tgz"
CHANNEL_NAME="trainingchannel"
ORDERER_ADDRESS="localhost:7050"
ORDERER_HOSTNAME="orderer.org1.example.com"
ORDERER_CA="${ROOTDIR}/organizations/ordererOrganizations/org1.example.com/orderers/orderer.org1.example.com/tls/ca.crt"

ORG_NAMES=(Org1 Org2 Org3 Org4 Org5)
ORG_MSPIDS=(Org1MSP Org2MSP Org3MSP Org4MSP Org5MSP)
ORG_DOMAINS=(org1.example.com org2.example.com org3.example.com org4.example.com org5.example.com)
ORG_PEER_ADDRESSES=(localhost:7051 localhost:9051 localhost:11051 localhost:12051 localhost:13051)

if [ ! -x "${PEER_BIN}" ]; then
  echo "Error: peer binary not found or not executable: ${PEER_BIN}"
  exit 1
fi

if ! command -v go &> /dev/null; then
  echo "Error: go not found"
  echo "Go is required to package golang chaincode with peer lifecycle chaincode package."
  exit 1
fi

if [ ! -d "${CC_SRC_PATH}" ]; then
  echo "Error: chaincode source directory is missing: ${CC_SRC_PATH}"
  exit 1
fi

if [ ! -f "${ORDERER_CA}" ]; then
  echo "Error: orderer TLS CA is missing: ${ORDERER_CA}"
  exit 1
fi

peer_tls_cert() {
  local index=$1
  echo "${ROOTDIR}/organizations/peerOrganizations/${ORG_DOMAINS[$index]}/peers/peer0.${ORG_DOMAINS[$index]}/tls/ca.crt"
}

peer_admin_msp() {
  local index=$1
  echo "${ROOTDIR}/organizations/peerOrganizations/${ORG_DOMAINS[$index]}/users/Admin@${ORG_DOMAINS[$index]}/msp"
}

set_peer_env() {
  local index=$1
  export CORE_PEER_LOCALMSPID="${ORG_MSPIDS[$index]}"
  export CORE_PEER_MSPCONFIGPATH
  CORE_PEER_MSPCONFIGPATH=$(peer_admin_msp "$index")
  export CORE_PEER_ADDRESS="${ORG_PEER_ADDRESSES[$index]}"
  export CORE_PEER_TLS_ENABLED=true
  export CORE_PEER_TLS_ROOTCERT_FILE
  CORE_PEER_TLS_ROOTCERT_FILE=$(peer_tls_cert "$index")
}

current_sequence() {
  local output

  set_peer_env 0
  if ! output=$("${PEER_BIN}" lifecycle chaincode querycommitted \
    --channelID "${CHANNEL_NAME}" \
    --name "${CC_NAME}" 2>/dev/null); then
    echo "0"
    return
  fi

  sed -n 's/.*Sequence: \([0-9][0-9]*\).*/\1/p' <<< "${output}" | head -n 1
}

install_chaincode() {
  local index=$1
  local output

  set_peer_env "$index"
  if output=$("${PEER_BIN}" lifecycle chaincode install "${CC_PACKAGE}" 2>&1); then
    echo "${output}"
    return
  fi

  if grep -qi "already successfully installed" <<< "${output}"; then
    echo "${output}"
    return
  fi

  echo "${output}"
  return 1
}

echo "=========================================="
echo "Deploying chaincode: ${CC_NAME}"
echo "Version: ${CC_VERSION}"
echo "Channel: ${CHANNEL_NAME}"
echo "=========================================="

echo ""
echo "Step 1: Packaging chaincode..."
rm -f "${CC_PACKAGE}"
"${PEER_BIN}" lifecycle chaincode package "${CC_PACKAGE}" \
  --path "${CC_SRC_PATH}" \
  --lang golang \
  --label "${CC_LABEL}"
echo "Packaged: ${CC_PACKAGE}"

echo ""
echo "Step 2: Calculating package ID..."
PACKAGE_ID=$("${PEER_BIN}" lifecycle chaincode calculatepackageid "${CC_PACKAGE}")
echo "Package ID: ${PACKAGE_ID}"

echo ""
echo "Step 3: Installing chaincode on all peers..."
for i in "${!ORG_NAMES[@]}"; do
  echo "Installing on ${ORG_NAMES[$i]} peer0 (${ORG_PEER_ADDRESSES[$i]})..."
  install_chaincode "$i"
done

echo ""
echo "Step 4: Approving chaincode for each organization..."
EXISTING_SEQUENCE=$(current_sequence)
EXISTING_SEQUENCE="${EXISTING_SEQUENCE:-0}"
CC_SEQUENCE=$((EXISTING_SEQUENCE + 1))
echo "Using sequence: ${CC_SEQUENCE}"

for i in "${!ORG_NAMES[@]}"; do
  echo "Approving for ${ORG_NAMES[$i]}..."
  set_peer_env "$i"
  "${PEER_BIN}" lifecycle chaincode approveformyorg \
    --channelID "${CHANNEL_NAME}" \
    --name "${CC_NAME}" \
    --version "${CC_VERSION}" \
    --package-id "${PACKAGE_ID}" \
    --sequence "${CC_SEQUENCE}" \
    --tls \
    --cafile "${ORDERER_CA}" \
    -o "${ORDERER_ADDRESS}" \
    --ordererTLSHostnameOverride "${ORDERER_HOSTNAME}"
done

echo ""
echo "Step 5: Checking commit readiness..."
set_peer_env 0
"${PEER_BIN}" lifecycle chaincode checkcommitreadiness \
  --channelID "${CHANNEL_NAME}" \
  --name "${CC_NAME}" \
  --version "${CC_VERSION}" \
  --sequence "${CC_SEQUENCE}" \
  --tls \
  --cafile "${ORDERER_CA}" \
  -o "${ORDERER_ADDRESS}" \
  --ordererTLSHostnameOverride "${ORDERER_HOSTNAME}" \
  --output json

echo ""
echo "Step 6: Committing chaincode definition..."
COMMIT_PEER_ARGS=()
for i in "${!ORG_NAMES[@]}"; do
  COMMIT_PEER_ARGS+=(--peerAddresses "${ORG_PEER_ADDRESSES[$i]}")
  COMMIT_PEER_ARGS+=(--tlsRootCertFiles "$(peer_tls_cert "$i")")
done

"${PEER_BIN}" lifecycle chaincode commit \
  --channelID "${CHANNEL_NAME}" \
  --name "${CC_NAME}" \
  --version "${CC_VERSION}" \
  --sequence "${CC_SEQUENCE}" \
  --tls \
  --cafile "${ORDERER_CA}" \
  -o "${ORDERER_ADDRESS}" \
  --ordererTLSHostnameOverride "${ORDERER_HOSTNAME}" \
  "${COMMIT_PEER_ARGS[@]}"

echo ""
echo "Step 7: Querying committed definition..."
"${PEER_BIN}" lifecycle chaincode querycommitted \
  --channelID "${CHANNEL_NAME}" \
  --name "${CC_NAME}"

echo ""
echo "=========================================="
echo "Chaincode deployment complete"
echo "=========================================="
