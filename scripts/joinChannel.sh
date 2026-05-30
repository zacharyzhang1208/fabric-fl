#!/usr/bin/env bash

set -e

ROOTDIR=$(cd "$(dirname "$0")/.." && pwd)

CHANNEL_NAME="trainingchannel"
CHANNEL_BLOCK="${ROOTDIR}/channel-artifacts/${CHANNEL_NAME}.block"
BOOTSTRAP_ORDERER="orderer.org1.example.com:7050"
PEER_ADMIN_MSP="/etc/hyperledger/fabric/admin-msp"
PEER_ORDERER_CA="/etc/hyperledger/fabric/orderer-tls/ca.crt"

ORDERERS=(
  "org1.example.com orderer.org1.example.com 7053"
  "org2.example.com orderer.org2.example.com 7053"
  "org3.example.com orderer.org3.example.com 7053"
  "org4.example.com orderer.org4.example.com 7053"
  "org5.example.com orderer.org5.example.com 7053"
)

PEERS=(
  "peer0.org1.example.com Org1MSP"
  "peer0.org2.example.com Org2MSP"
  "peer0.org3.example.com Org3MSP"
  "peer0.org4.example.com Org4MSP"
  "peer0.org5.example.com Org5MSP"
)

if [ ! -f "${CHANNEL_BLOCK}" ]; then
  echo "Missing channel block: ${CHANNEL_BLOCK}"
  exit 1
fi

join_orderer_to_channel() {
  local org_domain=$1
  local orderer_name=$2
  local admin_port=$3

  if docker run --rm \
    -v "${ROOTDIR}/organizations:/etc/hyperledger/fabric/organizations" \
    -v "${ROOTDIR}/channel-artifacts:/etc/hyperledger/fabric/channel-artifacts" \
    --network fabric_test \
    hyperledger/fabric-tools:latest \
    osnadmin channel list \
    -o "${orderer_name}:${admin_port}" \
    --ca-file "/etc/hyperledger/fabric/organizations/ordererOrganizations/${org_domain}/orderers/${orderer_name}/tls/ca.crt" \
    --client-cert "/etc/hyperledger/fabric/organizations/ordererOrganizations/${org_domain}/orderers/${orderer_name}/tls/server.crt" \
    --client-key "/etc/hyperledger/fabric/organizations/ordererOrganizations/${org_domain}/orderers/${orderer_name}/tls/server.key" 2>/dev/null | grep -Eq "\"name\"[[:space:]]*:[[:space:]]*\"${CHANNEL_NAME}\""; then
    echo "✓ ${orderer_name} already joined ${CHANNEL_NAME}"
    return
  fi

  docker run --rm \
    -v "${ROOTDIR}/organizations:/etc/hyperledger/fabric/organizations" \
    -v "${ROOTDIR}/channel-artifacts:/etc/hyperledger/fabric/channel-artifacts" \
    --network fabric_test \
    hyperledger/fabric-tools:latest \
    osnadmin channel join \
    --channelID "${CHANNEL_NAME}" \
    --config-block "/etc/hyperledger/fabric/channel-artifacts/${CHANNEL_NAME}.block" \
    -o "${orderer_name}:${admin_port}" \
    --ca-file "/etc/hyperledger/fabric/organizations/ordererOrganizations/${org_domain}/orderers/${orderer_name}/tls/ca.crt" \
    --client-cert "/etc/hyperledger/fabric/organizations/ordererOrganizations/${org_domain}/orderers/${orderer_name}/tls/server.crt" \
    --client-key "/etc/hyperledger/fabric/organizations/ordererOrganizations/${org_domain}/orderers/${orderer_name}/tls/server.key"

  echo "✓ ${orderer_name} joined ${CHANNEL_NAME}"
}

wait_for_orderer() {
  echo "Waiting for ${BOOTSTRAP_ORDERER} to serve ${CHANNEL_NAME}..."

  for i in {1..30}; do
    if docker exec \
      -e CORE_PEER_LOCALMSPID=Org1MSP \
      -e CORE_PEER_MSPCONFIGPATH="${PEER_ADMIN_MSP}" \
      -e CORE_PEER_TLS_ENABLED=true \
      -e CORE_PEER_TLS_ROOTCERT_FILE=/etc/hyperledger/fabric/tls/ca.crt \
      peer0.org1.example.com \
      peer channel fetch 0 "/tmp/${CHANNEL_NAME}.block" \
        -o "${BOOTSTRAP_ORDERER}" \
        -c "${CHANNEL_NAME}" \
        --tls \
        --cafile "${PEER_ORDERER_CA}" >/dev/null 2>&1; then
      echo "✓ Orderer is ready for ${CHANNEL_NAME}"
      return
    fi

    echo "  Attempt ${i}/30: orderer not ready yet, retrying in 1s..."
    sleep 1
  done

  echo "✗ Orderer failed to become ready after 30 seconds"
  exit 1
}

join_peer_to_channel() {
  local peer_name=$1
  local msp_id=$2

  docker exec \
    -e CORE_PEER_LOCALMSPID="${msp_id}" \
    -e CORE_PEER_MSPCONFIGPATH="${PEER_ADMIN_MSP}" \
    -e CORE_PEER_TLS_ENABLED=true \
    -e CORE_PEER_TLS_ROOTCERT_FILE=/etc/hyperledger/fabric/tls/ca.crt \
    "${peer_name}" sh -c \
      "peer channel list | grep -q '${CHANNEL_NAME}' || (peer channel fetch 0 /tmp/${CHANNEL_NAME}.block -o ${BOOTSTRAP_ORDERER} -c ${CHANNEL_NAME} --tls --cafile ${PEER_ORDERER_CA} && peer channel join -b /tmp/${CHANNEL_NAME}.block)"

  echo "✓ ${peer_name} joined ${CHANNEL_NAME}"
}

echo "========================================"
echo "Joining orderers to ${CHANNEL_NAME}"
echo "========================================"

for orderer in "${ORDERERS[@]}"; do
  read -r org_domain orderer_name admin_port <<< "${orderer}"
  join_orderer_to_channel "${org_domain}" "${orderer_name}" "${admin_port}"
done

wait_for_orderer

echo ""
echo "========================================"
echo "Joining peers to ${CHANNEL_NAME}"
echo "========================================"

for peer in "${PEERS[@]}"; do
  read -r peer_name msp_id <<< "${peer}"
  join_peer_to_channel "${peer_name}" "${msp_id}"
done

echo ""
echo "========================================"
echo "✓ Channel join complete"
echo "========================================"
echo "${CHANNEL_NAME}: Org1 through Org5, one peer per organization"
