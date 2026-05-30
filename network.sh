#!/usr/bin/env bash

set -e

ROOTDIR=$(cd "$(dirname "$0")" && pwd)
COMPOSE_FILE="${ROOTDIR}/docker-compose.yaml"
ORGANIZATIONS_DIR="${ROOTDIR}/organizations"
CHANNEL_BLOCK="${ROOTDIR}/channel-artifacts/trainingchannel.block"

usage() {
  echo "Usage: $0 {up|down|restart|ps|logs|reset}"
  echo "  up       Start the Fabric containers in the background"
  echo "  down     Stop and remove the Fabric containers"
  echo "  restart  Stop and start the Fabric containers"
  echo "  ps       Show container status"
  echo "  logs     Follow container logs"
  echo "  reset    Stop containers and remove ledger volumes"
}

check_network_inputs() {
  if [ ! -d "${ORGANIZATIONS_DIR}" ]; then
    echo "Error: organizations directory is missing."
    echo "Run ./scripts/createOrgs.sh before starting the network."
    exit 1
  fi

  if [ ! -f "${CHANNEL_BLOCK}" ]; then
    echo "Error: channel block is missing: ${CHANNEL_BLOCK}"
    echo "Run ./scripts/generateChannelArtifacts.sh before starting the network."
    exit 1
  fi
}

case "${1:-}" in
  up)
    check_network_inputs
    docker compose -f "${COMPOSE_FILE}" up -d
    ;;
  down)
    docker compose -f "${COMPOSE_FILE}" down
    ;;
  restart)
    docker compose -f "${COMPOSE_FILE}" down
    check_network_inputs
    docker compose -f "${COMPOSE_FILE}" up -d
    ;;
  ps)
    docker compose -f "${COMPOSE_FILE}" ps
    ;;
  logs)
    docker compose -f "${COMPOSE_FILE}" logs -f
    ;;
  reset)
    docker compose -f "${COMPOSE_FILE}" down -v
    ;;
  *)
    usage
    exit 1
    ;;
esac
