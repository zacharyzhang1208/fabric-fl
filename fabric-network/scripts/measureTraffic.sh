#!/usr/bin/env bash

set -euo pipefail

ROOTDIR=$(cd "$(dirname "$0")/.." && pwd)
OUTPUT_DIR="${ROOTDIR}/traffic"

CONTAINERS=(
  orderer.org1.example.com
  orderer.org2.example.com
  orderer.org3.example.com
  orderer.org4.example.com
  orderer.org5.example.com
  peer0.org1.example.com
  peer0.org2.example.com
  peer0.org3.example.com
  peer0.org4.example.com
  peer0.org5.example.com
)

usage() {
  cat <<EOF
Usage:
  $0 start [NAME]
  $0 stop NAME
  $0 snapshot [NAME]

Writes Docker container network byte counters for Fabric peers and orderers.
Use start before an experiment and stop after it. Results are written under:
  ${OUTPUT_DIR}
EOF
}

container_bytes() {
  local container=$1
  local rx tx

  rx=$(docker exec "${container}" cat /sys/class/net/eth0/statistics/rx_bytes)
  tx=$(docker exec "${container}" cat /sys/class/net/eth0/statistics/tx_bytes)
  printf "%s,%s,%s,%s\n" "${container}" "${rx}" "${tx}" "$((rx + tx))"
}

write_snapshot() {
  local path=$1

  mkdir -p "$(dirname "${path}")"
  {
    printf "container,rx_bytes,tx_bytes,total_bytes\n"
    for container in "${CONTAINERS[@]}"; do
      container_bytes "${container}"
    done
  } > "${path}"
}

sum_column() {
  local file=$1
  local column=$2

  awk -F, -v column="${column}" 'NR > 1 {sum += $column} END {printf "%.0f", sum}' "${file}"
}

write_delta() {
  local start_file=$1
  local stop_file=$2
  local delta_file=$3
  local summary_file=$4

  awk -F, '
    NR == FNR {
      if (FNR > 1) {
        rx[$1] = $2
        tx[$1] = $3
        total[$1] = $4
      }
      next
    }
    FNR == 1 {
      print "container,rx_bytes,tx_bytes,total_bytes"
      next
    }
    {
      printf "%s,%d,%d,%d\n", $1, $2 - rx[$1], $3 - tx[$1], $4 - total[$1]
    }
  ' "${start_file}" "${stop_file}" > "${delta_file}"

  local rx tx total
  rx=$(sum_column "${delta_file}" 2)
  tx=$(sum_column "${delta_file}" 3)
  total=$(sum_column "${delta_file}" 4)
  {
    printf "metric,bytes\n"
    printf "rx_bytes,%s\n" "${rx}"
    printf "tx_bytes,%s\n" "${tx}"
    printf "total_bytes,%s\n" "${total}"
  } > "${summary_file}"
}

command=${1:-}
name=${2:-$(date +%Y%m%d-%H%M%S)}

case "${command}" in
  start)
    write_snapshot "${OUTPUT_DIR}/${name}.start.csv"
    echo "Traffic baseline written: ${OUTPUT_DIR}/${name}.start.csv"
    ;;
  stop)
    if [ $# -lt 2 ]; then
      usage
      exit 1
    fi
    start_file="${OUTPUT_DIR}/${name}.start.csv"
    stop_file="${OUTPUT_DIR}/${name}.stop.csv"
    delta_file="${OUTPUT_DIR}/${name}.delta.csv"
    summary_file="${OUTPUT_DIR}/${name}.summary.csv"
    if [ ! -f "${start_file}" ]; then
      echo "Missing start snapshot: ${start_file}" >&2
      exit 1
    fi
    write_snapshot "${stop_file}"
    write_delta "${start_file}" "${stop_file}" "${delta_file}" "${summary_file}"
    echo "Traffic stop snapshot written: ${stop_file}"
    echo "Traffic delta written: ${delta_file}"
    echo "Traffic summary written: ${summary_file}"
    cat "${summary_file}"
    ;;
  snapshot)
    write_snapshot "${OUTPUT_DIR}/${name}.snapshot.csv"
    echo "Traffic snapshot written: ${OUTPUT_DIR}/${name}.snapshot.csv"
    ;;
  *)
    usage
    exit 1
    ;;
esac
