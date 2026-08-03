#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ADAPTER_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
RUNTIME_DIR="${ADAPTER_DIR}/.run"
PID_FILE="${RUNTIME_DIR}/fabric-adapter.pid"
LOG_FILE="${RUNTIME_DIR}/fabric-adapter.log"
BINARY="${RUNTIME_DIR}/fabric-adapter"

usage() {
  cat <<EOF
Usage: $0 {start|stop|restart|status}

Commands:
  start    Build and start the Fabric adapter in the background
  stop     Stop the background Fabric adapter
  restart  Stop and start the Fabric adapter
  status   Show whether the Fabric adapter is running

Runtime files:
  PID: ${PID_FILE}
  Log: ${LOG_FILE}
EOF
}

read_pid() {
  if [ ! -f "${PID_FILE}" ]; then
    return 1
  fi
  local pid
  pid=$(cat "${PID_FILE}")
  if [[ ! "${pid}" =~ ^[0-9]+$ ]]; then
    return 1
  fi
  printf '%s\n' "${pid}"
}

is_running() {
  local pid=${1:-}
  if [ -z "${pid}" ] || ! kill -0 "${pid}" 2>/dev/null; then
    return 1
  fi
  if [ -r "/proc/${pid}/cmdline" ]; then
    local argument
    while IFS= read -r -d '' argument; do
      if [ "${argument}" = "${BINARY}" ]; then
        return 0
      fi
    done <"/proc/${pid}/cmdline"
    return 1
  fi
  return 0
}

start_adapter() {
  mkdir -p "${RUNTIME_DIR}"

  local existing_pid=""
  existing_pid=$(read_pid 2>/dev/null || true)
  if is_running "${existing_pid}"; then
    echo "Fabric adapter is already running (PID ${existing_pid})."
    echo "Log: ${LOG_FILE}"
    return 0
  fi
  rm -f "${PID_FILE}"

  # shellcheck source=fabric-env.sh
  source "${SCRIPT_DIR}/fabric-env.sh"

  echo "Building Fabric adapter..."
  (
    cd "${ADAPTER_DIR}"
    "${GO_BIN}" build -o "${BINARY}" ./cmd/fabric-adapter
  )

  {
    echo
    echo "===== start $(date '+%Y-%m-%d %H:%M:%S') ====="
  } >>"${LOG_FILE}"
  local log_start_line
  log_start_line=$(wc -l <"${LOG_FILE}")
  nohup "${BINARY}" >>"${LOG_FILE}" 2>&1 </dev/null &
  local pid=$!
  printf '%s\n' "${pid}" >"${PID_FILE}"

  local attempt
  for attempt in $(seq 1 50); do
    if ! is_running "${pid}"; then
      rm -f "${PID_FILE}"
      echo "Fabric adapter failed to start. Recent log output:" >&2
      tail -n 20 "${LOG_FILE}" >&2 || true
      return 1
    fi
    if tail -n "+$((log_start_line + 1))" "${LOG_FILE}" | grep -q "Fabric adapter listening on"; then
      echo "Fabric adapter started in the background (PID ${pid})."
      echo "Log: ${LOG_FILE}"
      return 0
    fi
    sleep 0.1
  done

  echo "Fabric adapter is running but readiness was not confirmed (PID ${pid})."
  echo "Check the log: ${LOG_FILE}"
}

stop_adapter() {
  local pid=""
  pid=$(read_pid 2>/dev/null || true)
  if ! is_running "${pid}"; then
    rm -f "${PID_FILE}"
    echo "Fabric adapter is not running."
    return 0
  fi

  echo "Stopping Fabric adapter (PID ${pid})..."
  kill -TERM "${pid}"
  local attempt
  for attempt in $(seq 1 100); do
    if ! is_running "${pid}"; then
      rm -f "${PID_FILE}"
      echo "Fabric adapter stopped."
      return 0
    fi
    sleep 0.1
  done

  echo "Fabric adapter did not stop gracefully; sending SIGKILL." >&2
  kill -KILL "${pid}" 2>/dev/null || true
  rm -f "${PID_FILE}"
  echo "Fabric adapter stopped."
}

status_adapter() {
  local pid=""
  pid=$(read_pid 2>/dev/null || true)
  if is_running "${pid}"; then
    echo "Fabric adapter is running (PID ${pid})."
    echo "Log: ${LOG_FILE}"
    return 0
  fi
  rm -f "${PID_FILE}"
  echo "Fabric adapter is not running."
  return 3
}

COMMAND=${1:-}
case "${COMMAND}" in
  start)
    start_adapter
    ;;
  stop)
    stop_adapter
    ;;
  restart)
    stop_adapter
    start_adapter
    ;;
  status)
    status_adapter
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
