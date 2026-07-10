#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=fabric-env.sh
source "${SCRIPT_DIR}/fabric-env.sh"

cd "${ADAPTER_DIR}"
exec "${GO_BIN}" run ./cmd/fabric-cli "$@"
