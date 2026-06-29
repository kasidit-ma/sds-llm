#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
./scripts/run_lint.sh
./scripts/run_typecheck.sh
./scripts/run_tests.sh
