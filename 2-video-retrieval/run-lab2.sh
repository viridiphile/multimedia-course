#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$LAB_DIR"
PYTHON="$LAB_DIR/.venv/bin/python"
export MPLCONFIGDIR="$LAB_DIR/.mplconfig"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

case "${1:-full}" in
    setup|smoke|full|test|verify) action="${1:-full}";;
    -h|--help) echo "Usage: run-lab2.sh [setup|smoke|full|test|verify] [arguments]"; exit 0;;
    *) echo "Unknown action: $1. Run $0 --help." >&2; exit 2;;
esac
if [[ $# -gt 0 ]]; then shift; fi

if [[ ! -x "$PYTHON" ]]; then
    "${LAB2_PYTHON:-python3}" -m venv "$LAB_DIR/.venv"
fi

"$PYTHON" - <<'PY'
from pathlib import Path
import sys

assert Path(sys.prefix).resolve() == Path(".venv").resolve(), "Expected the Lab 2 .venv"
if not ((3, 9) <= sys.version_info[:2] <= (3, 12)):
    sys.exit("Lab 2 requires Python 3.9 through 3.12. Use LAB2_PYTHON to select a base interpreter for a new environment.")
PY

if ! "$PYTHON" - <<'PY'
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import sys

from pip._vendor.packaging.requirements import Requirement

for line in Path("requirements-lock.txt").read_text().splitlines():
    if not line.strip() or line.startswith("#"):
        continue
    req = Requirement(line)
    if req.marker and not req.marker.evaluate():
        continue
    try:
        installed = version(req.name)
    except PackageNotFoundError:
        sys.exit(1)
    if installed not in req.specifier:
        sys.exit(1)
PY
then
    echo "Installing pinned Lab 2 wheels into $LAB_DIR/.venv"
    "$PYTHON" -m pip --disable-pip-version-check --no-cache-dir install \
        --only-binary=:all: -r "$LAB_DIR/requirements-lock.txt"
    "$PYTHON" -m pip --disable-pip-version-check check
fi

case "$action" in
    setup)
        exec "$PYTHON" -c 'from src.resources import prepare_dataset, prepare_checkpoint; print(prepare_dataset()); print(prepare_checkpoint())'
        ;;
    test) exec "$PYTHON" -m pytest "$@";;
    verify) exec "$PYTHON" -m src.verify "$@";;
    *) exec "$PYTHON" -m src.experiment "$action" "$@";;
esac
