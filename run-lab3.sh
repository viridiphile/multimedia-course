#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAB="$REPO/3-video-summarization"
cd "$LAB"
PYTHON="$LAB/.venv/bin/python"
export MPLCONFIGDIR="$LAB/.mplconfig"

case "${1:-run}" in
    run|setup|test|verify) action="${1:-run}";;
    -h|--help)
        echo "Usage: run-lab3.sh [run|setup|test|verify] [arguments]"
        exit 0
        ;;
    *) echo "Unknown action '$1'. Run ./run-lab3.sh --help." >&2; exit 2;;
esac
if [[ $# -gt 0 ]]; then shift; fi

if [[ ! -x "$PYTHON" ]]; then
    "${LAB3_PYTHON:-python3}" -m venv "$LAB/.venv"
fi
"$PYTHON" - <<'PY'
from pathlib import Path
import sys

if Path(sys.prefix).resolve() != Path(".venv").resolve():
    sys.exit("The Lab 3 launcher requires its own .venv")
if not (3, 9) <= sys.version_info[:2] <= (3, 12):
    sys.exit("Use a native Python 3.9 through 3.12 to create the Lab 3 .venv")
PY

if ! "$PYTHON" - <<'PY'
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from pip._vendor.packaging.requirements import Requirement
import sys

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
    echo "Installing pinned Lab 3 wheels into $LAB/.venv"
    "$PYTHON" -m pip --disable-pip-version-check install --only-binary=:all: -r requirements-lock.txt
    "$PYTHON" -m pip --disable-pip-version-check check
fi

case "$action" in
    setup) exec "$PYTHON" -m scripts.prepare_dataset "$@";;
    test) exec "$PYTHON" -m pytest "$@";;
    verify) exec "$PYTHON" -m scripts.verify_results "$@";;
    run) exec "$PYTHON" -m scripts.run_experiment "$@";;
esac
