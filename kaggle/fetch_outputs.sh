#!/usr/bin/env bash
# Download the latest outputs of <user>/chessgpt-train into runs/kaggle/<name>-<timestamp>/.
#
#   kaggle/fetch_outputs.sh [name] [--no-ckpt]
#   --no-ckpt skips ckpt.pt (optimizer state, ~3x the model size): enough for evals.
set -euo pipefail
cd "$(dirname "$0")/.."

NAME="${1:-base}"
KAGGLE=".venv/bin/kaggle"
USER_NAME="${KAGGLE_USER:-$(.venv/bin/python kaggle/whoami.py)}"
DEST="runs/kaggle/$NAME-$(date +%Y%m%d-%H%M)"
mkdir -p "$DEST"

STATUS=$($KAGGLE kernels status "$USER_NAME/chessgpt-train" 2>&1 || true)
echo "$STATUS"
grep -qi "complete" <<<"$STATUS" || echo "warning: the run is not marked complete; outputs may be partial" >&2

PATTERN="runs/$NAME/.*"
[[ "${2:-}" == "--no-ckpt" ]] && PATTERN="runs/$NAME/(?!ckpt\\.pt).*"
$KAGGLE kernels output "$USER_NAME/chessgpt-train" -p "$DEST" --file-pattern "$PATTERN"

ls -la "$DEST" "$DEST/runs/$NAME" 2>/dev/null || ls -la "$DEST"
echo "log: $DEST/runs/$NAME/log.csv"
