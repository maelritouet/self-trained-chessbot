#!/usr/bin/env bash
# Upload trained checkpoints as a PRIVATE Kaggle dataset (<user>/chessgpt-models), so a
# fine-tune can start from them (kaggle/run.sh --init-from <name>.pt).
# First call creates it; later calls add a new version, which REPLACES the file set:
# pass every checkpoint you want to keep.
#
#   kaggle/push_models.sh name=path/to/best.pt [name=...] [-m message]
#   kaggle/push_models.sh base=runs/kaggle/base-20261110-1023/runs/base/best.pt \
#                         large=runs/kaggle/large-20261111-1208/runs/large/best.pt
set -euo pipefail
cd "$(dirname "$0")/.."

SLUG="chessgpt-models"
KAGGLE=".venv/bin/kaggle"
STAGE="kaggle/build/models"
MESSAGE=""

rm -rf "$STAGE" && mkdir -p "$STAGE"
while [[ $# -gt 0 ]]; do
    case "$1" in
        -m) MESSAGE="$2"; shift 2 ;;
        *=*) NAME="${1%%=*}"; SRC="${1#*=}"; shift
             [[ -f "$SRC" ]] || { echo "missing $SRC" >&2; exit 1; }
             # Hard link: no copy of a 100 MB checkpoint.
             ln "$SRC" "$STAGE/$NAME.pt" 2>/dev/null || cp "$SRC" "$STAGE/$NAME.pt"
             MESSAGE="${MESSAGE:+$MESSAGE | }$NAME.pt <- $SRC" ;;
        *) echo "expected name=path, got: $1" >&2; exit 2 ;;
    esac
done
ls "$STAGE"/*.pt >/dev/null 2>&1 || { echo "no checkpoints given" >&2; exit 2; }

USER_NAME="${KAGGLE_USER:-$(.venv/bin/python kaggle/whoami.py)}"
ID="$USER_NAME/$SLUG"
cat > "$STAGE/dataset-metadata.json" <<META
{
  "title": "$SLUG",
  "id": "$ID",
  "licenses": [{"name": "CC0-1.0"}],
  "subtitle": "chessgpt checkpoints (best.pt) to fine-tune from",
  "description": "best.pt files written by chessgpt.train: weights, model_args, vocab. Used by kaggle/run.sh --init-from."
}
META
du -sh "$STAGE"

if $KAGGLE datasets status "$ID" >/dev/null 2>&1; then
    echo "new version of $ID: $MESSAGE"
    $KAGGLE datasets version -p "$STAGE" -m "$MESSAGE"
else
    echo "creating private dataset $ID: $MESSAGE"
    $KAGGLE datasets create -p "$STAGE"   # private unless --public is passed
fi
echo "check processing with: $KAGGLE datasets status $ID"
