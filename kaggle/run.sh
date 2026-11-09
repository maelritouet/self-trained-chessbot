#!/usr/bin/env bash
# Push and start a training run on Kaggle (private GPU notebook <user>/chessgpt-train).
#
#   kaggle/run.sh [--config configs/base.yaml] [--name base] [--hours 11] [--no-smoke] [--resume]
#
# The notebook checks out the exact commit that is HEAD here, so HEAD must be pushed
# and the working tree clean: every Kaggle run maps to code anyone can check out.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="configs/base.yaml"; NAME="base"; HOURS="11"; EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --name) NAME="$2"; shift 2 ;;
        --hours) HOURS="$2"; shift 2 ;;
        --no-smoke) EXTRA+=(--no-smoke); shift ;;
        --resume) EXTRA+=(--resume); shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
KAGGLE=".venv/bin/kaggle"

[[ -f "$CONFIG" ]] || { echo "no such config: $CONFIG" >&2; exit 1; }
if ! git diff --quiet HEAD -- . ; then
    echo "uncommitted changes to tracked files: commit them first (Kaggle runs the pushed commit)" >&2
    git status --short --untracked-files=no >&2
    exit 1
fi
REMOTE_URL=$(git remote get-url origin 2>/dev/null) || { echo "no 'origin' remote: create the GitHub repo and push first" >&2; exit 1; }
git fetch --quiet origin
COMMIT=$(git rev-parse HEAD)
if ! git branch -r --contains "$COMMIT" | grep -q .; then
    echo "HEAD $COMMIT is not on origin: push it first" >&2
    exit 1
fi
# git@github.com:u/r.git -> https://github.com/u/r.git (the notebook clones anonymously)
REPO_URL=$(sed -E 's#^git@github.com:#https://github.com/#' <<<"$REMOTE_URL")
USER_NAME="${KAGGLE_USER:-$(.venv/bin/python kaggle/whoami.py)}"

OUT="kaggle/build/kernel"
rm -rf "$OUT"
.venv/bin/python kaggle/build_kernel.py --user "$USER_NAME" --repo-url "$REPO_URL" \
    --git-ref "$COMMIT" --config "$CONFIG" --run-name "$NAME" --max-hours "$HOURS" \
    --out "$OUT" "${EXTRA[@]}"

echo "pushing $USER_NAME/chessgpt-train @ ${COMMIT:0:7}: config=$CONFIG name=$NAME hours=$HOURS ${EXTRA[*]}"
$KAGGLE kernels push -p "$OUT"
cat <<EOF

started. Follow it with:
  $KAGGLE kernels status $USER_NAME/chessgpt-train
  $KAGGLE kernels logs $USER_NAME/chessgpt-train --follow
then download outputs with:  kaggle/fetch_outputs.sh $NAME
EOF
