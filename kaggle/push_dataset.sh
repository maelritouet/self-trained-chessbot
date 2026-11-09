#!/usr/bin/env bash
# Upload train.bin / val.bin / meta.json as a PRIVATE Kaggle dataset (<user>/chessgpt-data).
# First call creates it; later calls add a new version.
#
#   kaggle/push_dataset.sh [data_dir] [message]
#   kaggle/push_dataset.sh data/processed "2M games, min_elo 1500"
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR="${1:-data/processed}"
MESSAGE="${2:-}"
SLUG="${KAGGLE_DATASET_SLUG:-chessgpt-data}"
KAGGLE=".venv/bin/kaggle"
STAGE="kaggle/build/dataset"

for f in train.bin val.bin meta.json; do
    [[ -f "$DATA_DIR/$f" ]] || { echo "missing $DATA_DIR/$f (run scripts/build_dataset.py first)" >&2; exit 1; }
done

USER_NAME="${KAGGLE_USER:-$(.venv/bin/python kaggle/whoami.py)}"
ID="$USER_NAME/$SLUG"

# Stage the files with hard links (no copy of a GB-sized .bin), plus the metadata.
rm -rf "$STAGE" && mkdir -p "$STAGE"
for f in train.bin val.bin meta.json games.stats.json; do
    if [[ -f "$DATA_DIR/$f" ]]; then
        ln "$DATA_DIR/$f" "$STAGE/$f" 2>/dev/null || cp "$DATA_DIR/$f" "$STAGE/$f"
    fi
done
cat > "$STAGE/dataset-metadata.json" <<EOF
{
  "title": "$SLUG",
  "id": "$ID",
  "licenses": [{"name": "CC0-1.0"}],
  "subtitle": "Lichess movetext as uint8 char tokens for chessgpt",
  "description": "train.bin / val.bin (uint8 token ids, nanoGPT style) and meta.json (vocab, counts) built by scripts/build_dataset.py. Source: database.lichess.org (CC0)."
}
EOF

SUMMARY=$(.venv/bin/python -c "
import json; m = json.load(open('$STAGE/meta.json'))
print(f\"{m['n_games']['train'] + m['n_games']['val']:,} games, {m['n_tokens']['train']:,} train tokens\")")
MESSAGE="${MESSAGE:+$MESSAGE | }$SUMMARY | repo $(git rev-parse --short HEAD)"
du -sh "$STAGE"

if $KAGGLE datasets status "$ID" >/dev/null 2>&1; then
    echo "new version of $ID: $MESSAGE"
    $KAGGLE datasets version -p "$STAGE" -m "$MESSAGE"
else
    echo "creating private dataset $ID: $MESSAGE"
    $KAGGLE datasets create -p "$STAGE"   # private unless --public is passed
fi
echo "check processing with: $KAGGLE datasets status $ID"
