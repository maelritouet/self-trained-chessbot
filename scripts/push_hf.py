"""Upload the web demo's models and the model card to the Hugging Face Hub (decision 014).

    hf auth login                                    # once; the token stays in the HF cache
    python scripts/push_hf.py --create               # first time: create the public repo
    python scripts/push_hf.py                        # later: upload a new version

Uploads web/models/<id>/{model.onnx,meta.json} for every model in web/models.release.json, as
<id>/model.onnx and <id>/meta.json, plus docs/model_card.md as README.md, in one commit to the
repo named by "hf_repo" in the manifest. Each file is checked against the manifest's SHA-256
first, so the Hub gets exactly the bytes the page expects (fetch them with
scripts/fetch_models.py if needed).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "web" / "models.release.json"
CARD = ROOT / "docs" / "model_card.md"


def _sha256(path: Path) -> str:
    spec = importlib.util.spec_from_file_location("fetch_models", ROOT / "scripts" / "fetch_models.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.sha256(path)


def operations(manifest: dict) -> list[CommitOperationAdd]:
    ops = [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(CARD))]
    for m in manifest["models"]:
        for filename, info in m["files"].items():
            src = ROOT / "web" / "models" / m["id"] / filename
            if not src.exists() or _sha256(src) != info["sha256"]:
                raise SystemExit(f"{src}: missing or not the manifest's version; run scripts/fetch_models.py")
            ops.append(CommitOperationAdd(path_in_repo=f"{m['id']}/{filename}", path_or_fileobj=str(src)))
    return ops


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--create", action="store_true", help="create the public model repo first")
    ap.add_argument("--message", default="Upload models and model card")
    a = ap.parse_args(argv)
    manifest = json.loads(MANIFEST.read_text())
    repo = manifest["hf_repo"]
    api = HfApi()
    if a.create:
        print(api.create_repo(repo, repo_type="model", private=False, exist_ok=True))
    ops = operations(manifest)
    info = api.create_commit(repo, operations=ops, commit_message=a.message, repo_type="model")
    print(f"{info.commit_url} ({len(ops)} files)")


if __name__ == "__main__":
    main()
