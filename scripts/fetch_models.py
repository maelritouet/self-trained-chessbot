"""Download the web demo's models from the GitHub release into web/models/ (decision 013).

    python scripts/fetch_models.py                  # all models in web/models.release.json
    python scripts/fetch_models.py --only base      # just one (49 MB instead of ~320 MB)

Each file is checked against the SHA-256 in web/models.release.json, so a cut or corrupted
download fails loudly instead of loading a broken model. Files already present with the right
checksum are skipped. Then web/models/models.json is written (the page's model list; the first
entry loads by default). Standard library only: playing needs no torch.

Maintainer side, to make a new release from exported models (scripts/export_onnx.py):

    python scripts/fetch_models.py --stage dist/release --tag v1.1 \\
        large="Large (25.2M params)" base="Base (10.7M params)"

copies web/models/<id>/{model.onnx,meta.json} to flat release names (<id>.onnx,
<id>.meta.json) in dist/release/ and rewrites web/models.release.json. Upload the staged files
to the release with that tag, then commit the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "web" / "models.release.json"
MODELS_DIR = ROOT / "web" / "models"
REPO = "maelritouet/self-trained-chessbot"
FILES = ("model.onnx", "meta.json")  # what the page loads from web/models/<id>/
CHUNK = 1 << 20


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def asset_name(model_id: str, filename: str) -> str:
    """Release files are flat: large/model.onnx -> large.onnx, large/meta.json -> large.meta.json."""
    return f"{model_id}.onnx" if filename == "model.onnx" else f"{model_id}.{filename}"


def download(url: str, dest: Path, expected_sha: str, expected_size: int) -> None:
    """Stream url to dest via a .part file; rename only if size and SHA-256 match."""
    part = dest.with_name(dest.name + ".part")
    h = hashlib.sha256()
    got = 0
    with urllib.request.urlopen(url) as res, part.open("wb") as out:
        while block := res.read(CHUNK):
            out.write(block)
            h.update(block)
            got += len(block)
            if expected_size > 1_000_000:
                print(f"\r  {dest.parent.name}/{dest.name}: {got / 1e6:.0f} of {expected_size / 1e6:.0f} MB",
                      end="", flush=True)
    if expected_size > 1_000_000:
        print()
    if got != expected_size or h.hexdigest() != expected_sha:
        part.unlink()
        raise SystemExit(f"{url}: checksum or size mismatch ({got} bytes); try again")
    part.replace(dest)


def fetch(manifest: dict, only: list[str] | None, base_url: str | None) -> list[dict]:
    models = [m for m in manifest["models"] if not only or m["id"] in only]
    missing = set(only or []) - {m["id"] for m in models}
    if missing:
        raise SystemExit(f"not in the manifest: {', '.join(sorted(missing))}")
    base = base_url or f"https://github.com/{manifest['repo']}/releases/download/{manifest['tag']}"
    for m in models:
        for filename, info in m["files"].items():
            dest = MODELS_DIR / m["id"] / filename
            if dest.exists() and sha256(dest) == info["sha256"]:
                print(f"  {m['id']}/{filename}: already here")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            url = f"{base}/{asset_name(m['id'], filename)}"
            try:
                download(url, dest, info["sha256"], info["size"])
            except urllib.error.URLError as err:  # HTTPError (e.g. 404) is a subclass
                raise SystemExit(f"{url}: {getattr(err, 'reason', err)}") from None
    return models


def write_model_list(models: list[dict]) -> Path:
    path = MODELS_DIR / "models.json"
    path.write_text(json.dumps([{"id": m["id"], "label": m["label"]} for m in models], indent=2) + "\n")
    return path


def stage(out: Path, tag: str, specs: list[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    models = []
    for spec in specs:
        model_id, sep, label = spec.partition("=")
        if not sep or not label:
            raise SystemExit(f"expected id=label, got {spec!r}")
        files = {}
        for filename in FILES:
            src = MODELS_DIR / model_id / filename
            if not src.exists():
                raise SystemExit(f"{src} not found: export it with scripts/export_onnx.py first")
            shutil.copyfile(src, out / asset_name(model_id, filename))
            files[filename] = {"sha256": sha256(src), "size": src.stat().st_size}
        models.append({"id": model_id, "label": label, "files": files})
        print(f"  staged {model_id}")
    MANIFEST.write_text(json.dumps({"repo": REPO, "tag": tag, "models": models}, indent=2) + "\n")
    print(f"{out}: upload these files to release {tag}; manifest written to {MANIFEST.relative_to(ROOT)}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", metavar="ID", help="model ids to fetch (default: all)")
    ap.add_argument("--base-url", help="download from here instead of the GitHub release (for testing)")
    ap.add_argument("--stage", type=Path, metavar="DIR", help="maintainer: stage release files into DIR")
    ap.add_argument("--tag", help="with --stage: the release tag")
    ap.add_argument("specs", nargs="*", metavar="ID=LABEL", help="with --stage: models, in menu order")
    args = ap.parse_args(argv)
    if args.stage:
        if not args.tag or not args.specs:
            ap.error("--stage needs --tag and at least one ID=LABEL")
        stage(args.stage, args.tag, args.specs)
        return
    manifest = json.loads(MANIFEST.read_text())
    print(f"Fetching models from release {manifest['tag']} into {MODELS_DIR.relative_to(ROOT)}/")
    models = fetch(manifest, args.only, args.base_url)
    print(f"Wrote {write_model_list(models).relative_to(ROOT)}. "
          "Now: cd web && python3 -m http.server 8765, then open http://localhost:8765/")


if __name__ == "__main__":
    sys.exit(main())
