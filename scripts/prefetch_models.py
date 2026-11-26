"""Download the model pairs into the Hugging Face cache (the runtime loads them with local_files_only=True), at the
commits pinned in the config's model_revisions when present. Respects HF_ENDPOINT / HF_HUB_CACHE.
Usage: python scripts/prefetch_models.py [--config a.json,b.json] [main|x ...]"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from huggingface_hub import snapshot_download  # noqa: E402

from mlopd.experiment import load_config  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--config", default=None)
parser.add_argument("pairs", nargs="*", default=["main"])
args = parser.parse_args()
config = load_config(args.config)
revisions = config.get("model_revisions") or {}
for pair in args.pairs:
    for role in ("student", "teacher"):
        repo = config["model_pairs"][pair][role]
        path = snapshot_download(repo, revision=revisions.get(repo),
                                 allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
        print(repo, revisions.get(repo) or "default branch", path, flush=True)
