#!/usr/bin/env bash
# fastText lid.176 for the Latin-script languages of the target-language ratio (design §6.3).
set -euo pipefail
target="${1:-data/cache/lid.176.bin}"
mkdir -p "$(dirname "$target")"
curl -L -o "$target" https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
sha256sum "$target"
echo "export MLOPD_LID_MODEL=$PWD/$target   # and: pip install fasttext-wheel"
