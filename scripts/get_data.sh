#!/usr/bin/env bash
# Download + extract SportsMOT val (6.6 GB) and lay it out for TrackEval.
# Public and ungated on HuggingFace; val ground truth is public (test GT is not).
set -euo pipefail
DEST=${1:?usage: get_data.sh <data-dir>}
mkdir -p "$DEST/sportsmot_tar" "$DEST/sportsmot"
URL=https://huggingface.co/datasets/MCG-NJU/SportsMOT/resolve/main/dataset/val.tar
if [ ! -s "$DEST/sportsmot_tar/val.tar" ]; then
  echo "downloading SportsMOT val (~6.6 GB)..."
  curl -L --retry 3 --retry-delay 2 -o "$DEST/sportsmot_tar/val.tar" "$URL"
fi
if [ ! -d "$DEST/sportsmot/val" ]; then
  tar -xf "$DEST/sportsmot_tar/val.tar" -C "$DEST/sportsmot/"
fi
for f in basketball val; do
  curl -sL -o "$DEST/splits_$f.txt" \
    "https://huggingface.co/datasets/MCG-NJU/SportsMOT/resolve/main/splits_txt/$f.txt"
done
python - "$DEST" <<'PYX'
import sys
from pathlib import Path
d = Path(sys.argv[1])
b = set(d.joinpath("splits_basketball.txt").read_text().split())
v = set(d.joinpath("splits_val.txt").read_text().split())
both = sorted(b & v)
d.joinpath("basketball_val.txt").write_text("\n".join(both) + "\n")
print(f"{len(both)} basketball sequences in val -> {d/'basketball_val.txt'}")
PYX
python -m src.data.prepare_sportsmot --src "$DEST/sportsmot/val" --out "$DEST/trackeval_data"
