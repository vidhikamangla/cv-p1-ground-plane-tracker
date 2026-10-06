#!/usr/bin/env bash
# Run every tracker ablation row over the chosen sequences and build the table.
#
#   ./scripts/run_ablations.sh "seq1 seq2 seq3" [rows...]
#
# Rows needing ground space are skipped automatically when a sequence has no
# configs/homography/<seq>.json, so the image-space rows can be produced before
# calibration is finished.
set -euo pipefail
cd "$(dirname "$0")/.."

SEQS=${1:?usage: run_ablations.sh "seq1 seq2" [t0 t1 ...]}
shift || true
ROWS=("$@")
if [ ${#ROWS[@]} -eq 0 ]; then ROWS=(t0 t1 t2 t3 t4 t5 t6); fi

PY=${PY:-python}
DATA_ROOT=$($PY -c "import sys;sys.path.insert(0,'.');from src import config as C;print(C.load('configs/default.yaml').paths.data_root)")
TE_ROOT=$($PY -c "import sys;sys.path.insert(0,'.');from src import config as C;print(C.load('configs/default.yaml').paths.trackeval)")
FRAMES_ROOT=$($PY -c "import sys;sys.path.insert(0,'.');from src import config as C;print(C.load('configs/default.yaml').paths.frames_root)")
BENCH=SportsMOT; SPLIT=val; NAME="$BENCH-$SPLIT"

mkdir -p outputs/results outputs/tracks outputs/traces

for row in "${ROWS[@]}"; do
  CFG="configs/ablations/$row.yaml"
  RUN=$($PY -c "import sys;sys.path.insert(0,'.');from src import config as C;print(C.load('$CFG').run_name)")
  SPACE=$($PY -c "import sys;sys.path.insert(0,'.');from src import config as C;print(C.load('$CFG').tracker.space)")
  OUT_DIR="$DATA_ROOT/trackers/mot_challenge/$NAME/$RUN/data"
  mkdir -p "$OUT_DIR"

  RAN=0; TOTAL_FRAMES=0; TOTAL_TIME=0
  for seq in $SEQS; do
    HOMO="configs/homography/$seq.json"
    if [ "$SPACE" = "ground" ] && [ ! -f "$HOMO" ]; then
      echo "  [$RUN] skip $seq - no $HOMO yet"; continue
    fi
    ARGS=(--config "$CFG" --frames-dir "$FRAMES_ROOT/$seq/img1"
          --detections "outputs/dets/$seq.jsonl"
          --out "$OUT_DIR/$seq.txt"
          --traces "outputs/traces/$RUN/$seq.npz")
    [ -f "$HOMO" ] && ARGS+=(--homography "$HOMO")
    FPS=$($PY - "$FRAMES_ROOT/$seq" <<'PYX'
import configparser, sys
from pathlib import Path
ini = Path(sys.argv[1]) / "seqinfo.ini"
cp = configparser.ConfigParser()
print(int(cp.read(ini) and cp["Sequence"].get("frameRate", 25)) if ini.exists() else 25)
PYX
)
    ARGS+=(--fps "$FPS")
    echo "  [$RUN] $seq (fps=$FPS)"
    LOG=$($PY -m src.run_tracker "${ARGS[@]}")
    echo "    $LOG"
    RAN=$((RAN+1))
  done

  if [ "$RAN" -eq 0 ]; then echo "[$RUN] no sequences ran - skipped"; continue; fi
  $PY -m src.eval.run_trackeval --trackeval-root "$TE_ROOT" --data-root "$DATA_ROOT" \
      --benchmark "$BENCH" --split "$SPLIT" --tracker "$RUN" --sequences $SEQS \
      --out "outputs/results/${RUN}.csv" || echo "[$RUN] trackeval failed"
  cp -f "outputs/results/${RUN}_summary.json" "outputs/results/${RUN}_summary.json" 2>/dev/null || true
done

$PY -m src.eval.ablate --results-dir outputs/results \
    --out-md outputs/results/ablation.md --out-csv outputs/results/ablation.csv
