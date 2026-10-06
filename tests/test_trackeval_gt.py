"""Feed the ground truth in as if it were the tracker's output.

If HOTA is not 100 the evaluation wiring is wrong - wrong sequence names, an
off-by-one on frame indices, a mismatched seqmap, or the wrong class filter. Every
later number is meaningless until this passes, which is why it is a test and not a
manual check.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TE = os.environ.get("TRACKEVAL_ROOT", "")
DATA = os.environ.get("TRACKEVAL_DATA", "")
pytestmark = pytest.mark.skipif(
    not (TE and DATA and Path(TE).is_dir() and Path(DATA).is_dir()),
    reason="set TRACKEVAL_ROOT and TRACKEVAL_DATA to run the GT round-trip")


def test_gt_as_tracker_scores_100(tmp_path):
    bench, split = os.environ.get("TE_BENCH", "SportsMOT"), os.environ.get("TE_SPLIT", "val")
    name = f"{bench}-{split}"
    gt_dir = Path(DATA) / "gt" / "mot_challenge" / name
    seqs = sorted(d.name for d in gt_dir.iterdir() if (d / "gt" / "gt.txt").exists())
    assert seqs, f"no sequences with gt.txt under {gt_dir}"
    seqs = seqs[:2]

    tracker = "gt_roundtrip"
    out_dir = Path(DATA) / "trackers" / "mot_challenge" / name / tracker / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        for s in seqs:
            rows = []
            for line in (gt_dir / s / "gt" / "gt.txt").read_text().splitlines():
                if not line.strip():
                    continue
                p = line.split(",")
                # GT columns: frame,id,x,y,w,h,conf,class,visibility
                if len(p) > 6 and float(p[6]) == 0:
                    continue                       # ignored GT boxes are not predictions
                rows.append(f"{int(float(p[0]))},{int(float(p[1]))},{float(p[2]):.2f},"
                            f"{float(p[3]):.2f},{float(p[4]):.2f},{float(p[5]):.2f},1,-1,-1,-1")
            (out_dir / f"{s}.txt").write_text("\n".join(rows) + "\n")

        res = tmp_path / "gt_roundtrip.csv"
        cmd = [sys.executable, "-m", "src.eval.run_trackeval",
               "--trackeval-root", TE, "--data-root", DATA,
               "--benchmark", bench, "--split", split, "--tracker", tracker,
               "--sequences", *seqs, "--out", str(res), "--workers", "2"]
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parents[1])
        assert proc.returncode == 0, proc.stdout + proc.stderr

        import json
        summary = json.loads((res.parent / (res.stem + "_summary.json")).read_text())
        assert summary["HOTA"] == pytest.approx(100.0, abs=0.5), summary
        assert summary["IDF1"] == pytest.approx(100.0, abs=0.5), summary
        assert int(summary["IDSW"]) == 0, summary
    finally:
        shutil.rmtree(out_dir.parent, ignore_errors=True)
