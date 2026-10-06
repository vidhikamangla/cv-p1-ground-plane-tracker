"""Thin wrapper around TrackEval -> one metrics CSV per run.

TrackEval is vendored by clone (pinned commit, numpy-2 patched) rather than pip
installed, because the PyPI copies are stale and silently differ on HOTA.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="evaluate MOT results with TrackEval")
    ap.add_argument("--trackeval-root", required=True, help="clone of JonathonLuiten/TrackEval")
    ap.add_argument("--data-root", required=True, help="dir containing gt/ and trackers/")
    ap.add_argument("--benchmark", default="SportsMOT")
    ap.add_argument("--split", default="val")
    ap.add_argument("--tracker", required=True, help="tracker folder name under trackers/")
    ap.add_argument("--sequences", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    sys.path.insert(0, str(Path(a.trackeval_root).resolve()))
    import trackeval  # noqa: E402

    name = f"{a.benchmark}-{a.split}"
    eval_cfg = trackeval.Evaluator.get_default_eval_config()
    eval_cfg.update({"USE_PARALLEL": a.workers > 1, "NUM_PARALLEL_CORES": a.workers,
                     "PRINT_RESULTS": True, "PRINT_CONFIG": False,
                     "TIME_PROGRESS": False, "DISPLAY_LESS_PROGRESS": True,
                     "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False,
                     "PLOT_CURVES": False})
    ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    ds_cfg.update({
        "GT_FOLDER": str(Path(a.data_root) / "gt" / "mot_challenge"),
        "TRACKERS_FOLDER": str(Path(a.data_root) / "trackers" / "mot_challenge"),
        "BENCHMARK": a.benchmark, "SPLIT_TO_EVAL": a.split,
        "TRACKERS_TO_EVAL": [a.tracker], "CLASSES_TO_EVAL": ["pedestrian"],
        "DO_PREPROC": False,          # SportsMOT GT has no static/distractor classes
        "PRINT_CONFIG": False,
    })
    if a.sequences:
        ds_cfg["SEQ_INFO"] = {s: None for s in a.sequences}
        ds_cfg["SEQMAP_FILE"] = None

    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    evaluator = trackeval.Evaluator(eval_cfg)
    res, _msg = evaluator.evaluate([trackeval.datasets.MotChallenge2DBox(ds_cfg)], metrics)

    # TrackEval returns ratio metrics in [0,1]; everyone reports them as percentages.
    # Counts (IDSW, Frag, and the raw TP/FP/FN tallies) must NOT be scaled.
    RATIO = {"HOTA", "DetA", "AssA", "LocA", "OWTA", "HOTA(0)", "LocA(0)", "HOTALocA(0)",
             "DetRe", "DetPr", "AssRe", "AssPr", "MOTA", "MOTP", "MODA", "IDF1", "IDR",
             "IDP", "Rcll", "Prcn", "sMOTA", "CLR_Re", "CLR_Pr", "MTR", "PTR", "MLR",
             "CLR_F1", "FP_per_frame"}

    key = "MotChallenge2DBox"
    per = res[key][a.tracker]
    rows = []
    for seq, seqres in per.items():
        c = seqres["pedestrian"]
        row = {"sequence": seq}
        for group in ("HOTA", "CLEAR", "Identity"):
            for k, v in c.get(group, {}).items():
                try:
                    import numpy as np
                    val = float(np.mean(v)) if hasattr(v, "__len__") else float(v)
                    row[k] = val * 100.0 if k in RATIO else val
                except Exception:
                    pass
        rows.append(row)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = ["sequence"] + sorted({k for r in rows for k in r if k != "sequence"})
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    comb = next((r for r in rows if r["sequence"] == "COMBINED_SEQ"), None)
    summary = {k: comb.get(k) for k in ("HOTA", "DetA", "AssA", "MOTA", "IDF1",
                                        "IDSW", "Frag", "MOTP")} if comb else {}
    (out.parent / (out.stem + "_summary.json")).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
