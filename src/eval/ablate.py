"""Collect per-row TrackEval summaries into one markdown + CSV table."""
import argparse
import csv
import json
from pathlib import Path

COLS = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "Frag"]
LABELS = {
    "t0_image_iou_sort": "image / IoU / 1-stage (SORT-like)",
    "t1_image_iou_cmc": "+ CMC",
    "t2_image_iou_cmc_2stage": "+ two-stage association",
    "t3_image_maha_cmc_2stage": "Mahalanobis cost (still image space)",
    "t4_ground_maha_cmc": "ground plane, H_t = H_ref @ W(t->ref)",
    "t5_ground_maha_cmc_colour": "+ jersey-colour gate",
    "t6_ground_static_nocmc": "ground plane, STATIC H_ref, no CMC",
    "t7_ground_iou_cmc_2stage": "ground plane + CMC, cost held at IoU (isolates the space)",
    "t8_ground_maha_cmc_jacobianR": "ground + CMC + per-detection R = J Rpx J^T",
    "t9_ground_adaptive_cmc": "ground + ADAPTIVE CMC (apply the warp only when the camera moved)",
}


def main():
    ap = argparse.ArgumentParser(description="build the tracker ablation table")
    ap.add_argument("--results-dir", required=True, help="dir of <run>_summary.json files")
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--order", nargs="*", default=list(LABELS))
    ap.add_argument("--note", default=None)
    a = ap.parse_args()

    rows = []
    for run in a.order:
        f = Path(a.results_dir) / f"{run}_summary.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        row = {"row": run.split("_")[0].upper(), "run": run,
               "change": LABELS.get(run, run)}
        for c in COLS:
            v = d.get(c)
            row[c] = None if v is None else (round(float(v), 2) if c not in ("IDSW", "Frag")
                                             else int(round(float(v))))
        fps_f = Path(a.results_dir) / f"{run}_fps.json"
        row["fps"] = round(json.loads(fps_f.read_text()).get("fps", 0), 1) if fps_f.exists() else None
        rows.append(row)

    if not rows:
        raise SystemExit(f"no *_summary.json found in {a.results_dir}")

    keys = ["row", "change"] + COLS + ["fps"]
    Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["row", "run", "change"] + COLS + ["fps"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    def fmt(v):
        return "-" if v is None else str(v)

    lines = ["| " + " | ".join(keys) + " |",
             "|" + "|".join(["---"] * len(keys)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(k)) for k in keys) + " |")
    if a.note:
        lines += ["", a.note]
    Path(a.out_md).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
