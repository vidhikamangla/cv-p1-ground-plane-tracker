"""Market-1501 evaluation, implemented to the standard protocol.

The protocol detail that matters and that naive implementations get wrong: for each query
(person p, camera c), gallery images of **the same person from the same camera** are
neither correct nor incorrect - they are *junk* and must be removed from the ranking
before scoring, along with the distractor identity -1. Keep them and Rank-1 inflates
towards 100 because the easiest match is always the same person in the same view, which
is exactly the case cross-camera re-ID is supposed to exclude.

Also evaluated here: **multi-shot aggregation**. Market has several images per identity
per camera, so averaging their embeddings into one gallery vector is a direct measurement
of what track-level aggregation buys a tracker - the same operation `best_crop.aggregate`
performs on a track's best K crops.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

NAME_RE = re.compile(r"^(-?\d+)_c(\d+)")


def parse_name(p: Path):
    """-> (person_id, camera_id) from e.g. 0001_c1s1_001051_00.jpg"""
    m = NAME_RE.match(p.name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def list_split(root: Path):
    paths = sorted(p for p in root.glob("*.jpg") if not p.name.startswith("._"))
    pids, cams, keep = [], [], []
    for p in paths:
        pid, cam = parse_name(p)
        if pid is None:
            continue
        keep.append(p); pids.append(pid); cams.append(cam)
    return keep, np.asarray(pids), np.asarray(cams)


def evaluate(qf, q_pid, q_cam, gf, g_pid, g_cam, topk=(1, 5, 10)):
    """Cosine similarity ranking with junk removal. -> dict of CMC@k and mAP."""
    sim = qf @ gf.T                                  # both L2-normalised
    order = np.argsort(-sim, axis=1)

    all_cmc, all_ap, valid = np.zeros(max(topk)), [], 0
    for i in range(len(qf)):
        idx = order[i]
        # junk: same identity AND same camera, plus the -1 distractor identity
        junk = ((g_pid[idx] == q_pid[i]) & (g_cam[idx] == q_cam[i])) | (g_pid[idx] == -1)
        idx = idx[~junk]
        if idx.size == 0:
            continue
        matches = (g_pid[idx] == q_pid[i]).astype(np.int32)
        if matches.sum() == 0:
            continue                                  # no cross-camera ground truth
        valid += 1

        cmc = matches.cumsum()
        cmc[cmc > 1] = 1
        all_cmc += cmc[:max(topk)] if cmc.size >= max(topk) else \
            np.pad(cmc, (0, max(topk) - cmc.size), constant_values=cmc[-1])

        # average precision over the ranked list
        pos = np.where(matches == 1)[0]
        prec = [(r + 1) / (p + 1) for r, p in enumerate(pos)]
        all_ap.append(float(np.mean(prec)))

    if valid == 0:
        raise SystemExit("no valid queries - check the junk-removal logic")
    cmc = all_cmc / valid
    return {**{f"Rank-{k}": float(100 * cmc[k - 1]) for k in topk},
            "mAP": float(100 * np.mean(all_ap)), "n_queries": valid}


def aggregate_gallery(gf, g_pid, g_cam):
    """Collapse each (identity, camera) group into one averaged, re-normalised vector."""
    keys = {}
    for i, (p, c) in enumerate(zip(g_pid, g_cam)):
        keys.setdefault((int(p), int(c)), []).append(i)
    feats, pids, cams = [], [], []
    for (p, c), idxs in keys.items():
        v = gf[idxs].mean(0)
        v = v / max(float(np.linalg.norm(v)), 1e-12)
        feats.append(v); pids.append(p); cams.append(c)
    return np.stack(feats), np.asarray(pids), np.asarray(cams)


def main():
    ap = argparse.ArgumentParser(description="Market-1501 re-ID evaluation")
    ap.add_argument("--root", required=True, help="Market-1501-v15.09.15 directory")
    ap.add_argument("--model", default="vit_small_patch14_reg4_dinov2.lvd142m")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--aspect", type=float, default=1.0,
                    help="width/height of the model input; 0.5 keeps person proportions")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--cache", default="outputs/reid/market_feats.npz")
    ap.add_argument("--out", default="outputs/results/reid_market.json")
    ap.add_argument("--faiss", action="store_true", help="also time a FAISS index lookup")
    a = ap.parse_args()

    root = Path(a.root)
    qp, q_pid, q_cam = list_split(root / "query")
    gp, g_pid, g_cam = list_split(root / "bounding_box_test")
    print(f"{len(qp)} query images, {len(gp)} gallery images, "
          f"{len(set(q_pid.tolist()))} query identities")

    cache = Path(a.cache)
    if cache.exists():
        z = np.load(cache)
        qf, gf = z["qf"], z["gf"]
        print(f"loaded cached embeddings {qf.shape} / {gf.shape}")
    else:
        from .embed import Embedder
        emb = Embedder(a.model, size=a.size, aspect=a.aspect)
        qf, qk = emb.embed_paths(qp, batch=a.batch)
        gf, gk = emb.embed_paths(gp, batch=a.batch)
        q_pid, q_cam = q_pid[qk], q_cam[qk]
        g_pid, g_cam = g_pid[gk], g_cam[gk]
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, qf=qf, gf=gf, q_pid=q_pid, q_cam=q_cam, g_pid=g_pid, g_cam=g_cam)
        print(f"embedded -> {cache}  dim={emb.dim}")

    if cache.exists() and "q_pid" in np.load(cache):
        z = np.load(cache)
        q_pid, q_cam, g_pid, g_cam = z["q_pid"], z["q_cam"], z["g_pid"], z["g_cam"]

    results = {"model": a.model, "dim": int(qf.shape[1]), "aspect": a.aspect,
               "single_shot": evaluate(qf, q_pid, q_cam, gf, g_pid, g_cam)}
    print("single-shot gallery :", json.dumps(results["single_shot"]))

    ga_f, ga_pid, ga_cam = aggregate_gallery(gf, g_pid, g_cam)
    results["multi_shot_aggregated"] = evaluate(qf, q_pid, q_cam, ga_f, ga_pid, ga_cam)
    results["gallery_size_single"] = int(len(gf))
    results["gallery_size_aggregated"] = int(len(ga_f))
    print("aggregated gallery  :", json.dumps(results["multi_shot_aggregated"]))

    if a.faiss:
        import time
        import faiss
        index = faiss.IndexFlatIP(gf.shape[1])
        index.add(gf.astype(np.float32))
        t0 = time.perf_counter()
        D, I = index.search(qf.astype(np.float32), 10)
        dt = time.perf_counter() - t0
        results["faiss"] = {"index": "IndexFlatIP", "n_vectors": int(index.ntotal),
                            "query_ms_total": round(1000 * dt, 2),
                            "query_ms_per_query": round(1000 * dt / len(qf), 4)}
        print("faiss:", json.dumps(results["faiss"]))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
