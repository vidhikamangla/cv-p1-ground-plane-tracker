"""Appearance embeddings for person crops.

Uses a self-supervised DINOv2 backbone off the shelf rather than a re-ID-trained network.
That is a deliberate, stated choice: it gives an honest *zero-shot* baseline with no
training on Market-1501, so the numbers below measure the retrieval pipeline - best-crop
selection, aggregation, indexing - rather than a borrowed fine-tuned checkpoint.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class Embedder:
    def __init__(self, model_name="vit_small_patch14_reg4_dinov2.lvd142m",
                 device="cuda:0", size=224, half=True, aspect=1.0):
        """`aspect` = width/height. Person crops are ~1:2, and squashing them into a
        square throws away the body proportions, so a 2:1 tall input is the fairer
        test of whether the embedding can see identity at all."""
        import timm
        self.device = torch.device(device)
        # DINOv2 checkpoints are published at a fixed 518px; timm rebuilds them at another
        # size by interpolating the position embeddings. 224 is 16 patches of 14, and
        # person crops are small enough that 518 would be mostly upsampling noise.
        try:
            hw = (size, max(14, int(round(size * aspect / 14)) * 14))
            self.model = timm.create_model(model_name, pretrained=True, num_classes=0,
                                           img_size=hw)
        except TypeError:
            self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model.eval().to(self.device)
        if half and self.device.type == "cuda":
            self.model = self.model.half()
        self.half = half and self.device.type == "cuda"
        self.size = size
        self.hw = (size, max(14, int(round(size * aspect / 14)) * 14))
        self.mean = torch.tensor(MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=self.device).view(1, 3, 1, 1)
        with torch.no_grad():
            d = self.model(torch.zeros(1, 3, *self.hw, device=self.device,
                                       dtype=torch.float16 if self.half else torch.float32))
        self.dim = int(d.shape[-1])

    @torch.no_grad()
    def embed_bgr(self, crops, batch=64):
        """crops: list of HWC BGR uint8 arrays -> [N, dim] L2-normalised float32."""
        import cv2
        out = []
        for i in range(0, len(crops), batch):
            chunk = crops[i:i + batch]
            arr = np.stack([cv2.resize(c, (self.hw[1], self.hw[0]),
                                       interpolation=cv2.INTER_LINEAR)[:, :, ::-1]
                            for c in chunk])
            t = torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 3, 1, 2)))
            t = t.to(self.device).float().div_(255.0).sub_(self.mean).div_(self.std)
            if self.half:
                t = t.half()
            f = self.model(t).float()
            f = torch.nn.functional.normalize(f, dim=1)
            out.append(f.cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.dim), np.float32)

    def embed_paths(self, paths, batch=64, progress=True):
        import cv2
        from tqdm import tqdm
        feats, keep = [], []
        it = range(0, len(paths), batch)
        if progress:
            it = tqdm(list(it), desc="embedding")
        for i in it:
            chunk = paths[i:i + batch]
            crops, idx = [], []
            for j, p in enumerate(chunk):
                im = cv2.imread(str(p))
                if im is not None:
                    crops.append(im)
                    idx.append(i + j)
            if crops:
                feats.append(self.embed_bgr(crops, batch=batch))
                keep.extend(idx)
        return (np.concatenate(feats) if feats else np.zeros((0, self.dim), np.float32),
                np.asarray(keep, dtype=np.int64))


def main():
    ap = argparse.ArgumentParser(description="embed a folder of crops")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="vit_small_patch14_reg4_dinov2.lvd142m")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    paths = sorted(p for p in Path(a.images).rglob("*.jpg") if not p.name.startswith("._"))
    emb = Embedder(a.model, size=a.size)
    feats, keep = emb.embed_paths(paths, batch=a.batch)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, feats=feats, names=np.asarray([paths[i].name for i in keep]))
    print(f"{len(feats)} embeddings of dim {emb.dim} -> {a.out}")


if __name__ == "__main__":
    main()
