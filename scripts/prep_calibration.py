#!/usr/bin/env python3
"""Build the calibration array for Hailo quantisation from the licence-clean corpus.

Preprocessing matches `RFDETRKeypointPreview.predict`: bilinear resize to square,
antialias=False. Normalization is left to `compile_hef.py`, which compiles it in.

Run:  pixi run -e gate python scripts/prep_calibration.py --out calib_576.npy
"""
from __future__ import annotations

import argparse
import glob
import io
import os
import sys


def holdout_ids(path):
    import pyarrow.parquet as pq
    if not os.path.exists(path):
        sys.exit("FAIL  no holdout manifest at %s: cannot prove these frames are not the "
                 "blinded set" % path)
    return set(pq.read_table(path)["image_id"].to_pylist())


def load(shard_dir, holdout, n, resolution):
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from PIL import Image
    from torchvision.transforms import functional as F

    shards = sorted(glob.glob(os.path.join(shard_dir, "images_*.parquet")))
    if not shards:
        sys.exit("FAIL  no shards under %s: run download_coco_images.py first" % shard_dir)

    out, ids = [], []
    for shard in shards:
        for row in pq.read_table(shard).to_pylist():
            if row["image_id"] in holdout:
                sys.exit("FAIL  CONTAMINATION: image_id %d is one of the %d blinded holdout "
                         "images. Refusing to calibrate on it." % (row["image_id"], len(holdout)))
            img = Image.open(io.BytesIO(row["image"])).convert("RGB")
            t = torch.from_numpy(np.array(img)).permute(2, 0, 1)
            t = F.resize(t, [resolution, resolution], antialias=False)
            out.append(t.permute(1, 2, 0).numpy())
            ids.append(row["image_id"])
            if len(out) >= n:
                return np.stack(out), ids
    return np.stack(out), ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards",
                    default="../../6-datasource/anny-render-corpus/coco_images_train2017")
    ap.add_argument("--holdout",
                    default="../../6-datasource/anny-render-corpus/"
                            "coco_person_commercial_val2017/images.parquet")
    ap.add_argument("--out", default="calib_576.npy")
    ap.add_argument("-n", type=int, default=256)
    ap.add_argument("--resolution", type=int, default=576)
    a = ap.parse_args()

    import numpy as np

    held = holdout_ids(a.holdout)
    arr, ids = load(a.shards, held, a.n, a.resolution)
    if arr.shape[0] < a.n:
        sys.exit("FAIL  asked for %d calibration frames, corpus yielded %d"
                 % (a.n, arr.shape[0]))
    if arr.dtype != np.uint8:
        sys.exit("FAIL  calibration array is %s, expected uint8" % arr.dtype)

    np.save(a.out, arr)
    print("  ok    %d frames, none among the %d blinded ids" % (len(ids), len(held)))
    print("  ok    %s  shape=%s dtype=%s  %.0f MB"
          % (a.out, arr.shape, arr.dtype, arr.nbytes / 1e6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
