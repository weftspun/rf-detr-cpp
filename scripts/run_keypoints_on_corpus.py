"""Run the keypoint detector over corpus images and write annotated frames out.

END TO END, ON THE LICENCE-CLEAN HALF. Images come from `coco_images_train2017`, the 8,220
train2017 photographs that survived the licence filter and the holdout exclusion. They are
stored as zstd parquet with a sha256 beside every payload, so a frame drawn here can be
traced back to the row it came from.

THE HOLDOUT IS RE-ASSERTED HERE TOO, AND THAT IS NOT BELT-AND-BRACES. The 523 blinded images
must not be inspected while developing -- CLAUDE.md is explicit that looking at them to decide
whether an approach is working is training on them by hand, just slowly. This script DRAWS
PICTURES A HUMAN LOOKS AT, which is exactly that failure mode, so it checks rather than trusts
the shard it was handed.

num_windows=1 IS THE DEPLOYABLE CONFIGURATION, not the checkpoint default of 2. The default
exports the 868-node graph the Dataflow Compiler refuses; at one window it is 825 nodes and
every operator is inside the accepted set, at a measured 1.35x wall-clock. Running inference
at the configuration that can actually ship keeps this honest -- a demo at num_windows=2 would
be a demo of something that does not deploy.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import io
import os
import sys


def load_images(shard_dir, holdout_parquet, n):
    import pyarrow.parquet as pq
    from PIL import Image

    holdout = set()
    if os.path.exists(holdout_parquet):
        holdout = set(pq.read_table(holdout_parquet)["image_id"].to_pylist())
    else:
        sys.exit("FAIL  no holdout manifest at %s: cannot prove these frames are not the "
                 "blinded set, and an unproven claim here is the failure the rule names"
                 % holdout_parquet)

    out = []
    for shard in sorted(glob.glob(os.path.join(shard_dir, "images_*.parquet"))):
        for row in pq.read_table(shard).to_pylist():
            if row["image_id"] in holdout:
                sys.exit("FAIL  CONTAMINATION: image_id %d is one of the %d blinded holdout "
                         "images. Refusing to render it." % (row["image_id"], len(holdout)))
            if hashlib.sha256(row["image"]).hexdigest() != row["sha256"]:
                sys.exit("FAIL  sha256 mismatch on image_id %d" % row["image_id"])
            out.append((row["image_id"], Image.open(io.BytesIO(row["image"])).convert("RGB")))
            if len(out) >= n:
                return out, len(holdout)
    return out, len(holdout)


def draw(img, kp_result, threshold):
    """Annotate in place. Circles at each keypoint, radius scaled to the image so a 640px
    photo and a 1280px one look the same."""
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    r = max(2, int(min(img.size) * 0.008))
    n_drawn = 0
    xy = getattr(kp_result, "xy", None)
    conf = getattr(kp_result, "confidence", None)
    if xy is None:
        return 0
    for i, person in enumerate(xy):
        c = ["#ff3b30", "#34c759", "#0a84ff", "#ffd60a", "#bf5af2"][i % 5]
        for j, (x, y) in enumerate(person):
            if conf is not None:
                try:
                    if float(conf[i]) < threshold:
                        continue
                except (TypeError, IndexError):
                    pass
            if x <= 0 and y <= 0:
                continue                      # unlabelled slot, not a point at the origin
            d.ellipse([x - r, y - r, x + r, y + r], fill=c, outline="white")
            n_drawn += 1
    return n_drawn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="coco_images_train2017")
    ap.add_argument("--holdout", default="coco_person_commercial_val2017/images.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("-n", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--resolution", type=int, default=576)
    ap.add_argument("--num-windows", type=int, default=1)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    images, n_holdout = load_images(a.shards, a.holdout, a.n)
    print("  ok    %d images, none among the %d blinded holdout ids, all sha256-verified"
          % (len(images), n_holdout))

    from rfdetr import RFDETRKeypointPreview
    model = RFDETRKeypointPreview(resolution=a.resolution, num_windows=a.num_windows)
    print("  ..    model at resolution %d, num_windows=%d (the deployable configuration)"
          % (a.resolution, a.num_windows))

    total_pts = 0
    for image_id, img in images:
        res = model.predict(img, threshold=a.threshold)
        res = res[0] if isinstance(res, list) else res
        n = draw(img, res, a.threshold)
        total_pts += n
        path = os.path.join(a.out, "kp_%012d.png" % image_id)
        img.save(path)
        print("  ..    %s  %d keypoints drawn" % (os.path.basename(path), n))
    print("  ok    %d frames written to %s, %d keypoints total" % (len(images), a.out, total_pts))
    # Zero keypoints across every frame is a silent failure that looks like a clean run.
    if total_pts == 0:
        print("  FAIL  no keypoints on any frame: the model ran and drew nothing")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
