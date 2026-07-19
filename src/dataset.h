// A minimal COCO-format dataset loader, generalizing what
// demos/train_step_demo.cpp originally hardcoded to one image. Consumes
// the directory format produced by gen_reference/gen_coco_dataset.py: a
// manifest.txt (one per-image filename per line, so the loader doesn't
// need to glob a directory) plus one binary file per image containing
// three write_arr'd records in order (image (H,W,3), boxes (N,4) cxcywh
// normalized, labels (N,) float-encoded int32) -- the same format
// gen_reference/gen_reference_real_image.py used for the single-image
// case, generalized to a whole directory.
#pragma once

#include "loss.h"

#include <cstdint>
#include <string>
#include <vector>

struct DatasetItem {
    std::vector<float> pixels_hwc; // (H,W,3) row-major, NOT yet WHCN-transposed
    int64_t width = 0, height = 0, channels = 0;
    DetectionTarget target;
};

// Loads a gen_coco_dataset.py output directory. Files are read lazily
// (per get() call, not cached) -- these are small per-image dumps, and
// not caching keeps memory bounded regardless of dataset size.
class CocoDataset {
public:
    bool load(const std::string & dir);
    size_t size() const { return files_.size(); }
    bool get(size_t i, DatasetItem & out) const;

private:
    std::string dir_;
    std::vector<std::string> files_;
};

// Converts a DatasetItem's (H,W,3) row-major pixels into the WHCN
// (W fastest) float layout ggml tensors expect -- the same transpose
// convention used throughout this project's test suite and demos (e.g.
// tests/test_segmentation_xlarge.cpp's whc_to_ggml).
std::vector<float> dataset_item_pixels_whcn(const DatasetItem & item);
