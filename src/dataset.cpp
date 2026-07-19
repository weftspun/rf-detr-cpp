#include "dataset.h"

#include <cstdio>
#include <cstring>

// Reads one {i32 ndim, i64 dims[ndim], f32 data} record, matching
// gen_reference/*.py's write_arr format.
static bool read_arr(FILE * f, std::vector<int64_t> & shape, std::vector<float> & data) {
    int32_t ndim = 0;
    if (fread(&ndim, 4, 1, f) != 1 || ndim <= 0 || ndim > 8) {
        return false;
    }
    shape.resize(ndim);
    if (fread(shape.data(), 8, ndim, f) != (size_t) ndim) {
        return false;
    }
    int64_t n = 1;
    for (int64_t d : shape) {
        n *= d;
    }
    data.resize(n);
    return fread(data.data(), 4, n, f) == (size_t) n;
}

bool CocoDataset::load(const std::string & dir) {
    dir_ = dir;
    files_.clear();
    std::string manifest_path = dir + "/manifest.txt";
    FILE * f = fopen(manifest_path.c_str(), "rb");
    if (!f) {
        return false;
    }
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
            line[--len] = '\0';
        }
        if (len > 0) {
            files_.emplace_back(line);
        }
    }
    fclose(f);
    return !files_.empty();
}

bool CocoDataset::get(size_t i, DatasetItem & out) const {
    if (i >= files_.size()) {
        return false;
    }
    std::string path = dir_ + "/" + files_[i];
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }

    std::vector<int64_t> px_shape, box_shape, label_shape;
    std::vector<float> box_data, label_data_f;
    bool ok = read_arr(f, px_shape, out.pixels_hwc) &&
              read_arr(f, box_shape, box_data) &&
              read_arr(f, label_shape, label_data_f);
    fclose(f);
    if (!ok || px_shape.size() != 3) {
        return false;
    }

    out.height = px_shape[0];
    out.width = px_shape[1];
    out.channels = px_shape[2];
    out.target.boxes = box_data;
    out.target.labels.resize(label_data_f.size());
    for (size_t k = 0; k < label_data_f.size(); k++) {
        out.target.labels[k] = (int32_t) label_data_f[k];
    }
    return true;
}

std::vector<float> dataset_item_pixels_whcn(const DatasetItem & item) {
    const int64_t H = item.height, W = item.width, C = item.channels;
    std::vector<float> out((size_t) W * H * C);
    for (int64_t h = 0; h < H; h++) {
        for (int64_t w = 0; w < W; w++) {
            for (int64_t c = 0; c < C; c++) {
                out[c * H * W + h * W + w] = item.pixels_hwc[h * W * C + w * C + c];
            }
        }
    }
    return out;
}
