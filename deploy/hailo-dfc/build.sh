#!/bin/sh
# Assemble the build context from the repository, so nothing is hand-copied twice.
#
# The wheel stays out of git and is fetched by hand; everything else comes from the
# tree, because a second copy of a gate is the thing that drifts from the first.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)

test -f "$here"/hailo_dataflow_compiler-*.whl || {
  echo "missing: hailo_dataflow_compiler-*.whl -- fetch it from hailo.ai and put it here" >&2
  exit 1
}

mkdir -p "$here/scripts" "$here/models"
cp "$repo/scripts/gate_dfc_parse.py" "$here/scripts/"

# Export the graphs on this machine first; the Linux side only checks them.
echo "exporting device-half graphs with the macOS gate..."
for res in "$@"; do
  uv run --quiet --with torch --with numpy --with onnx --with onnxruntime --with rfdetr \
    python "$repo/scripts/gate_onnx_device.py" --resolution "$res" \
    --out "$here/models/backbone_$res.onnx"
done
echo "context ready: $(du -sh "$here" | cut -f1)"
