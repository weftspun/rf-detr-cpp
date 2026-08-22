# Compiled layer census: backbone, 576, `num_windows=1`, Hailo-10H

Clipped from the HAR that DFC 5.3.0 produced, because the HAR itself is 102 MB and
regenerable from `scripts/gate_onnx_device.py` plus `scripts/fold_static_tile.py`. The
census is the measurement; the blob is not.

Reproduce with `deploy/hailo-dfc/build.sh 576` then `gate_dfc_parse.py`.

    ONNX in   825 nodes, 23 distinct operators
    HN out    336 layers
    arch      hailo10h, DFC 5.3.0

| count | Hailo layer type      |
| ----- | --------------------- |
| 130   | normalization         |
| 65    | conv                  |
| 46    | layer_normalization   |
| 25    | ew_add                |
| 24    | matmul                |
| 14    | feature_splitter      |
| 12    | softmax               |
| 5     | format_conversion     |
| 4     | concat                |
| 4     | slice                 |
| 2     | const_input           |
| 2     | shortcut              |
| 2     | output_layer          |
| 1     | input_layer           |

Two things worth reading off this rather than inferring.

**`layer_normalization` survives as a first-class layer, 46 of them.** The
`IndexError` in LayerNorm axis mapping that the unfolded graph raises is therefore not
LayerNorm being unsupported. It is a downstream symptom, which is what
`0005-hailo-edge-compile.md` records.

**4 concat layers compile.** Concat is not unsupported either. The rejected one was
concat over the batch axis specifically, which is what windowing produces.

Not measured here: cycles, latency, or utilisation. Those need the DFC profiler, and
every throughput figure quoted elsewhere assumes a 30% utilisation that nothing has
checked.
