"""Simplify at a fixed input shape, fold static Tile, then parse with the DFC.

Runs on the Linux machine, because it ends in a DFC parse.

WHAT IT FIXES. DFC 5.3.0 rejects the device half with

    UnsupportedConcatLayerError in op /encoder/encoder/embeddings/Tile:
    Unsupported concat over axis batch

That Tile replicates DINOv2's CLS token once per attention window, [1,1,384] to
[4,1,384] at 576 with num_windows=2. Both its inputs are constant once the input shape
is fixed, so it is a constant wearing an operator's clothes. Folding it is exact:
measured max|diff| 0.0 against the unfolded graph.

`onnxsim` alone does not remove it. Simplifying at a fixed input shape takes the graph
from 868 to 569 nodes and clears Expand, ConstantOfShape, Where and Shape entirely,
and Tile survives that pass. Hence the explicit fold here.

WHAT IT DOES NOT FIX, and this is the reason to read the output rather than the exit
code. A second blocker sits behind the first:

    UnsupportedModelError: Unsupported dimensions at concat layer
    (translated from /encoder/encoder/embeddings/Concat_1)

That is the CLS token joining the patch tokens, [4,1,384] with [4,576,384] giving
[4,577,384]. 576 is a 24x24 grid and 577 is not, and the DFC maps token sequences onto
spatial feature maps. The compiler's own recommended start_node_names and
end_node_names do not get past it.

A NOTE ON DIAGNOSING THIS COMPILER. Its first reported error is often not the real one.
A plain parse of the unfolded graph raises `IndexError: list index out of range` from
LayerNorm axis mapping, which is a downstream symptom of the simplify-and-retry path
rather than the cause. Passing `end_node_names` makes it report the honest error and a
recommendation. Minimal reproductions are also unreliable here: isolated rank-3
LayerNorm, an isolated window reshape and an isolated CLS concat all parse cleanly
while the same constructs fail inside the real graph, because format inference is
whole-graph rather than local.
"""
import collections, warnings, sys
warnings.filterwarnings("ignore")
import numpy as np, onnx, onnxruntime as ort
from onnx import helper, numpy_helper
from onnxsim import simplify
from hailo_sdk_client import ClientRunner

SRC = "/work/models/backbone_576.onnx"
SIM = "/work/models/backbone_576_folded.onnx"
RES = 576

m = onnx.load(SRC)
sm, ok = simplify(m, overwrite_input_shapes={"image": [1, 3, RES, RES]})
print(f"simplify valid={ok}: {len(m.graph.node)} -> {len(sm.graph.node)} nodes", flush=True)

g = sm.graph
targets = [n for n in g.node if n.op_type == "Tile"]
if targets:
    probe = onnx.ModelProto(); probe.CopyFrom(sm)
    probe.graph.output.extend(helper.make_empty_tensor_value_info(n.output[0]) for n in targets)
    so = ort.SessionOptions(); so.log_severity_level = 3
    s = ort.InferenceSession(probe.SerializeToString(), so, providers=["CPUExecutionProvider"])
    vals = dict(zip([o.name for o in s.get_outputs()],
                    s.run(None, {"image": np.random.randn(1, 3, RES, RES).astype(np.float32)})))
    for n in targets:
        v = vals[n.output[0]]
        g.initializer.append(numpy_helper.from_array(v, n.output[0]))
        g.node.remove(n)
        print(f"folded {n.name} -> initializer {list(v.shape)}", flush=True)
onnx.save(sm, SIM)
print(f"Tile now {collections.Counter(n.op_type for n in g.node)['Tile']}", flush=True)

print("=== DFC parse ===", flush=True)
try:
    ClientRunner(hw_arch="hailo10h").translate_onnx_model(SIM, "dh_folded")
    print("RESULT: PARSE OK")
except Exception as e:
    print(f"RESULT: REJECTED {type(e).__name__}")
    print(str(e)[:1800])
