"""One minimal ONNX graph per operator the model uses, for the compiler to accept or refuse.

WHY PER-OPERATOR. `DEVICE_OPS` in gate_onnx_device.py is a hand-maintained allowlist, and
its own docstring says it "is a claim about this compiler, and only [the DFC] can check it".
Today that claim is checked in aggregate: a whole graph parses or does not, and a rejection
names one operator while saying nothing about the other forty-seven. The full keypoint model
uses 48 distinct operators, of which 15 sit outside DEVICE_OPS and only 4 of those appear in
KNOWN_BLOCKERS. The other 11 are neither permitted nor refused; they are unexamined.

A graph per operator turns the allowlist into a table of measurements.

EVERY GRAPH IS PROVEN BEFORE IT REACHES THE COMPILER. Each is checked with onnx.checker and
executed in onnxruntime here. Without that a DFC rejection is ambiguous between "the compiler
refuses this operator" and "we built a malformed graph", and the second reads exactly like
the first.

AN OPERATOR WITH NO BUILDER IS NAMED, NOT DROPPED. The required list is read from the real
exported model, so anything this file cannot construct is reported UNCHECKED and counted
against the total, per PITFALLS 3.

Usage:  python scripts/gen_op_tests.py --out op_tests [--ops scripts/model_ops.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto as T
from onnx import helper, numpy_helper

FLOAT = T.FLOAT
# NPU-plausible: a small NCHW activation. Spatial operators get this; the rest get the
# smallest shape that exercises the operator honestly.
IMG = (1, 8, 16, 16)


def _t(name, shape, dtype=FLOAT):
    return helper.make_tensor_value_info(name, dtype, list(shape))


def _init(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name)


def _out(shape=IMG, dtype=FLOAT):
    return _t("Y", shape, dtype)


def build(op):
    """Return (nodes, inputs, outputs, initializers) or None if unbuildable here."""
    X = _t("X", IMG)

    if op in {"Relu", "Sigmoid", "Sqrt", "Erf", "Exp", "Sin", "Cos", "Identity"}:
        return [helper.make_node(op, ["X"], ["Y"])], [X], [_out()], []

    if op in {"Add", "Sub", "Mul", "Div"}:
        w = _init("W", np.random.rand(*IMG).astype(np.float32) + 1.0)
        return [helper.make_node(op, ["X", "W"], ["Y"])], [X], [_out()], [w]

    # Predicates end on the bool rather than casting it away, because whether the compiler
    # carries a bool tensor is part of the answer.
    if op in {"Greater", "Less", "Equal"}:
        w = _init("W", np.random.rand(*IMG).astype(np.float32))
        return [helper.make_node(op, ["X", "W"], ["Y"])], [X], [_out(IMG, T.BOOL)], [w]
    if op in {"IsInf", "IsNaN"}:
        return [helper.make_node(op, ["X"], ["Y"])], [X], [_out(IMG, T.BOOL)], []
    if op == "Not":
        B = _t("X", IMG, T.BOOL)
        return [helper.make_node("Not", ["X"], ["Y"])], [B], [_out(IMG, T.BOOL)], []
    if op == "And":
        B = _t("X", IMG, T.BOOL)
        w = _init("W", np.ones(IMG, dtype=bool))
        return [helper.make_node("And", ["X", "W"], ["Y"])], [B], [_out(IMG, T.BOOL)], [w]
    if op == "Where":
        c = _init("C", (np.random.rand(*IMG) > 0.5))
        w = _init("W", np.zeros(IMG, dtype=np.float32))
        return [helper.make_node("Where", ["C", "X", "W"], ["Y"])], [X], [_out()], [c, w]

    # OPSET 17 PUTS `axes` IN DIFFERENT PLACES FOR DIFFERENT REDUCERS, and the checker
    # caught it: ReduceSum took axes as an input at 13, ReduceMax not until 18. Getting
    # this wrong would have sent a malformed graph to the compiler and read back as a
    # rejection of the operator.
    if op in {"ReduceMean", "ReduceMax"}:
        n = helper.make_node(op, ["X"], ["Y"], axes=[2, 3], keepdims=1)
        return [n], [X], [_out((1, 8, 1, 1))], []
    if op == "ReduceSum":
        ax = _init("axes", np.array([2, 3], dtype=np.int64))
        n = helper.make_node(op, ["X", "axes"], ["Y"], keepdims=1)
        return [n], [X], [_out((1, 8, 1, 1))], [ax]

    if op == "Reshape":
        s = _init("S", np.array([1, 8, 256], dtype=np.int64))
        return [helper.make_node("Reshape", ["X", "S"], ["Y"])], [X], [_out((1, 8, 256))], [s]
    if op == "Transpose":
        n = helper.make_node("Transpose", ["X"], ["Y"], perm=[0, 2, 3, 1])
        return [n], [X], [_out((1, 16, 16, 8))], []
    if op == "Flatten":
        return [helper.make_node("Flatten", ["X"], ["Y"], axis=1)], [X], [_out((1, 2048))], []
    if op == "Squeeze":
        a = _init("axes", np.array([0], dtype=np.int64))
        return [helper.make_node("Squeeze", ["X", "axes"], ["Y"])], [X], [_out((8, 16, 16))], [a]
    if op == "Unsqueeze":
        a = _init("axes", np.array([0], dtype=np.int64))
        return [helper.make_node("Unsqueeze", ["X", "axes"], ["Y"])], [X], [_out((1,) + IMG)], [a]
    if op == "Concat":
        w = _init("W", np.zeros(IMG, dtype=np.float32))
        n = helper.make_node("Concat", ["X", "W"], ["Y"], axis=1)
        return [n], [X], [_out((1, 16, 16, 16))], [w]
    if op == "Split":
        n = helper.make_node("Split", ["X"], ["Y", "Y2"], axis=1)
        return [n], [X], [_out((1, 4, 16, 16)), _t("Y2", (1, 4, 16, 16))], []
    if op == "Slice":
        st = _init("starts", np.array([0], dtype=np.int64))
        en = _init("ends", np.array([4], dtype=np.int64))
        ax = _init("axes", np.array([1], dtype=np.int64))
        n = helper.make_node("Slice", ["X", "starts", "ends", "axes"], ["Y"])
        return [n], [X], [_out((1, 4, 16, 16))], [st, en, ax]
    if op == "Pad":
        p = _init("pads", np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64))
        n = helper.make_node("Pad", ["X", "pads"], ["Y"], mode="constant")
        return [n], [X], [_out((1, 8, 18, 18))], [p]
    if op == "Expand":
        s = _init("S", np.array([2, 8, 16, 16], dtype=np.int64))
        return [helper.make_node("Expand", ["X", "S"], ["Y"])], [X], [_out((2, 8, 16, 16))], [s]
    if op == "Cast":
        n = helper.make_node("Cast", ["X"], ["Y"], to=int(T.FLOAT16))
        return [n], [X], [_out(IMG, T.FLOAT16)], []

    if op == "MatMul":
        A = _t("X", (1, 16, 32))
        w = _init("W", np.random.rand(32, 32).astype(np.float32))
        return [helper.make_node("MatMul", ["X", "W"], ["Y"])], [A], [_out((1, 16, 32))], [w]
    if op == "Gemm":
        A = _t("X", (16, 32))
        w = _init("W", np.random.rand(32, 32).astype(np.float32))
        b = _init("B", np.zeros(32, dtype=np.float32))
        return [helper.make_node("Gemm", ["X", "W", "B"], ["Y"])], [A], [_out((16, 32))], [w, b]
    if op == "Conv":
        w = _init("W", np.random.rand(8, 8, 3, 3).astype(np.float32))
        n = helper.make_node("Conv", ["X", "W"], ["Y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
        return [n], [X], [_out()], [w]
    if op == "LayerNormalization":
        A = _t("X", (1, 16, 32))
        s = _init("S", np.ones(32, dtype=np.float32))
        b = _init("B", np.zeros(32, dtype=np.float32))
        n = helper.make_node("LayerNormalization", ["X", "S", "B"], ["Y"], axis=-1)
        return [n], [A], [_out((1, 16, 32))], [s, b]
    if op == "Softmax":
        return [helper.make_node("Softmax", ["X"], ["Y"], axis=1)], [X], [_out()], []

    # The indexing family. Every known blocker lives here, which is the point.
    if op == "Gather":
        i = _init("I", np.array([0, 2, 4], dtype=np.int64))
        n = helper.make_node("Gather", ["X", "I"], ["Y"], axis=1)
        return [n], [X], [_out((1, 3, 16, 16))], [i]
    if op == "GatherElements":
        i = _init("I", np.zeros(IMG, dtype=np.int64))
        n = helper.make_node("GatherElements", ["X", "I"], ["Y"], axis=1)
        return [n], [X], [_out()], [i]
    if op == "ScatterND":
        idx = _init("I", np.array([[0, 0, 0, 0]], dtype=np.int64))
        upd = _init("U", np.array([1.0], dtype=np.float32))
        n = helper.make_node("ScatterND", ["X", "I", "U"], ["Y"])
        return [n], [X], [_out()], [idx, upd]
    if op == "TopK":
        A = _t("X", (1, 64))
        k = _init("K", np.array([8], dtype=np.int64))
        n = helper.make_node("TopK", ["X", "K"], ["Y", "Y2"], axis=1)
        return [n], [A], [_out((1, 8)), _t("Y2", (1, 8), T.INT64)], [k]
    if op == "GridSample":
        g = _init("G", np.random.uniform(-1, 1, (1, 16, 16, 2)).astype(np.float32))
        n = helper.make_node("GridSample", ["X", "G"], ["Y"],
                             mode="bilinear", padding_mode="zeros", align_corners=0)
        return [n], [X], [_out()], [g]
    if op == "Resize":
        roi = _init("roi", np.array([], dtype=np.float32))
        s = _init("scales", np.array([1, 1, 2, 2], dtype=np.float32))
        n = helper.make_node("Resize", ["X", "roi", "scales"], ["Y"], mode="linear")
        return [n], [X], [_out((1, 8, 32, 32))], [roi, s]

    # Shape-producing. Constant-foldable in a real graph, tested anyway because the model
    # still emits them and the compiler still has to swallow them.
    if op == "Shape":
        return [helper.make_node("Shape", ["X"], ["Y"])], [X], [_out((4,), T.INT64)], []
    if op == "ConstantOfShape":
        s = _init("S", np.array([1, 8, 16, 16], dtype=np.int64))
        n = helper.make_node("ConstantOfShape", ["S"], ["Y"],
                             value=numpy_helper.from_array(np.array([0.0], dtype=np.float32)))
        return [n], [], [_out()], [s]
    if op == "Range":
        st = _init("start", np.array(0, dtype=np.float32))
        li = _init("limit", np.array(16, dtype=np.float32))
        de = _init("delta", np.array(1, dtype=np.float32))
        n = helper.make_node("Range", ["start", "limit", "delta"], ["Y"])
        return [n], [], [_out((16,))], [st, li, de]
    if op == "Constant":
        n = helper.make_node("Constant", [], ["Y"],
                             value=numpy_helper.from_array(np.zeros(IMG, dtype=np.float32)))
        return [n], [], [_out()], []
    return None


def emit(op, out_dir):
    """(status, detail). Proven here before it is trusted as a compiler result."""
    built = build(op)
    if built is None:
        return "UNCHECKED", "no builder in gen_op_tests.py"
    nodes, inputs, outputs, inits = built

    # THE DFC MATCHES end_node_names AGAINST NODE NAMES, NOT TENSOR NAMES. Unnamed nodes
    # produced "InvalidHNError: The original node name Y in end_node_names is missing in
    # the HN" across two thirds of the corpus -- a harness artefact that reads exactly like
    # a refusal, which is the ambiguity this generator's own docstring warns about and the
    # first run walked straight into. Naming each node after the tensor it produces makes
    # the two agree.
    for n in nodes:
        if not n.name and len(n.output) >= 1:
            n.name = n.output[0]

    # AND IT WANTS A REAL GRAPH INPUT. Shape, Range and ConstantOfShape were built entirely
    # from initializers, so the compiler reported "Number of expected inputs: 1, Inputs
    # found: 0" -- again about the graph's shape rather than the operator. A dummy input,
    # added and immediately consumed, gives it one without changing what is being tested.
    if not inputs:
        dummy = _t("Xin", (1, 1, 1, 1))
        nodes = [helper.make_node("Identity", ["Xin"], ["Xid"], name="Xid")] + list(nodes)
        inputs = [dummy]
        outputs = list(outputs) + [_t("Xid", (1, 1, 1, 1))]

    # THE OP MUST SIT INSIDE A NETWORK, NOT BE ONE. A single-operator graph cannot
    # distinguish "the compiler refuses this" from "the compiler absorbed this into its
    # neighbour and emitted no layer for it" -- both surface as
    #
    #     InvalidHNError: The original node name Y in end_node_names is missing in the HN
    #
    # and reshape-like operators (Flatten, Squeeze, Expand) are absorbed by design. The
    # first two runs reported Flatten and Erf as REFUSED, which is not believable and was
    # the harness talking.
    #
    # So the operator under test is sandwiched: a Conv produces its input and a Conv
    # consumes its output, giving the compiler a real layer on either side and an
    # unambiguous end node. A bool or non-4D output is cast and reshaped back first, and an
    # operator whose output cannot be fed onward is recorded UNWRAPPED so its verdict is
    # read with that caveat rather than silently trusted.
    wrapped = False
    if inputs and len(outputs) == 1:
        o = outputs[0]
        shape = [d.dim_value for d in o.type.tensor_type.shape.dim]
        elem = o.type.tensor_type.elem_type
        if len(shape) == 4 and shape[0] == 1 and elem in (T.FLOAT, T.BOOL):
            pre_w = _init("PW", np.random.rand(IMG[1], IMG[1], 1, 1).astype(np.float32))
            head = [helper.make_node("Conv", ["Xin", "PW"], ["Xc"], name="Xc",
                                     kernel_shape=[1, 1])]
            inits = list(inits) + [pre_w]
            # A bool-input operator needs a bool between the Conv and itself, or the wrap
            # hands a float tensor to And/Not and the checker rejects the graph before the
            # compiler ever sees it.
            if inputs[0].type.tensor_type.elem_type == T.BOOL:
                thr = _init("THR", np.full(IMG, 0.5, dtype=np.float32))
                head.append(helper.make_node("Greater", ["Xc", "THR"], ["X"], name="X"))
                inits.append(thr)
            else:
                head.append(helper.make_node("Identity", ["Xc"], ["X"], name="X"))
            nodes = head + list(nodes)
            inputs = [_t("Xin", IMG)]
            tail = []
            src = "Y"
            if elem == T.BOOL:
                tail.append(helper.make_node("Cast", ["Y"], ["Yf"], name="Yf",
                                             to=int(T.FLOAT)))
                src = "Yf"
            post_w = _init("QW", np.random.rand(4, shape[1], 1, 1).astype(np.float32))
            tail.append(helper.make_node("Conv", [src, "QW"], ["Z"], name="Z",
                                         kernel_shape=[1, 1]))
            inits.append(post_w)
            nodes = list(nodes) + tail
            outputs = [_t("Z", (1, 4, shape[2], shape[3]))]
            wrapped = True

    graph = helper.make_graph(nodes, "op_" + op, inputs, outputs, initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    try:
        onnx.checker.check_model(model, full_check=True)
    except Exception as exc:                                          # noqa: BLE001
        return "BROKEN", "onnx.checker: %s: %s" % (type(exc).__name__, exc)
    path = os.path.join(out_dir, op + ".onnx")
    onnx.save(model, path)
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        feeds = {}
        for i in sess.get_inputs():
            shape = [d if isinstance(d, int) else 1 for d in i.shape]
            if i.type == "tensor(bool)":
                feeds[i.name] = (np.random.rand(*shape) > 0.5)
            else:
                feeds[i.name] = np.random.rand(*shape).astype(np.float32)
        sess.run(None, feeds)
    except Exception as exc:                                          # noqa: BLE001
        return "BROKEN", "onnxruntime: %s: %s" % (type(exc).__name__, exc)
    return ("OK" if wrapped else "OK-UNWRAPPED"), path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="op_tests")
    ap.add_argument("--ops", default=os.path.join("scripts", "model_ops.json"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    required = sorted(json.load(open(a.ops)))
    results = {op: emit(op, a.out) for op in required}
    ok = [o for o, (s, _) in results.items() if s.startswith("OK")]
    unwrapped = [o for o, (s, _) in results.items() if s == "OK-UNWRAPPED"]
    broken = {o: d for o, (s, d) in results.items() if s == "BROKEN"}
    unchecked = [o for o, (s, _) in results.items() if s == "UNCHECKED"]
    print("%d operators required by the model" % len(required))
    print("  built and proven : %d" % len(ok))
    print("  of those UNWRAPPED (verdict is weaker): %d  %s"
          % (len(unwrapped), ", ".join(unwrapped)))
    print("  BROKEN generator : %d" % len(broken))
    for o, d in broken.items():
        print("     %s: %s" % (o, d))
    print("  UNCHECKED (no builder, named not dropped): %d" % len(unchecked))
    for o in unchecked:
        print("     %s" % o)
    json.dump({"ok": ok, "unwrapped": unwrapped, "broken": broken,
               "unchecked": unchecked},
              open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    # A broken generator is a FAIL: it would otherwise reach the compiler and come back as
    # a rejection that says nothing about the operator.
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
