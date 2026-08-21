# The Linux half of the operator gate

`scripts/gate_onnx_device.py` runs on macOS and decides whether the device half stays
inside `DEVICE_OPS`. That list is a claim about the Hailo Dataflow Compiler, and only
the compiler can check it. The DFC wheel is tagged `py3-none-linux_x86_64`, so it does
not run on macOS at all, which is why this exists.

## Why Python 3.10

The wheel declares no `Requires-Python`. Its pins decide it instead: `tensorflow==2.19.1`,
`numpy==1.26.4`, `onnxruntime==1.18.0`. Python 3.10 is inside all three, and Ubuntu
22.04 ships 3.10 as its system Python, so the base image and the wheel already agree.

## The wheel is not in this repository

It is 522 MB and gated behind a Hailo developer account, so it is fetched by hand and
placed beside the `Dockerfile` before building. `.gitignore` keeps it out.

    cp ~/Downloads/hailo_dataflow_compiler-5.3.0-py3-none-linux_x86_64.whl .

## Build and run

    fly apps create weftspun-hailo-dfc
    fly deploy --remote-only --ha=false
    fly ssh console -a weftspun-hailo-dfc -C \
      "/opt/dfc/bin/python /work/scripts/gate_dfc_parse.py /work/models/backbone_576.onnx --expect pass"

Destroy the machine when the measurement is recorded. CLAUDE.md asks for the teardown
to be double-checked, and for anything that matters to be committed first: here that is
the HAR and the parse output, which are the measurement rather than a by-product.

    fly apps destroy weftspun-hailo-dfc
    fly apps list        # the double-check

## What a disagreement means

The two gates are run as a pair and their disagreement is the finding, not an error:

| macOS | DFC | meaning |
| --- | --- | --- |
| PASS | parses | the allowlist held |
| PASS | rejects X | `DEVICE_OPS` is too generous; move X to `KNOWN_BLOCKERS` |
| FAIL on X | parses | `DEVICE_OPS` is too strict; move X in |

`gate_dfc_parse.py` exits non-zero on any of the bottom two rows, including when the
parse itself succeeded.
