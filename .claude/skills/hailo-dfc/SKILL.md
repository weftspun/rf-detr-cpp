---
name: hailo-dfc
description: Run the Hailo Dataflow Compiler gates from a Windows desk - parse, cost and emulate ONNX operators for hailo10h. Use when asked whether something deploys to the ASUS UGen300, when an operator is refused by the compiler, or when rewriting a refused operator into supported ones. Covers the Git Bash path-mangling and mount traps that cost five debugging rounds.
---

# Running the Dataflow Compiler from this desk

The DFC wheel is `linux_x86_64` only and this workspace is Windows. Docker Desktop and WSL
(Fedora) are both present, so the compiler runs in a container. Most of the difficulty is not
the compiler.

## The traps, in the order they bite

**`MSYS_NO_PATHCONV=1` on every `docker -v` and every `wsl` call.** Git Bash rewrites
anything that looks like a POSIX path. `wsl -- cat /home/x` becomes
`cat 'C:/Program Files/Git/home/x'`, and a `$VAR` inside a quoted `bash -lc` can arrive
empty. That silently produced a systemd unit reading `-v /op_tests_deform:/w/deform`, Docker
created four empty directories, and the emulation reported an empty table rather than an
error. Export it once at the top of the command.

**Prefer a script file to inline quoting through `wsl`.** Even with path conversion off,
nested quotes through `wsl -d X -- bash -lc '...'` are unreliable. Write the script to
`/c/...`, then `wsl -- bash /mnt/c/...`.

**Mount paths are `/mnt/c/...` from WSL and `/c/...` from Git Bash.** Both reach the same
Windows directory; the daemon is Windows-side either way.

**The compiler writes into the directory it reads.** Its simplify-and-retry path emits
`<name>.sim.onnx` beside the input, so a second run globs its own artefacts and reports 72
operators where 48 were tested. Scan the manifest's list, not the directory, and clean
`*.sim.onnx` and `*.har` after every run.

## The four gates, in order

    scripts/gen_op_tests.py          one minimal ONNX graph per operator, self-verified
    deploy/hailo-dfc/scripts/gate_dfc_ops.py     parse each: accepted or refused
    deploy/hailo-dfc/scripts/measure_dfc_cost.py ONNX nodes against Hailo layers
    deploy/hailo-dfc/scripts/emulate_dfc.py      native and quantized against onnxruntime

Build once: `docker build -t weftspun-hailo-dfc deploy/hailo-dfc` (the 522 MB wheel must sit
beside the Dockerfile; it is gitignored).

**A parse is not a measurement of correctness.** It says the operators are expressible. Only
`emulate_dfc.py` answers whether translation preserved the function, and it separates
translation error (native vs onnxruntime) from precision error (quantized vs native) because
those call for different fixes.

**A one-operator graph cannot tell "refused" from "absorbed".** Reshape-like operators are
folded into their neighbours and emit no layer, which surfaces as
`InvalidHNError: node name Y in end_node_names is missing in the HN` — identical to a real
refusal. Sandwich the operator between convolutions, and record any that cannot be wrapped so
their verdict is read with that caveat.

## Long runs as a systemd unit

WSL Fedora has systemd enabled, so a run that outlives a shell goes there:

    wsl -d FedoraLinux-44 -- systemctl --user restart hailo-emulate
    wsl -d FedoraLinux-44 -- journalctl --user -u hailo-emulate -f

`Type=oneshot` with `RemainAfterExit=yes`, never `simple`: this is a measurement that ends,
and `simple` reports success the moment docker starts.

## Rewriting a refused operator

Four have been done — `GridSample`, `ScatterND`, `GatherElements`, `TopK` — and one kernel
does all of them. Bilinear interpolation is a tent that vanishes beyond one pixel, so at
integer positions it is exactly a one-hot:

    tent(t)   = relu(1 - sqrt(t*t))         |t| the long way: Abs is not in the operator set
    out[i]    = sum_k data[k] * tent(idx - k)

That moves the index out of the ADDRESS and into a MULTIPLIER, which is the whole trick: the
compiler refuses data-dependent addressing, not data-dependent arithmetic. Proved in
`weftspun/lean-deform-exact`, six theorems, zero admitted goals.

**Three shapes the compiler refuses, learned by hitting each.** Implicit rank-3 broadcasting
(`operands could not be broadcast together with shapes (104,) (52,)`). `.expand`, which torch
lowers to `Expand` plus `ConstantOfShape`/`Equal`/`Where` scaffolding. And `Reshape` used as a
view (`UnsupportedShuffleLayerError`). When an index is a compile-time constant it never
needed to be a tensor dimension: emit one small block per slot and `Concat`. More nodes, all
of them measured passing.

**Watch the numerics, not just the parse.** `max(a,b) = (a+b+|a-b|)/2` cancels catastrophically
when operands differ in magnitude -- a `-1e30` padding sentinel returned `0.0` instead of
`3.0`, and the error is about `eps * max(|a|,|b|)` absolute, so at float32 a large sentinel
alone costs ~1e-3. And an `eps` inside `sqrt(t*t + eps)` biases the weight by `sqrt(eps)` at
the knot: `1e-12` cost a part per million and removing it improved agreement a millionfold.

**Every rewrite ships a negative control that must fail.** An offset outside the clamp, a
fractional index, a sentinel of the wrong magnitude, a tie-break ramp too large or too small.
Without one, a rewrite that agrees with whatever it is handed proves nothing.

## Reporting

`num_windows` is a real compatibility switch, not a default to inherit:
`RFDETRKeypointPreviewConfig.num_windows` is 2, which exports the 868-node graph the DFC
refuses; at 1 it is 825 nodes and clears the operator set. Cost is 1.35x wall-clock.

Pair every physical measurement with a household object, per the workspace rule, and state
what was NOT measured -- no schedule, no cycle count, no device -- rather than letting a parse
be read as a deployment.
