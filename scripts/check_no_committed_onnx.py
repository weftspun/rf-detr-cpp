#!/usr/bin/env python3
"""Gate: no ONNX artifact is committed to this repository.

CLAUDE.md blocklists ONNX as our interchange format and our runtime. `gate_onnx_device.py`
still writes one, and is exempt for a reason BLOCKLIST.md states: the file is a control
rather than a deliverable. It is exported, run once to diff against PyTorch, and dropped.

THE EXEMPTION IS BOUNDED BY WHERE THE FILE GOES, so something has to hold that boundary.
`--out` defaults under `tempfile.gettempdir()`, and a caller may pass any path they like --
including one inside the working tree, where it would be committed by the next `git add -A`
and become the thing the row forbids. That is one careless flag away and nothing reported
it before this.

WHAT IT CHECKS. Tracked files only. An untracked `.onnx` sitting in the tree is a scratch
file, which is the permitted case, and `.gitignore` already carries `*.sim.onnx` and
`*.har` for the compiler's own leavings.

    python scripts/check_no_committed_onnx.py [--self-test]
"""
import argparse
import subprocess
import sys

SUFFIXES = (".onnx", ".onnx.data", ".har", ".hef")


def tracked(root):
    out = subprocess.run(["git", "-C", root, "ls-files"], capture_output=True, text=True)
    if out.returncode != 0:
        return None
    return [f for f in out.stdout.split("\n") if f]


def scan(root):
    files = tracked(root)
    if files is None:
        return None
    return [f for f in files if f.endswith(SUFFIXES)]


def self_test():
    """A gate that has never rejected a planted artifact has not shown it can reject a real one."""
    import os
    import tempfile

    problems = []
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["git", "-C", d, "init", "-q"], check=True)
        subprocess.run(["git", "-C", d, "config", "user.email", "t@t"], check=True)
        subprocess.run(["git", "-C", d, "config", "user.name", "t"], check=True)
        open(os.path.join(d, "keep.py"), "w").write("# ordinary\n")
        subprocess.run(["git", "-C", d, "add", "-A"], check=True)
        subprocess.run(["git", "-C", d, "commit", "-qm", "x"], check=True)
        if scan(d):
            problems.append("FAIL: a clean tree reported an artifact")

        # untracked is the permitted case and must stay permitted
        open(os.path.join(d, "scratch.onnx"), "w").write("x")
        if scan(d):
            problems.append("FAIL: an UNTRACKED .onnx was rejected; scratch files are the exempt case")

        # tracked is the forbidden case
        subprocess.run(["git", "-C", d, "add", "-f", "scratch.onnx"], check=True)
        if not scan(d):
            problems.append("FAIL: a TRACKED .onnx went uncaught -- the gate is decoration")

        # and a .har, which the compiler writes
        open(os.path.join(d, "model.har"), "w").write("x")
        subprocess.run(["git", "-C", d, "add", "-f", "model.har"], check=True)
        if len(scan(d)) != 2:
            problems.append("FAIL: a tracked .har went uncaught")

    for p in problems:
        print(p)
    print("self-test: %d problem(s)" % len(problems))
    return 1 if problems else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()

    hits = scan(args.root)
    if hits is None:
        print("FAIL: not a git repository, so tracked files cannot be listed")
        return 1
    for h in hits:
        print(f"  {h}")
    if hits:
        print(f"FAIL: {len(hits)} ONNX/compiler artifact(s) committed; the exemption covers "
              "scratch files only -- see BLOCKLIST.md")
        return 1
    print("ok   no ONNX or compiler artifact is committed; the gate's output stays scratch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
