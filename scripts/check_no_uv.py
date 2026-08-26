#!/usr/bin/env python3
"""Gate: no `uv` invocation reaches the tree.

CLAUDE.md blocklists `uv` for project environments -- "an environment nothing declares
and nobody can rebuild" -- and this repository declares its environments in `pixi.toml`.
Between those two facts sat 21 files whose documented run line was still
`uv run --with torch --with numpy --with rfdetr ...`, naming dependencies in a shell
argument with no lock and no versions. The manifest existed; the instructions pointed
somewhere else, so the manifest was decorative for anyone following the docstring.

WHY A GATE AND NOT A NOTE. The entry it enforces has no automation behind it anywhere in
the workspace, so it has been resting on everyone remembering. This one is cheap: a
substring, a skip list, an exit code.

WHAT IT DELIBERATELY ALLOWS. `pixi.toml` may name `uv` in prose, because the comment
explaining why an environment exists has to be able to say what it replaced. A gate that
forbids discussing the thing it forbids makes the reasoning unwritable.

    python scripts/check_no_uv.py [--self-test]
"""
import argparse
import pathlib
import re
import sys

PATTERN = re.compile(r"\buv\s+(?:run|pip|sync|add|venv)\b|\buvx\b")
SKIP_DIRS = {".git", ".pixi", "build", "node_modules", "__pycache__", ".venv"}
# The manifest carries the argument for its own existence and must name what it replaced.
ALLOW = {"pixi.toml", "scripts/check_no_uv.py"}
SUFFIXES = {".py", ".md", ".sh", ".toml", ".h", ".hpp", ".cpp", ".cc", ".yml", ".yaml", ".txt"}


def scan(root):
    hits = []
    for p in sorted(pathlib.Path(root).rglob("*")):
        if not p.is_file() or p.suffix not in SUFFIXES:
            continue
        if SKIP_DIRS & set(p.relative_to(root).parts):
            continue
        rel = str(p.relative_to(root))
        if rel in ALLOW:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if PATTERN.search(line):
                hits.append((rel, n, line.strip()))
    return hits


def self_test():
    """A gate that has never found a planted violation has not shown it can find a real one."""
    import tempfile

    problems = []
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        (root / "clean.py").write_text('# pixi run -e reference python thing.py\n')
        if scan(root):
            problems.append("FAIL: clean tree reported a violation")
        (root / "dirty.sh").write_text("uv run --with torch thing.py\n")
        if not scan(root):
            problems.append("FAIL: planted `uv run` was not caught -- the gate is decoration")
        (root / "dirty.sh").unlink()
        (root / "dirty2.md").write_text("run `uvx ruff` here\n")
        if not scan(root):
            problems.append("FAIL: planted `uvx` was not caught")
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
    hits = scan(pathlib.Path(args.root).resolve())
    for rel, n, line in hits:
        print(f"  {rel}:{n}: {line[:100]}")
    if hits:
        print(f"FAIL: {len(hits)} uv invocation(s); this repository uses pixi -- see CLAUDE.md")
        return 1
    print("ok   no uv invocations; environments come from pixi.toml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
