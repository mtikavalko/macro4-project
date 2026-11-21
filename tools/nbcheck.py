import argparse
import difflib
import os
import re
import sys
import time

import nbformat
from nbclient import NotebookClient

from nbtext import output_text

NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
DURATION = re.compile(r"\b\d+(?:\.\d+)?s\b")
FIGURE_SIZE = re.compile(r"<Figure size \d+x\d+")


def mask(text):
    text = DURATION.sub("<t>s", text)
    return FIGURE_SIZE.sub("<Figure size WxH", text)


def cell_texts(nb):
    texts = {}
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        parts = []
        for output in cell.get("outputs", []):
            parts.append(output_text(output))
        texts[i] = mask("\n".join(parts)).splitlines()
    return texts


def cell_errors(nb):
    errors = {}
    for i, cell in enumerate(nb["cells"]):
        for output in cell.get("outputs", []):
            if output["output_type"] == "error":
                errors[i] = output["ename"] + ": " + output["evalue"]
    return errors


def line_changed(old, new, rtol, atol):
    if NUMBER.sub("#", old) != NUMBER.sub("#", new):
        return True
    for a, b in zip(NUMBER.findall(old), NUMBER.findall(new)):
        x = float(a)
        y = float(b)
        if abs(x - y) > atol + rtol * abs(x):
            return True
    return False


def changed_lines(old, new, rtol, atol):
    if len(old) != len(new):
        return list(difflib.unified_diff(old, new, lineterm="", n=0))[2:]
    lines = []
    for a, b in zip(old, new):
        if line_changed(a, b, rtol, atol):
            lines.append("- " + a)
            lines.append("+ " + b)
    return lines


parser = argparse.ArgumentParser(
    description="Re-execute a notebook in memory and report outputs that changed.")
parser.add_argument("notebook")
parser.add_argument("--rtol", type=float, default=1e-6)
parser.add_argument("--atol", type=float, default=1e-5)
parser.add_argument("--timeout", type=int, default=900)
parser.add_argument("--max-lines", type=int, default=40)
parser.add_argument("--write", action="store_true",
                    help="overwrite the notebook with the re-executed outputs")
args = parser.parse_args()

nb = nbformat.read(args.notebook, as_version=4)
old = cell_texts(nb)

start = time.time()
workdir = os.path.dirname(os.path.abspath(args.notebook))
client = NotebookClient(nb, timeout=args.timeout, allow_errors=True,
                        resources={"metadata": {"path": workdir}})
client.execute()
elapsed = time.time() - start

new = cell_texts(nb)
errors = cell_errors(nb)
name = os.path.basename(args.notebook)
print(f"{name}: executed {len(new)} code cells in {elapsed:.1f}s")

changed = 0
for i in new:
    lines = changed_lines(old.get(i, []), new[i], args.rtol, args.atol)
    if not lines and i not in errors:
        continue
    changed += 1
    print()
    if i in errors:
        print(f"[{i}] ERROR {errors[i]}")
        continue
    print(f"[{i}] changed")
    for line in lines[:args.max_lines]:
        print("  " + line)
    if len(lines) > args.max_lines:
        print(f"  ... {len(lines) - args.max_lines} more")

print()
settings = f"durations and figure sizes masked; rtol={args.rtol:g} atol={args.atol:g}"
if changed:
    print(f"CHANGED: {changed} of {len(new)} code cells ({len(errors)} with errors). {settings}")
else:
    print(f"NO CHANGES: all {len(new)} code cells match. {settings}")

if args.write:
    nbformat.write(nb, args.notebook)
    print(f"wrote re-executed outputs to {args.notebook}")

sys.exit(1 if changed else 0)
