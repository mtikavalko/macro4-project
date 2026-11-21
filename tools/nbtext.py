import json
import sys

USAGE = "usage: python3 tools/nbtext.py NOTEBOOK [--outline] [CELL | FIRST-LAST]"


def output_text(output):
    kind = output["output_type"]
    if kind == "stream":
        return "".join(output["text"])
    if kind == "error":
        return output["ename"] + ": " + output["evalue"]
    data = output.get("data", {})
    parts = []
    if "text/plain" in data:
        parts.append("".join(data["text/plain"]))
    for mime in sorted(data):
        if mime.startswith("image/"):
            parts.append("[" + mime + " omitted]")
    return "\n".join(parts)


def first_line(cell):
    source = "".join(cell["source"]).strip()
    if not source:
        return ""
    return source.splitlines()[0][:100]


def main(args):
    outline = "--outline" in args
    if outline:
        args.remove("--outline")
    if not args:
        print(USAGE)
        sys.exit(2)

    nb = json.load(open(args[0], encoding="utf-8"))
    cells = nb["cells"]
    first = 0
    last = len(cells) - 1
    if len(args) > 1:
        if "-" in args[1]:
            lo, hi = args[1].split("-")
            first = int(lo)
            last = int(hi)
        else:
            first = int(args[1])
            last = first

    for i in range(first, min(last, len(cells) - 1) + 1):
        cell = cells[i]
        if outline:
            print(f"[{i}] {cell['cell_type']:8} {first_line(cell)}")
            continue
        print(f"--- [{i}] {cell['cell_type']}")
        print("".join(cell["source"]))
        if cell["cell_type"] != "code":
            continue
        for output in cell.get("outputs", []):
            text = output_text(output).rstrip()
            if text:
                print(">>> " + text.replace("\n", "\n    "))


if __name__ == "__main__":
    main(sys.argv[1:])
