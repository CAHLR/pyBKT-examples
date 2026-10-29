"""Compare each script's results with the stored pyBKT output and the published results.

Run it from the repo root after the scripts. It prints a Markdown table. A
result that moved doesn't fail the run, because a pyBKT fix can move results on
purpose; update expected.json once the change is understood. The run fails
only when a script left no results file, which means it crashed.
"""
import json
import os
import sys


def flatten(value, prefix=""):
    """Turn nested results into ("rmse_bkt.Venn Diagram", 0.33) pairs."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from flatten(item, f"{prefix}.{key}" if prefix else str(key))
    else:
        yield prefix, value


def cell(value):
    return "" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)


def main():
    with open("expected.json") as f:
        expected = json.load(f)
    tolerance = expected["tolerance"]

    rows, moved, missing = [], 0, []
    for script, entry in expected["scripts"].items():
        path = os.path.join("results", script + ".json")
        if not os.path.exists(path):
            missing.append(script)
            continue
        with open(path) as f:
            result = dict(flatten(json.load(f)))
        stored = dict(flatten(entry["pybkt"]))
        published = dict(flatten(entry.get("published", {})))
        for key in sorted(set(result) | set(stored)):
            now, before, paper = result.get(key), stored.get(key), published.get(key)
            changed = now is None or before is None or abs(now - before) > tolerance
            moved += changed
            gap = "" if now is None or paper is None else cell(now - paper)
            rows.append(
                f"| {script} | {key} | {cell(now)} | {cell(before)} | "
                f"{'moved' if changed else ''} | {cell(paper)} | {gap} |"
            )

    print(f"{moved} of {len(rows)} results moved by more than {tolerance} from expected.json.\n")
    print("| Script | Result | Now | Stored | | Published | Gap to published |")
    print("|---|---|---|---|---|---|---|")
    print("\n".join(rows))
    if missing:
        print(f"\nNo results from: {', '.join(missing)}. Check that script's log.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
