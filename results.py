"""Save a script's results as JSON in results/, for compare.py and the replication workflow."""
import json
import os


def save(script, results):
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", script + ".json"), "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.write("\n")
