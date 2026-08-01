"""Internal result validator; never copied into the anonymous package."""

import json
import sys
from pathlib import Path


EXPECTED = 30
required = {"dataset", "seed", "final_epoch", "AUROC", "AUPRC", "runtime_seconds"}
paths = [Path(item) for item in sys.argv[1:]]
if len(paths) != EXPECTED:
    raise SystemExit("expected {} result files, got {}".format(EXPECTED, len(paths)))
rows = []
for path in paths:
    row = json.loads(path.read_text(encoding="utf-8"))
    if set(row) != required:
        raise SystemExit("unexpected result fields in {}".format(path))
    if row["final_epoch"] <= 0 or not (0 <= row["AUROC"] <= 1) or not (0 <= row["AUPRC"] <= 1):
        raise SystemExit("invalid result values in {}".format(path))
    rows.append(row)
identity = {(row["dataset"], row["seed"]) for row in rows}
if len(identity) != EXPECTED:
    raise SystemExit("duplicate or missing dataset-seed identities")
print(json.dumps({"valid": len(rows), "datasets": sorted({row["dataset"] for row in rows})}, sort_keys=True))
