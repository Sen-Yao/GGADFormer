"""Internal result validator; never copied into the anonymous package."""

import json
import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--expected", type=int, required=True)
parser.add_argument("--execution-sha", required=True)
parser.add_argument("--execution-host", required=True)
parser.add_argument("paths", nargs="+")
args = parser.parse_args()
required = {
    "dataset",
    "seed",
    "final_epoch",
    "AUROC",
    "AUPRC",
    "runtime_seconds",
    "wandb_run_id",
    "execution_sha",
    "protocol_id",
    "execution_host",
}
paths = [Path(item) for item in args.paths]
if len(paths) != args.expected:
    raise SystemExit("expected {} result files, got {}".format(args.expected, len(paths)))
rows = []
for path in paths:
    row = json.loads(path.read_text(encoding="utf-8"))
    if set(row) != required:
        raise SystemExit("unexpected result fields in {}".format(path))
    if row["final_epoch"] <= 0 or not (0 <= row["AUROC"] <= 1) or not (0 <= row["AUPRC"] <= 1):
        raise SystemExit("invalid result values in {}".format(path))
    if row["execution_sha"] != args.execution_sha:
        raise SystemExit("execution SHA mismatch in {}".format(path))
    if row["execution_host"] != args.execution_host:
        raise SystemExit("execution host mismatch in {}".format(path))
    if row["protocol_id"] != "vecgad-package-v2-full-revalidation-019fbc58":
        raise SystemExit("protocol mismatch in {}".format(path))
    rows.append(row)
identity = {(row["dataset"], row["seed"]) for row in rows}
if len(identity) != args.expected:
    raise SystemExit("duplicate or missing dataset-seed identities")
print(json.dumps({"valid": len(rows), "datasets": sorted({row["dataset"] for row in rows})}, sort_keys=True))
