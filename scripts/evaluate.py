# scripts/evaluate.py
"""Run a Stage 3 evaluation on an existing, immutable Stage 2 inference run."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.paths import PathResolver, project_root
from data.manifests import load_records
from evaluation.protocols.uvdoc import inspect_uvdoc_assets
from evaluation.runner import evaluate_inference_run, load_evaluation_config


def main() -> int :
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--inference-run", required=False)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cpu", help="Use cuda for long UVDoc line evaluations when available")
    parser.add_argument("--official-source", default=None, help="Pinned external UVDoc Git checkout")
    parser.add_argument("--verify-parity-id", action="append", default=[])
    parser.add_argument("--run-matlab", action="store_true")
    parser.add_argument("--preflight-assets", action="store_true", help="Check UVDoc benchmark assets without scoring")
    args = parser.parse_args()
    root = project_root(args.root)
    protocols = load_evaluation_config(root)["protocols"]
    if (args.protocol not in protocols) :
        parser.error(f"Unknown protocol: {args.protocol}. Available: {', '.join(protocols)}")
    if (args.preflight_assets) :
        if (protocols[args.protocol]["dataset"] != "uvdoc_benchmark") :
            parser.error("Asset preflight currently supports UVDoc Benchmark only")
        records = load_records("uvdoc_benchmark", root)
        report = inspect_uvdoc_assets(records, PathResolver(root))
        totals = {key : sum(bool(row[key]) for row in report) for key in report[0]
                  if key.endswith("_available")}
        print(json.dumps({"expected" : len(records), "available" : totals,
                          "missing" : [r for r in report if not all(r[k] for k in totals)]}, indent=2))
        return 0 if all(x == len(records) for x in totals.values()) else 2
    if (not args.inference_run) :
        parser.error("--inference-run is required unless --preflight-assets is selected")
    stamp = datetime.now(timezone.utc).strftime("stage3_%Y%m%dT%H%M%SZ_") + uuid4().hex[:8]
    output = Path(args.output) if args.output else root / "runs/evaluation" / stamp
    result = evaluate_inference_run(root, args.inference_run, args.protocol, output,
                                    device=args.device, official_source=args.official_source,
                                    verify_parity_ids=args.verify_parity_id,
                                    run_matlab=args.run_matlab)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    failed = sum(n for k, n in result["status_counts"].items() if k not in ("OK",))
    return 0 if not failed else 2


if (__name__ == "__main__") :
    raise SystemExit(main())
