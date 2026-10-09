#!/usr/bin/env python3
"""Verify v3 smoke/full-train derived outputs without opening any dataset video."""
import argparse
import json
from pathlib import Path
import sys

import audit_blur_train_v3 as s
import publish_blur_train_v3 as publisher


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", "--output-dir", dest="run_dir", type=Path,
                    default=s.ARTIFACTS / "blur_train_v3_smoke")
    ap.add_argument("--manifest", type=Path, default=s.p.gate.MANIFEST)
    ap.add_argument("--config", type=Path, default=s.DEFAULT_CONFIG)
    ap.add_argument("--scope", choices=("smoke", "full-train"), default="smoke")
    policy = ap.add_mutually_exclusive_group()
    policy.add_argument("--require-complete", action="store_true")
    policy.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--verify-artifacts", action="store_true",
                    help="Retained for CLI clarity; verifier always checks artifact hashes")
    ap.add_argument("--write-report", type=Path, help="Exclusive-create machine-readable verification report")
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    if args.scope == "full-train" and not args.allow_partial:
        args.require_complete = True
    result = publisher.verify_run(args.run_dir, args.manifest, args.config, args.scope, True)
    if args.write_report:
        publisher.write_exclusive(args.write_report,
            json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    compact = dict(run_status=result["run_status"], coverage=result["coverage"],
                   integrity={key: result["integrity"][key] for key in
                              ("expected_but_missing", "present_but_unexpected", "duplicate",
                               "terminal_failure", "incomplete", "corrupt", "held_out",
                               "artifact_hashes_verified", "aggregate_rebuilt_equal")},
                   video_opens=0, video_probes=0, video_decodes=0,
                   validation_video_access=0, test_video_access=0)
    print(json.dumps(compact, indent=2, sort_keys=True))
    if result["run_status"] == "INVALID":
        return 2
    if args.require_complete and result["run_status"] != "COMPLETE":
        return 2
    if not args.allow_partial and result["run_status"] != "COMPLETE":
        return 2
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, KeyError, TypeError, json.JSONDecodeError, publisher.PublicationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
