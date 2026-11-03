"""Explicit, durable one-shot preflight -> fresh training; default is read-only status.

The product heartbeat invokes --launch-once only after human authorization.
An exclusive persistent claim prevents retries/duplicate runs even after crashes.
No automatic resume, fallback architecture, parameter search or test interface.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from datetime import datetime, timezone

from development.aca_net_v1.data import ROOT, require
from development.aca_net_v1.train import atomic_json, canonical_hash
from . import experiment as exp

CONTROL = exp.FOLDER / "artifacts/automatic_launch"


def status(control=CONTROL):
    path = control / "state.json"
    if not path.exists():
        return {"status": "claimed_without_state" if (control / "claim.json").exists() else "not_launched"}
    value = json.loads(path.read_text())
    pid = value.get("pid")
    if pid and value["status"] in {"starting", "preflight", "training"}:
        try:
            # Include process start time to prevent a recycled PID being mistaken for this run.
            value["process_alive"] = process_start(pid) == value.get("process_start")
        except (FileNotFoundError, ProcessLookupError):
            value["process_alive"] = False
    return value


def process_start(pid):
    # Fields after the final ')' start at field 3; starttime is field 22.
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def write_state(control, **changes):
    path = control / "state.json"
    value = json.loads(path.read_text()) if path.exists() else {}
    value.update(changes)
    value["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(path, value)
    return value


def verify_preparation(preparation, source_root, config):
    report = json.loads((preparation / "report.json").read_text())
    saved = json.loads((preparation / "plan.json").read_text())
    require(report.get("status") == "cpu_preparation_passed_gpu_preflight_pending" and
            report.get("parameters") == exp.PARAMETERS and
            report.get("cuda_context_initialized") is False and report.get("test_videos_opened") == 0 and
            report.get("plan_sha256") == canonical_hash(saved), "Invalid CPU preparation")
    require(canonical_hash(exp.build_plan(source_root, config)) == canonical_hash(saved),
            "Code/config/data/runtime changed since CPU preparation; do not launch")


def launch(source_root, preparation, *, accepted, control=CONTROL):
    # No mutation if we already launched, or if the baseline/GPU/preparation is not ready.
    if (control / "claim.json").exists():
        return status(control)
    config = exp.config_from(exp.FOLDER / "config.json")
    exp.assert_launch_prerequisites(accepted)
    verify_preparation(preparation, source_root, config)
    control.mkdir(parents=True, exist_ok=True)
    try:
        with (control / "claim.json").open("x") as handle:
            json.dump({"source_root": str(source_root.resolve()), "preparation": str(preparation.resolve()),
                       "automatic_retries": False}, handle)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        return status(control)
    write_state(control, status="claimed", preparation=str(preparation.resolve()),
                source_root=str(source_root.resolve()), test_videos_opened=0)
    try:
        with (control / "console.log").open("ab", buffering=0) as log:
            child = subprocess.Popen([sys.executable, "-B", "-u", "-m", __package__ + ".after_baseline",
                                      "--worker", "--source-root", str(source_root.resolve()),
                                      "--preparation", str(preparation.resolve())],
                                     cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        # The worker owns state updates after spawn (avoid racing/overwriting its status).
        return {"status": "dispatched", "pid": child.pid, "control": str(control)}
    except Exception as error:
        write_state(control, status="failed", error=repr(error))
        raise


def worker(source_root, preparation, control=CONTROL):
    from . import preflight, train
    claim = json.loads((control / "claim.json").read_text())
    require(claim["source_root"] == str(source_root.resolve()) and
            claim["preparation"] == str(preparation.resolve()), "Claim arguments mismatch")
    # Another manually invoked worker cannot reuse the claim.
    with (control / "worker_claim").open("x") as handle:
        handle.write(str(os.getpid()))
    write_state(control, status="starting", pid=os.getpid(), process_start=process_start(os.getpid()))
    try:
        config = exp.config_from(exp.FOLDER / "config.json")
        exp.assert_launch_prerequisites(True)
        verify_preparation(preparation, source_root, config)
        write_state(control, status="preflight")
        report = preflight.run(source_root, config, accepted=True)
        report_path = Path(report["output"]) / "report.json"
        write_state(control, preflight_report=str(report_path))
        result = train.run_training(source_root, config, report_path, accepted=True,
                    on_created=lambda output: write_state(control, status="training", run=str(output)))
        require(result.get("status") == "completed", "Training stopped before successful completion; no automatic resume")
        write_state(control, status="completed", summary=result)
    except BaseException as error:
        write_state(control, status="failed", error=repr(error))
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--preparation", type=Path)
    parser.add_argument("--accept-candidate-architecture", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--launch-once", action="store_true")
    mode.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.launch_once or args.worker:
        require(args.source_root is not None and args.preparation is not None, "Source root and CPU preparation required")
        result = worker(args.source_root, args.preparation) if args.worker else launch(
            args.source_root, args.preparation, accepted=args.accept_candidate_architecture)
    else:
        result = status()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
