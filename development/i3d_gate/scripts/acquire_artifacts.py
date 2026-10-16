#!/usr/bin/env python3
"""Acquire Gate 1C artifacts from predeclared canonical URLs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
import urllib.error
import urllib.request

import yaml


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "development/i3d_gate/configs/i3d_gate_1c.yaml"


class RedirectRecorder(urllib.request.HTTPRedirectHandler):
    def __init__(self) -> None:
        super().__init__()
        self.chain: list[dict[str, Any]] = []

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self.chain.append({"status": code, "from": req.full_url, "to": newurl})
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def download(url: str) -> tuple[bytes, list[dict[str, Any]], str]:
    redirects = RedirectRecorder()
    opener = urllib.request.build_opener(redirects)
    request = urllib.request.Request(url, headers={"User-Agent": "Basketball51-I3D-Gate-1C/1"})
    with opener.open(request, timeout=120) as response:
        body = response.read()
        final_url = response.geturl()
    return body, redirects.chain, final_url


def entries(config: dict[str, Any]) -> list[dict[str, Any]]:
    candidate = config["candidate"]
    implementation = candidate["implementation"]
    official = candidate["official_upstream"]
    result = [{
        "role": "converted_checkpoint",
        "filename": candidate["converted_checkpoint"]["filename"],
        "url": candidate["converted_checkpoint"]["canonical_url"],
        "expected_bytes": candidate["converted_checkpoint"]["expected_bytes"],
        "expected_sha256": None,
        "source_commit": implementation["commit"],
        "license_notice": "MIT port; upstream Apache-2.0 notice chain retained; weight-specific grant unspecified",
        "subdirectory": "converted",
    }]
    for item in official["files"]:
        result.append({
            "role": "official_raw_checkpoint_component",
            "filename": item["filename"],
            "url": item["canonical_url"],
            "expected_bytes": item["expected_bytes"],
            "expected_sha256": None,
            "source_commit": official["commit"],
            "license_notice": official["license"],
            "subdirectory": "official_raw",
        })
    result.extend([
        {
            "role": "conversion_model_source", "filename": implementation["model_source_filename"],
            "url": implementation["model_source_url"], "expected_bytes": None,
            "expected_sha256": implementation["model_source_expected_sha256"],
            "source_commit": implementation["commit"], "license_notice": implementation["license"],
            "subdirectory": "source/hassony2",
        },
        {
            "role": "conversion_script_source", "filename": implementation["conversion_source_filename"],
            "url": implementation["conversion_source_url"], "expected_bytes": None,
            "expected_sha256": implementation["conversion_source_expected_sha256"],
            "source_commit": implementation["commit"], "license_notice": implementation["license"],
            "subdirectory": "source/hassony2",
        },
        {
            "role": "conversion_license", "filename": "LICENSE",
            "url": implementation["license_url"], "expected_bytes": None,
            "expected_sha256": implementation["license_expected_sha256"],
            "source_commit": implementation["commit"], "license_notice": implementation["license"],
            "subdirectory": "source/hassony2",
        },
        {
            "role": "official_model_source", "filename": "i3d.py",
            "url": official["model_source_url"], "expected_bytes": None,
            "expected_sha256": official["model_source_expected_sha256"],
            "source_commit": official["commit"], "license_notice": official["license"],
            "subdirectory": "source/google_deepmind",
        },
        {
            "role": "official_license", "filename": "LICENSE",
            "url": official["license_url"], "expected_bytes": None,
            "expected_sha256": official["license_expected_sha256"],
            "source_commit": official["commit"], "license_notice": official["license"],
            "subdirectory": "source/google_deepmind",
        },
    ])
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    artifact_root = ROOT / config["artifact_directory"]
    if artifact_root.exists():
        raise ValueError(f"Refusing to reuse existing artifact directory: {artifact_root}")
    records = []
    acquired_at = datetime.now(timezone.utc).isoformat()
    for spec in entries(config):
        body, redirect_chain, final_url = download(spec["url"])
        actual_sha256 = sha256_bytes(body)
        if spec["expected_bytes"] is not None and len(body) != spec["expected_bytes"]:
            raise ValueError(f"Byte count mismatch for {spec['filename']}: {len(body)}")
        if spec["expected_sha256"] is not None and actual_sha256 != spec["expected_sha256"]:
            raise ValueError(f"SHA-256 mismatch for {spec['filename']}: {actual_sha256}")
        destination = artifact_root / spec["subdirectory"] / spec["filename"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(body)
        records.append({
            "role": spec["role"],
            "canonical_url": spec["url"],
            "redirect_chain": redirect_chain,
            "final_url": final_url,
            "filename": str(destination.relative_to(artifact_root)),
            "expected_bytes": spec["expected_bytes"],
            "bytes": len(body),
            "sha256": actual_sha256,
            "publisher_sha256_available": False,
            "gate_1b_sha256_expectation_available": spec["expected_sha256"] is not None,
            "source_commit": spec["source_commit"],
            "license_notice": spec["license_notice"],
            "acquired_at_utc": acquired_at,
        })
    manifest = {
        "schema_version": "basketball.i3d-gate-1c.acquisition-manifest.v1",
        "gate_version": config["version"],
        "acquired_at_utc": acquired_at,
        "artifact_directory": str(artifact_root.relative_to(ROOT)),
        "files": records,
    }
    (artifact_root / "acquisition_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, TypeError, ValueError, urllib.error.URLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
