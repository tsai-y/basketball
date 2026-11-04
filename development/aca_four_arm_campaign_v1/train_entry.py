"""Explicit scheduling adapter; delegates all numerics and CLI to frozen train.py."""
import argparse
import json
import sys
from pathlib import Path
from .gpu_guard import assert_idle_gpu, load_policy


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--gpu-policy', required=True, type=Path)
    parser.add_argument('--gpu-policy-sha256', required=True)
    args, remaining = parser.parse_known_args()
    policy = load_policy(args.gpu_policy, args.gpu_policy_sha256)
    from development.aca_four_arm_v1 import train
    original_guard, original_argv = train.assert_idle_gpu, sys.argv
    # This explicit binding changes only the pre-launch GPU scheduling check.
    # start_training, initialized model, fit, optimizer, data, selection and pins stay original.
    train.assert_idle_gpu = lambda: assert_idle_gpu(policy)
    sys.argv = [original_argv[0], *remaining]
    print('SCHEDULING_POLICY: ' + json.dumps({'path': str(args.gpu_policy),
                                             'sha256': args.gpu_policy_sha256}), flush=True)
    try:
        train.main()
    finally:
        train.assert_idle_gpu = original_guard
        sys.argv = original_argv


if __name__ == '__main__':
    main()
