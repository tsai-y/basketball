"""One CPU-only train/validation pass, stopping at the first observed failure.

No models, checkpoint loading, test video access, source edits or frame caches.
Run as a module so spawn workers inherit the same isolated implementation.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import sys
import time
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import torch

from . import foundation as f

DATASETS = {}


def now():
    return datetime.now(ZoneInfo('Asia/Taipei')).isoformat()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def initialize(source_root):
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    for split in ('train', 'validation'):
        DATASETS[split] = f.FoundationDataset(source_root=source_root, split=split)
    f.need(not torch.cuda.is_initialized(), 'Unexpected CUDA initialization')


def inspect_one(job):
    split, index, expected_id = job
    try:
        dataset = DATASETS[split]
        item = dataset[index]
        f.need(item['video_id'] == expected_id, 'Job/loader ID mismatch')
        tensor = item.pop('tensor')
        meta = item['metadata']
        f.need(meta['decoded_frames'] == meta['frame_count'] and meta['source_unchanged_during_decode'],
               'Incomplete or changed source')
        f.need(tuple(meta['selected_indices']) == f.baseline.uniform_indices(meta['frame_count']),
               'Unexpected frame sampling')
        source_sha = f.baseline.guarded_source_sha256(dataset.scope, expected_id)
        f.need(not torch.cuda.is_initialized(), 'Unexpected CUDA initialization')
        return dict(status='passed', video_id=expected_id, split=split,
                    class_id=item['class_id'], binary_label=item['binary_label'],
                    original_class=item['original_class'], source_group=item['source_group'],
                    source_sha256=source_sha, tensor_sha256=hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
                    tensor_shape=list(tensor.shape), tensor_dtype=str(tensor.dtype),
                    finite=True, contiguous=True, metadata=meta)
    except Exception as error:
        return dict(status='failed', video_id=expected_id, split=split,
                    error_type=type(error).__name__, error=str(error))


def run(args):
    f.need(1 <= args.workers <= 4, 'Use one to four CPU workers')
    f.need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Explicitly disable CUDA visibility')
    output = Path(args.output).absolute()
    allowed_output = f.ROOT / 'development/aca_eight_class_data_v1/artifacts'
    f.need(output.is_relative_to(allowed_output), 'Output must be in isolated artifacts')
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    contract = f.load_contract()
    rows, serialized = f.audit_rows(contract)
    (output / 'audit_manifest.csv').write_bytes(serialized)
    jobs = []
    for split in ('train', 'validation'):
        scope = f.FoundationScope(source_root=args.source_root, split=split)
        jobs.extend((split, index, row['video_id']) for index, row in enumerate(scope.rows))
    expected = {job[2] for job in jobs}
    reviewed = json.loads((output.parent / 'representative16/records.json').read_text())
    golden = {r['video_id']: r for r in reviewed}
    f.need(len(jobs) == len(expected) == 8739 and set(golden) <= expected, 'Unexpected audit scope')
    source_names = [
        'development/aca_eight_class_data_v1/foundation.py',
        'development/aca_eight_class_data_v1/audit_foundation.py',
        'development/aca_eight_class_data_v1/test_foundation.py',
        'docs/research/specs/input_foundation_v1.json',
    ]
    code_sha = {name: f.baseline.file_sha256(f.ROOT / name) for name in source_names}
    code_sha.update(contract['immutable_dependencies_sha256'])
    audit_plan = dict(schema='input_foundation_full_audit_plan_v1', started_at_taiwan=now(),
                      input_contract_sha256=f.CONTRACT_SHA256, audit_manifest_sha256=contract['audit_manifest_sha256'],
                      expected={'train': 7192, 'validation': 1547}, workers=args.workers,
                      source_root=str(Path(args.source_root).absolute()), source_files_sha256=code_sha,
                      decode_passes_per_source=1, source_hash_reads_per_source=1,
                      content_policy='whole_clip_preserved_quality_flags_only', test_video_access=False,
                      no_training=True)
    save(output / 'plan.json', audit_plan)
    counts, seen, completed = Counter(), set(), []
    progress = dict(status='running', started_at_taiwan=audit_plan['started_at_taiwan'], completed=0,
                    total=len(jobs), by_split={}, stage='full_decode', test_videos_opened=0)
    save(output / 'state.json', progress)
    failure = None
    print(f'OUTPUT: {output}', flush=True)
    with (output / 'records.jsonl').open('x') as stream:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                                 initializer=initialize, initargs=(args.source_root,)) as pool:
            iterator = iter(jobs)
            pending = {pool.submit(inspect_one, next(iterator)) for _ in range(args.workers)}
            while pending and failure is None:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                batch = []
                for future in done:
                    try:
                        record = future.result()
                    except Exception as error:
                        record = dict(status='failed', error_type=type(error).__name__, error=str(error))
                    batch.append(record)
                # Inspect the whole finished batch before submitting more work.
                for record in batch:
                    if record['status'] == 'passed' and record['video_id'] in golden:
                        previous = golden[record['video_id']]
                        if any(record[k] != previous[k] for k in ('source_sha256', 'tensor_sha256')):
                            record = dict(record, status='failed', error='Fixed16 source/tensor parity mismatch')
                    stream.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')
                    if record['status'] != 'passed':
                        failure = record
                        continue
                    f.need(record['video_id'] not in seen, 'Repeated audit ID')
                    seen.add(record['video_id'])
                    counts[record['split']] += 1
                    completed.append(record)
                if failure:
                    for future in pending:
                        future.cancel()
                    break
                for _ in batch:
                    job = next(iterator, None)
                    if job is not None:
                        pending.add(pool.submit(inspect_one, job))
                if len(completed) // 50 > progress['completed'] // 50 or len(completed) == len(jobs):
                    stream.flush()
                    progress.update(updated_at_taiwan=now(), completed=len(completed), by_split=dict(counts),
                                    elapsed_seconds=round(time.monotonic() - start, 1),
                                    last_video=completed[-1]['video_id'])
                    save(output / 'state.json', progress)
                    print(f'PROGRESS {len(completed)}/{len(jobs)} train={counts["train"]} validation={counts["validation"]}', flush=True)
    if failure:
        progress.update(status='stopped_on_problem', updated_at_taiwan=now(), completed=len(completed), by_split=dict(counts))
        save(output / 'failure.json', dict(failure, stopped_at_taiwan=now(), completed=dict(counts),
                                         note='No new jobs submitted after observed failure; up to workers-1 in-flight jobs may finish without being counted. No retry or repair.'))
        save(output / 'state.json', progress)
        print('STOPPED_ON_PROBLEM', flush=True)
        return 1
    f.need(seen == expected and dict(counts) == audit_plan['expected'], 'Incomplete final coverage')
    for name, expected_hash in code_sha.items():
        f.need(f.baseline.file_sha256(f.ROOT / name) == expected_hash, 'Source implementation changed during audit')
    ordered = sorted(completed, key=lambda r: r['video_id'])
    identity = hashlib.sha256()
    for r in ordered:
        identity.update(f'{r["video_id"]}\t{r["source_sha256"]}\t{r["tensor_sha256"]}\n'.encode())
    summary = dict(schema='input_foundation_full_audit_v1', status='completed', completed_at_taiwan=now(),
                   elapsed_seconds=round(time.monotonic() - start, 1), counts=dict(counts), total=len(completed),
                   failures=0, full_decode_eof_checks_passed=len(completed), tensor_checks_passed=len(completed),
                   fixed16_source_and_tensor_parity_passed=len(golden),
                   decoded_source_frames=sum(r['metadata']['decoded_frames'] for r in completed),
                   frame_count_range=[min(r['metadata']['frame_count'] for r in completed), max(r['metadata']['frame_count'] for r in completed)],
                   fps_counts=dict(Counter(str(r['metadata']['fps']) for r in completed)),
                   per_split_class_counts={s: dict(Counter(r['original_class'] for r in completed if r['split'] == s)) for s in counts},
                   per_split_binary_counts={s: dict(Counter(str(r['binary_label']) for r in completed if r['split'] == s)) for s in counts},
                   dataset_source_and_tensor_sha256=identity.hexdigest(),
                   records_sha256=f.baseline.file_sha256(output / 'records.jsonl'),
                   source_files_sha256=code_sha, input_contract_sha256=f.CONTRACT_SHA256,
                   audit_manifest_sha256=contract['audit_manifest_sha256'],
                   runtime=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, opencv=cv2.__version__),
                   test_videos_opened=0, excluded_videos_opened=0, model_forwards=0, optimizer_updates=0,
                   CUDA_context_initialized=False, source_content_modified=False,
                   notes='Engineering coverage only. Whole clips preserved. No visual relabeling, content prevalence estimate, test score or author-exact reproduction claim.')
    save(output / 'summary.json', summary)
    progress.update(status='completed', completed=len(completed), by_split=dict(counts), updated_at_taiwan=now())
    save(output / 'state.json', progress)
    print(f'COMPLETED {len(completed)}; zero failures; fixed16 parity passed', flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError('Refusing to overwrite an existing audit directory')
    try:
        return run(args)
    except Exception as error:
        # A crash is visible as failure, never as a completed audit.
        output = Path(args.output)
        if output.is_dir() and not (output / 'summary.json').exists():
            save(output / 'failure.json', dict(status='stopped_on_problem', error_type=type(error).__name__,
                                               error=str(error), stopped_at_taiwan=now()))
        raise


if __name__ == '__main__':
    sys.exit(main())
