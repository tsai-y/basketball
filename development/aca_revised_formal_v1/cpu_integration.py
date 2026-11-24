"""Bounded revised formal-entry acceptance: one train clip, no model forward/update."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile

import torch
from torch import nn

from . import formal_train as revised
from development.adjudicated_input_v1 import adapter as labels
from development.aca_eight_class_data_v1 import loop_core as core
from development.aca_eight_class_data_v1.gpu_readiness import equal
from development.aca_reconstruction_v2 import training_runtime as old_runtime

OUT = revised.INTEGRATION.parent


def rejects(fn):
    try:
        fn()
    except ValueError:
        return True
    raise AssertionError('Invalid identity was accepted')


def main():
    revised.need(not OUT.exists(), 'Integration output exists; inspect instead of overwrite')
    OUT.mkdir()
    torch.set_num_threads(4)
    parent = revised.parent_plan()
    plan = revised.data_plan(parent)
    protected = json.loads((labels.REVISION_DIR/'cpu_acceptance_v1/report.json').read_text())['protected_artifact_hashes']
    for p, h in protected.items():
        revised.need(labels.digest(Path(p).read_bytes()) == h, 'Historical pre-check mismatch')
    # This only starts loader workers, with exactly one pre-adjudicated train ID.
    dataset = revised.RevisedDataset(plan['source_root'], 'train', [labels.TARGET], plan['reference'])
    direct_dataset = labels.AdjudicatedDataset(source_root=plan['source_root'], split='train',
                                               video_ids=[labels.TARGET])
    revised.need(len(dataset) == len(direct_dataset) == 1, 'Unexpected loader scope')
    state = {'epoch': 0, 'cursor': 0}
    stream = revised.RevisedEpochStream({'train': dataset}, plan, lambda: state)
    try:
        batch = stream('train', 0)
        revised.need(stream.observed == 1 and stream.iterator is None, 'Unexpected worker output')
    finally:
        stream.close()
    ref = plan['reference'][labels.TARGET]
    revised.need(batch['targets'].tolist() == [1] and batch['binary_targets'].tolist() == [1]
                 and batch['records'][0]['original_class'] == '3p1'
                 and labels.tensor_sha(batch['tensor'][0]) == ref['tensor_sha256'],
                 'Real worker target/input mismatch')
    # Exercise the actual formal loop input path and checkpoint identity with a
    # tiny CPU parameter holder. No forward/backward/optimizer step is performed.
    counters = {'forward': 0}
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__(); self.weight = nn.Parameter(torch.zeros(1))
        def forward(self, x):
            counters['forward'] += 1
            raise AssertionError('No model inference permitted during integration')
    target_index = plan['train_ids'].index(labels.TARGET)
    def fetch(split, index):
        revised.need(split == 'train' and index == target_index, 'Unexpected formal fetch')
        return batch
    with tempfile.TemporaryDirectory() as temp:
        checkpoint = Path(temp)/'last.pt'
        loop = revised.RevisedTrainingLoop(Tiny(), plan, fetch, checkpoint, 'integration-fixture')
        x, y = loop._batch('train', target_index)
        revised.need(y.tolist() == [1] and x.device.type == 'cpu', 'Formal loop got old target')
        unchanged_numerical_loop = revised.RevisedTrainingLoop._run is old_runtime.TrainingLoop._run
        revised.need(unchanged_numerical_loop, 'Numerical loop changed')
        old_plan_rejected = rejects(lambda: revised.RevisedTrainingLoop(
            Tiny(), parent, fetch, checkpoint, 'old-data'))
        good_batch = deepcopy(batch)
        batch['label_revision_sha256'] = 'old-data'
        stale_identity_rejected = rejects(lambda: loop._batch('train', target_index))
        batch.clear(); batch.update(good_batch)
        payload = loop._payload()
        revised.need(payload['plan']['label_revision_sha256'] == labels.MANIFEST_SHA
                     and payload['label_revision_sha256'] == labels.MANIFEST_SHA,
                     'Checkpoint missing data identity')
        core.atomic_save(payload, checkpoint)
        resumed = revised.RevisedTrainingLoop(Tiny(), plan, fetch, checkpoint, 'integration-fixture')
        resumed.resume()
        revised.need(resumed.state == loop.state, 'New checkpoint resume state differs')
        # Same-shaped fixture weights with a historical plan must be rejected
        # before mutating any live model, optimizer, or RNG.
        bad = deepcopy(payload)
        bad['plan'] = parent
        bad['plan_sha256'] = core.plan_digest(parent)
        core.atomic_save(bad, checkpoint)
        rejected_loop = revised.RevisedTrainingLoop(Tiny(), plan, fetch, checkpoint, 'integration-fixture')
        before = core.capture_rng()
        wrong_checkpoint_rejected = rejects(rejected_loop.resume)
        revised.need(equal(before, core.capture_rng()) and not rejected_loop.started,
                     'Rejected checkpoint mutated execution state')
        revised.need(counters['forward'] == 0 and loop.state['updates'] == 0,
                     'Unexpected inference/update')
    for p, h in protected.items():
        revised.need(labels.digest(Path(p).read_bytes()) == h, 'Historical artifacts changed')
    report = {
        'status': 'passed', 'created_at_taiwan': revised.runtime.now(),
        'data_version': labels.VERSION, 'label_revision_sha256': labels.MANIFEST_SHA,
        'real_worker_decodes': 1, 'worker_count_setting': 2,
        'video_id': labels.TARGET, 'split': 'train', 'worker_batch_target': 1,
        'worker_binary_target': 1, 'original_class': '3p1', 'effective_class': '2p1',
        'worker_tensor_sha256': labels.tensor_sha(batch['tensor'][0]),
        'loop_batch_target': y.item(), 'old_plan_rejected': old_plan_rejected,
        'stale_identity_rejected': stale_identity_rejected,
        'wrong_checkpoint_rejected_before_mutation': wrong_checkpoint_rejected,
        'new_fixture_checkpoint_resumed': True, 'temporary_fixture_checkpoints_removed': True,
        'numerical_loop_same_function': unchanged_numerical_loop,
        'optimizer_updates': 0, 'model_forward_calls': counters['forward'],
        'validation_video_decodes': 0, 'test_video_decodes': 0,
        'historical_artifacts_unchanged': True,
        'code_sha256': {name: h for name, h in plan['code_sha256'].items()
                        if name.startswith(('development/adjudicated_input_v1/',
                                            'development/aca_revised_formal_v1/'))},
        'scope': 'CPU data/worker/formal-loop input/checkpoint wiring; no classifier result',
    }
    revised.runtime.atomic_json(revised.INTEGRATION, report)
    print(json.dumps({k: report[k] for k in ('status', 'real_worker_decodes', 'worker_batch_target',
                     'old_plan_rejected', 'stale_identity_rejected', 'optimizer_updates')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
