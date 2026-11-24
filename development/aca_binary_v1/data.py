"""Keep adjudicated eight-class provenance; explicitly route binary targets."""
from development.aca_revised_formal_v1.formal_train import RevisedDataset
from development.adjudicated_input_v1 import adapter as labels
from .runtime import EpochStream as BaseStream
from .core import need

class BinaryCollator:
    def __init__(self):
        self.delegate = labels.AdjudicatedCollator()

    def __call__(self, items):
        batch = self.delegate(items)
        batch['eight_class_targets'] = batch['targets']
        batch['targets'] = batch['binary_targets'].clone()
        batch['task'] = 'basketball51_direct_binary'
        return batch

class EpochStream(BaseStream):
    def __init__(self, datasets, plan, state_getter):
        labels.require_revision_identity(plan)
        super().__init__(datasets, plan, state_getter, collate=BinaryCollator(), workers=2)


def guard_fetch(fetch, plan):
    def fetch_binary(split, index):
        batch = fetch(split, index)
        labels.require_revision_identity(batch)
        vid = plan[split+'_ids'][index];ref = plan['reference'][vid]
        need(batch.get('task') == plan['task'] and batch['targets'].tolist() == [ref['binary_label']]
             and batch['binary_targets'].tolist() == [ref['binary_label']]
             and batch['eight_class_targets'].tolist() == [ref['class_id']], 'Binary target/provenance mismatch')
        return batch
    return fetch_binary
