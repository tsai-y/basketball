"""Two-class metric/loss boundaries; preserve original RNG and atomic checkpoints."""
import copy
import math
import torch
from torch.nn import functional as F
from development.aca_eight_class_data_v1.loop_core import (need, capture_rng, restore_rng, finite_tree, atomic_save, plan_digest, FixtureLoop as OriginalFixtureLoop)
from .metrics import CLASSES, BinaryMetrics
from development.aca_reconstruction_v2.model import training_mode

def metrics_state(metric):
    return {'schema': 'direct_binary_metrics_v1', 'classes': list(CLASSES),
            'matrix': copy.deepcopy(metric._matrix), 'loss_sum': metric._loss_sum,
            'count': metric._count}

def restore_metrics(state):
    need(state['schema'] == 'direct_binary_metrics_v1' and state['classes'] == list(CLASSES),
         'Metric schema/classes mismatch')
    matrix, count, loss = state['matrix'], state['count'], state['loss_sum']
    need(type(count) is int and count >= 0 and len(matrix) == 2 and
         all(len(row) == 2 and all(type(v) is int and v >= 0 for v in row) for row in matrix),
         'Invalid confusion/count')
    need(sum(map(sum, matrix)) == count and type(loss) in (float, int) and
         math.isfinite(loss) and loss >= 0 and (count > 0 or loss == 0), 'Invalid metric totals')
    metric = BinaryMetrics()
    metric._matrix, metric._count, metric._loss_sum = copy.deepcopy(matrix), count, float(loss)
    return metric

def summed_ce(logits, targets):
    """Device-local FP32 binary loss; metrics remain detached on CPU."""
    need(logits.dtype == torch.float32 and logits.ndim == 2 and logits.shape[1] == 2 and
         targets.dtype == torch.long and targets.shape == (len(logits),) and
         targets.device == logits.device and len(logits) > 0, 'Invalid logits/targets')
    need(torch.isfinite(logits).all().item() and ((targets >= 0) & (targets < 2)).all().item(),
         'Nonfinite logits or invalid targets')
    value = F.cross_entropy(logits, targets, reduction='sum')
    need(torch.isfinite(value).item(), 'Nonfinite loss')
    return value

class FixtureLoop(OriginalFixtureLoop):
    def _batch(self, split, index):
            batch = self.fetch(split, index)
            records, x, y = batch['records'], batch['tensor'], batch['targets']
            need(len(records) == 1 and records[0]['split'] == split and
                 records[0]['video_id'] == self.plan[split + '_ids'][index], 'Data order/split mismatch')
            need(x.device.type == 'cpu' and x.dtype == torch.float32 and x.shape[0] == 1 and
                 torch.isfinite(x).all().item() and y.device.type == 'cpu' and
                 y.dtype == torch.long and y.shape == (1,) and 0 <= y.item() < 2, 'Invalid microbatch')
            return x, y
