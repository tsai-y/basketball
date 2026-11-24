"""Frozen eight-class input adapter and CPU metrics; no training entry point.

Binary predictions are mapped from eight-class argmax, NOT summed probabilities.
Macro-F1 always averages all eight classes, with zero for undefined class F1.
"""
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from . import foundation as f

CLASSES = ('2p0', '2p1', '3p0', '3p1', 'ft0', 'ft1', 'mp0', 'mp1')
CLASS_TO_BINARY = (0, 1, 0, 1, 0, 1, 0, 1)


class EightClassDataset(Dataset):
    """Delegate pixel loading unchanged; reject other splits before source access."""
    def __init__(self, *, source_root, split, video_ids=None):
        f.need(split in ('train', 'validation'), 'Only train/validation accepted')
        self.foundation = f.FoundationDataset(source_root=source_root, split=split,
                                              video_ids=video_ids)
        contract = self.foundation.scope.contract
        f.need(tuple(contract['classes']) == CLASSES and
               tuple(contract['class_to_binary']) == CLASS_TO_BINARY, 'Class order mismatch')

    def __len__(self):
        return len(self.foundation)

    def __getitem__(self, index):
        item = self.foundation[index]
        _validate_item(item)
        return item


def _validate_item(item):
    cls = item['class_id']
    f.need(type(cls) is int and 0 <= cls < 8, 'Invalid eight-class ID')
    f.need(item['original_class'] == CLASSES[cls], 'Original class / class_id mismatch')
    f.need(type(item['binary_label']) is int and
           item['binary_label'] == CLASS_TO_BINARY[cls], 'Binary mapping mismatch')
    f.need(item['split'] in ('train', 'validation'), 'Only train/validation accepted')
    f.need(item['metadata']['input_contract_sha256'] == f.CONTRACT_SHA256,
           'Input contract mismatch')
    x = item['tensor']
    f.need(isinstance(x, torch.Tensor) and x.device.type == 'cpu' and
           x.dtype == torch.float32 and tuple(x.shape) == (3, 50, 120, 160) and
           x.is_contiguous() and torch.isfinite(x).all().item(), 'Invalid input tensor')


def collate_eight_class(items):
    """Stack frozen pixels; keep provenance as records, targets from class_id only."""
    f.need(bool(items), 'Empty batch')
    for item in items:
        _validate_item(item)
    f.need(len({item['split'] for item in items}) == 1, 'Mixed splits in batch')
    f.need(len({item['video_id'] for item in items}) == len(items), 'Duplicate video in batch')
    return {
        'tensor': torch.stack([item['tensor'] for item in items]),
        'targets': torch.tensor([item['class_id'] for item in items], dtype=torch.long),
        'binary_targets': torch.tensor([item['binary_label'] for item in items], dtype=torch.long),
        'records': [{k: item[k] for k in ('video_id', 'original_class', 'source_group',
                                         'split', 'metadata')} for item in items],
        'input_contract_sha256': f.CONTRACT_SHA256,
    }


def _validate_logits_targets(logits, targets):
    f.need(isinstance(logits, torch.Tensor) and logits.device.type == 'cpu' and
           logits.dtype in (torch.float32, torch.float64) and logits.ndim == 2 and
           logits.shape[1] == 8 and logits.shape[0] > 0, 'Need CPU float logits [N,8]')
    f.need(isinstance(targets, torch.Tensor) and targets.device.type == 'cpu' and
           targets.dtype == torch.long and targets.shape == (logits.shape[0],),
           'Need CPU int64 eight-class targets [N]')
    f.need(torch.isfinite(logits).all().item(), 'Nonfinite logits')
    f.need(((targets >= 0) & (targets < 8)).all().item(), 'Target outside eight classes')


def eight_class_ce(logits, targets):
    """Mean unweighted CE, no smoothing; retains autograd, CPU-only in this version."""
    _validate_logits_targets(logits, targets)
    loss = F.cross_entropy(logits, targets, reduction='mean')
    f.need(torch.isfinite(loss).item(), 'Nonfinite CE')
    return loss


def _confusion_result(matrix):
    n = sum(map(sum, matrix))
    recalls, f1s = [], []
    for i in range(len(matrix)):
        true_count = sum(matrix[i])
        predicted_count = sum(row[i] for row in matrix)
        tp = matrix[i][i]
        recalls.append(tp / true_count if true_count else 0.)
        denominator = true_count + predicted_count
        f1s.append(2 * tp / denominator if denominator else 0.)
    return {'count': n, 'accuracy': sum(matrix[i][i] for i in range(len(matrix))) / n,
            'macro_f1': sum(f1s) / len(matrix), 'recall_per_class': recalls,
            'f1_per_class': f1s,
            'confusion_rows_true_columns_predicted': [row[:] for row in matrix]}


class EightClassMetrics:
    """Accumulate sample-weighted CE and confusion, without keeping graphs."""
    def __init__(self):
        self._matrix = [[0] * 8 for _ in range(8)]
        self._loss_sum = 0.
        self._count = 0

    def add(self, logits, targets):
        _validate_logits_targets(logits, targets)
        with torch.no_grad():
            value = F.cross_entropy(logits.detach().double(), targets, reduction='sum')
            f.need(torch.isfinite(value).item(), 'Nonfinite CE')
            predicted = logits.detach().argmax(dim=1).tolist()
            actual = targets.tolist()
        # Validation precedes mutation, so a rejected batch leaves metrics intact.
        for target, prediction in zip(actual, predicted):
            self._matrix[target][prediction] += 1
        self._loss_sum += value.item()
        self._count += len(actual)

    def result(self):
        f.need(self._count > 0, 'No samples; no metric result')
        binary = [[0, 0], [0, 0]]
        for true in range(8):
            for pred in range(8):
                binary[CLASS_TO_BINARY[true]][CLASS_TO_BINARY[pred]] += self._matrix[true][pred]
        return {
            'task': 'basketball51_eight_class', 'class_order': list(CLASSES),
            'ce': self._loss_sum / self._count, **_confusion_result(self._matrix),
            'macro_f1_rule': 'all_8_classes_zero_division_0',
            'argmax_tie_rule': 'lowest_class_index',
            'derived_binary': {
                'method': 'map_eight_class_argmax_not_probability_sum',
                'class_order': ['Miss', 'Make'], **_confusion_result(binary),
            },
        }
