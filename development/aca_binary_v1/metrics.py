"""Direct two-class CE and confusion; no eight-class prediction mapping."""
import torch
from torch.nn import functional as F
from development.aca_eight_class_data_v1 import foundation as f
CLASSES = ("Miss", "Make")

def _validate_logits_targets(logits, targets):
    f.need(isinstance(logits, torch.Tensor) and logits.device.type == 'cpu' and
           logits.dtype in (torch.float32, torch.float64) and logits.ndim == 2 and
           logits.shape[1] == 2 and logits.shape[0] > 0, 'Need CPU float logits [N,2]')
    f.need(isinstance(targets, torch.Tensor) and targets.device.type == 'cpu' and
           targets.dtype == torch.long and targets.shape == (logits.shape[0],),
           'Need CPU int64 binary targets [N]')
    f.need(torch.isfinite(logits).all().item(), 'Nonfinite logits')
    f.need(((targets >= 0) & (targets < 2)).all().item(), 'Target outside two classes')


def binary_ce(logits, targets):
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


class BinaryMetrics:
    """Accumulate sample-weighted CE and confusion, without keeping graphs."""
    def __init__(self):
        self._matrix = [[0] * 2 for _ in range(2)]
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
        return {'task': 'basketball51_direct_binary', 'class_order': list(CLASSES),
                'ce': self._loss_sum / self._count, **_confusion_result(self._matrix),
                'macro_f1_rule': 'both_classes_zero_division_0',
                'prediction_method': 'direct_two_logit_argmax',
                'argmax_tie_rule': 'lowest_class_index_Miss'}
