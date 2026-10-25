"""Annotation safeguards, not detection accuracy or human-label tests."""

import copy
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import patch

from development.yolo_audit_v1 import target_label_pilot as pilot


def fixture():
    proposal = {'id': 'D1', 'xyxy': [10, 10, 30, 100], 'confidence': .7}
    return {'schema': 'yolo_target_label_review_v1', 'pilot_id': 'fixed',
            'protocol': pilot.PROTOCOL, 'classes': pilot.CLASS_NAMES,
            'images': [{'image_id': 'L001', 'image_sha256': 'pixels', 'status': 'pending',
                        'scene': 'unreviewed', 'annotations': [], 'proposals': [proposal],
                        'proposal_states': {'D1': 'pending'},
                        'class_visibility': {name: 'unreviewed' for name in pilot.CLASS_NAMES},
                        'annotation_complete': False, 'notes': ''}]}


def completed(base):
    review = copy.deepcopy(base)
    row = review['images'][0]
    row.update(status='reviewed', scene='court_play', annotation_complete=True,
               annotations=[{'id': 'A1', 'class_id': 0, 'xyxy': [10, 10, 30, 100],
                             'source_proposal_id': 'D1'}], proposal_states={'D1': 'accepted'},
               class_visibility={'active_player': 'present', 'ball': 'absent', 'rim': 'absent'})
    return review


class TargetLabelPilotTests(unittest.TestCase):
    def test_selection_balances_eight_classes_and_distinct_sources(self):
        rows = []
        for index, group in enumerate(('2p0','2p1','3p0','3p1','ft0','ft1','mp0','mp1')):
            for j in range(2):
                rows.append({'sample_id': f'Y{100+index*2+j}', 'video_id': f'{group}/{j}.mp4',
                             'split': 'train', 'excluded': '0', 'original_class': group,
                             'source_group': f'source{index}_{j}'})
        a = pilot.choose_samples({'split': 'train', 'samples': rows})
        b = pilot.choose_samples({'split': 'train', 'samples': list(reversed(rows))})
        self.assertEqual(a, b)
        self.assertEqual(len(a), 8)
        self.assertEqual(len({r['source_group'] for r in a}), 8)
        self.assertEqual(len({r['original_class'] for r in a}), 8)
        self.assertFalse({r['sample_id'] for r in a} & pilot.EXCLUDE)

    def test_nontrain_and_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            pilot.choose_samples({'split': 'test', 'samples': []})
        row = {'sample_id': 'same', 'split': 'train', 'excluded': '0'}
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            pilot.choose_samples({'split': 'train', 'samples': [row, row]})

    def test_person_predictions_are_not_ground_truth(self):
        base = fixture()
        result = pilot.validate_review(base, base)
        self.assertEqual(result['training_ready_images'], 0)
        self.assertFalse(result['yolo_training_labels_exported'])
        self.assertEqual(base['images'][0]['annotations'], [])

    def test_complete_human_review_can_be_ready_but_does_not_train(self):
        base = fixture()
        result = pilot.validate_review(base, completed(base))
        self.assertEqual(result['training_ready_ids'], ['L001'])
        self.assertFalse(result['training_started'])

    def test_uncertain_ball_prevents_incomplete_negative_label(self):
        base = fixture()
        review = completed(base)
        review['images'][0]['class_visibility']['ball'] = 'uncertain'
        with self.assertRaisesRegex(ValueError, 'uncertain'):
            pilot.validate_review(base, review)

    def test_claimed_visible_rim_needs_an_actual_box(self):
        base = fixture()
        review = completed(base)
        review['images'][0]['class_visibility']['rim'] = 'present'
        with self.assertRaisesRegex(ValueError, 'visibility/box'):
            pilot.validate_review(base, review)

    def test_accepted_proposal_needs_one_linked_player_box(self):
        base = fixture()
        review = completed(base)
        review['images'][0]['annotations'] = []
        with self.assertRaisesRegex(ValueError, 'Proposal/annotation'):
            pilot.validate_review(base, review)

    def test_excluded_image_requires_reason_and_is_not_negative_sample(self):
        base = fixture()
        review = copy.deepcopy(base)
        review['images'][0]['status'] = 'excluded'
        with self.assertRaisesRegex(ValueError, 'reason'):
            pilot.validate_review(base, review)
        review['images'][0]['notes'] = 'Transition, no unambiguous court-play objects'
        result = pilot.validate_review(base, review)
        self.assertEqual(result['training_ready_images'], 0)
        self.assertEqual(result['status_counts']['excluded'], 1)

    def test_closeup_cannot_be_declared_ready_for_this_protocol(self):
        base = fixture()
        review = completed(base)
        review['images'][0]['scene'] = 'closeup'
        with self.assertRaisesRegex(ValueError, 'court-play'):
            pilot.validate_review(base, review)

    def test_foreign_pilot_or_image_or_mapping_rejected(self):
        base = fixture()
        for key, value in [('pilot_id', 'wrong'), ('classes', ['person']), ('protocol', 'wrong')]:
            review = copy.deepcopy(base)
            review[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                pilot.validate_review(base, review)
        review = copy.deepcopy(base)
        review['images'][0]['image_sha256'] = 'wrong'
        with self.assertRaises(ValueError):
            pilot.validate_review(base, review)

    def test_invalid_bbox_rejected(self):
        for box in ([0,0,0,1], [-1,0,20,20], [0,0,321,240], [0,0,float('nan'),1], [0,0,True,1]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                pilot.validate_box(box)

    def test_review_check_cli_is_read_only(self):
        base = fixture()
        with patch.object(pilot, 'read_json', return_value=base), \
                patch.object(pilot, 'prepare', side_effect=AssertionError('Prepare called')), \
                patch.object(pilot, 'write_json', side_effect=AssertionError('Write called')), \
                redirect_stdout(io.StringIO()) as stream:
            pilot.main(['--check-review', 'review.json', '--pilot-dir', 'pilot'])
        self.assertIn('"training_ready_images": 0', stream.getvalue())


if __name__ == '__main__':
    unittest.main()
