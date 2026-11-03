from collections import Counter
from copy import deepcopy
import unittest

from development.yolo_train_probe_v1.ball_review_pool_v3 import (
    select_sources,indices,check_pending_state,CLASSES,TRAIN,DEV,
)


class BallReviewPoolV3Tests(unittest.TestCase):
    def fixture(self):
        rows=[{'video_id':f'{c}/g{i:02d}.mp4','source_group':f'g{i:02d}',
               'original_class':c,'split':'train','excluded':'0'} for i in range(17) for c in CLASSES]
        return rows,{'g00':TRAIN,'g01':DEV}

    def test_balanced_new_groups_and_preserved_old_reservations(self):
        rows,old=self.fixture();before=deepcopy(old);selected,a=select_sources(rows,old)
        self.assertEqual(old,before);self.assertTrue(all(a[g]==v for g,v in old.items()))
        self.assertEqual(len(selected),16)
        for part in (TRAIN,DEV):
            self.assertEqual(Counter(r['original_class'] for r in selected if r['detector_partition']==part),Counter(CLASSES))
        tg={r['source_group'] for r in selected if r['detector_partition']==TRAIN}
        dg={r['source_group'] for r in selected if r['detector_partition']==DEV}
        self.assertEqual(len(tg),8);self.assertEqual(len(dg),5)
        self.assertTrue(tg.isdisjoint(dg));self.assertTrue((tg|dg).isdisjoint(old))

    def test_order_independent_selection_before_any_ball_predictions(self):
        rows,old=self.fixture()
        self.assertEqual(select_sources(rows,old),select_sources(list(reversed(rows)),old))

    def test_nontrain_excluded_duplicate_rejected(self):
        for field,value in [('split','validation'),('split','test'),('excluded','1')]:
            rows,old=self.fixture();rows[0][field]=value
            with self.assertRaises(ValueError):select_sources(rows,old)
        rows,old=self.fixture();rows.append(deepcopy(rows[0]))
        with self.assertRaises(ValueError):select_sources(rows,old)

    def test_insufficient_groups_or_class_fail_without_fallback(self):
        rows,old=self.fixture();rows=[r for r in rows if r['source_group']<'g10']
        with self.assertRaises(ValueError):select_sources(rows,old)
        rows,old=self.fixture();rows=[r for r in rows if r['original_class']!='mp1']
        with self.assertRaises(ValueError):select_sources(rows,old)

    def test_five_fixed_positions_not_model_dependent(self):
        self.assertEqual(indices(146),(14,43,73,102,131))
        self.assertEqual(len(set(indices(10))),5)
        for n in (9,True,146.):
            with self.assertRaises(ValueError):indices(n)

    def test_empty_prediction_is_not_negative_or_train_ready(self):
        r={'human_confirmed':False,'annotation_complete':False,'training_ready':False,
           'visible_ball_status':'unreviewed','no_ball_confirmed':False,'model_predictions_not_ground_truth':[]}
        check_pending_state(r)
        for field,value in [('training_ready',True),('human_confirmed',True),('annotation_complete',True),
                            ('no_ball_confirmed',True),('visible_ball_status','absent')]:
            bad=deepcopy(r);bad[field]=value
            with self.assertRaises(ValueError):check_pending_state(bad)


if __name__=='__main__':unittest.main()
