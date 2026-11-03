from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from development.yolo_train_probe_v1.ball_resubstitution_v2 import training_rows,summarize,render_page


class BallResubstitutionV2Tests(unittest.TestCase):
    def fixture(self):
        return [{'image_id':str(i),'partition':'train' if i<14 else 'val','source_split':'train',
                 'source_group':f'g{i}','video_id':f'v{i}','image_sha256':f'sha{i}',
                 'annotations':[{'class_id':0,'xyxy':[1,2,7,9]}]} for i in range(18)]

    def test_only14_train_rows_without_mutation(self):
        records=self.fixture();old=deepcopy(records);rows=training_rows(records)
        self.assertEqual(len(rows),14);self.assertTrue(all(r['partition']=='train' for r in rows))
        self.assertEqual(records,old)

    def test_missing_duplicate_or_nontrain_rejected(self):
        r=self.fixture();r[0]['source_split']='test'
        with self.assertRaises(ValueError):training_rows(r)
        r=self.fixture();r[1]['image_id']=r[0]['image_id']
        with self.assertRaises(ValueError):training_rows(r)
        with self.assertRaises(ValueError):training_rows(self.fixture()[:-1])

    def test_development_overlap_rejected(self):
        for field in ('source_group','video_id','image_sha256'):
            r=self.fixture();r[-1][field]=r[0][field]
            with self.assertRaises(ValueError):training_rows(r)

    def test_empty_or_non_ball_labels_rejected(self):
        for annotations in ([],[{'class_id':1,'xyxy':[1,2,7,9]}],[{'class_id':0,'xyxy':[2,2,1,9]}]):
            r=self.fixture();r[0]['annotations']=annotations
            with self.assertRaises(ValueError):training_rows(r)

    def test_summary_separates_inherited_and_new(self):
        rows=[{'image_id':str(i),'annotation_origin':'old' if i<11 else 'new',
               'matching':{'ground_truth':1,'predicted':1,'matched':1,'missed':0,'extra':0}} for i in range(14)]
        rows[-1]['matching']={'ground_truth':1,'predicted':1,'matched':0,'missed':1,'extra':1}
        s=summarize(rows);self.assertEqual(s['fixed_conf025_iou05']['matched'],13)
        self.assertEqual(s['by_annotation_origin']['old']['matched'],11)
        self.assertEqual(s['by_annotation_origin']['new']['matched'],2)
        self.assertEqual(s['images_with_missed_balls'],['13']);self.assertEqual(s['images_with_extra_boxes'],['13'])
        with self.assertRaises(ValueError):summarize(rows[:-1])

    def test_display_names_checkpoint_role_correctly(self):
        report={'images':[],'summary':{'fixed_conf025_iou05':{'matched':14,'ground_truth':14,'missed':0,'extra':0}}}
        with tempfile.TemporaryDirectory() as folder:
            render_page(folder,report,'last');page=(Path(folder)/'index.html').read_text()
            self.assertIn('右為新14張版的 last 權重',page);self.assertNotIn('右為新14張版best模型',page)
            with self.assertRaises(ValueError):render_page(folder,report,'unknown')


if __name__=='__main__':unittest.main()
