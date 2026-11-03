from copy import deepcopy
import unittest

from development.yolo_train_probe_v1.ball_compare_v2 import validate_inputs,PARAMETERS


class BallComparisonV2Tests(unittest.TestCase):
    def fixture(self):
        records=[];inputs=[]
        for i in range(4):
            r={'image_id':f'D{i}','partition':'val','source_group':f'dev{i}','video_id':f'dev{i}.mp4',
               'frame_index':i,'image_sha256':f'dsha{i}','annotations':[{'class_id':0,'xyxy':[1,2,8,9]}]}
            records.append(r);inputs.append({**{k:r[k] for k in ('source_group','video_id','frame_index','image_sha256')},
                'item_id':r['image_id'],'ground_truth':[{'class_id':1,'xyxy':[1,2,8,9]}]})
        records.append({'image_id':'T','partition':'train','source_group':'train','video_id':'train.mp4'})
        for i in range(30):inputs.append({'item_id':f'U{i}','source_group':f'u{i//5}',
            'video_id':f'u{i//5}.mp4','image_sha256':f'usha{i}','ground_truth':None})
        return {'parameters':deepcopy(PARAMETERS),'inputs':inputs},records

    def test_exact_same34_without_mutation(self):
        p,r=self.fixture();old=deepcopy(p)
        self.assertEqual(len(validate_inputs(p,r)),34);self.assertEqual(p,old)

    def test_threshold_and_input_count_not_tuned(self):
        p,r=self.fixture();p['parameters']['conf']=.1
        with self.assertRaises(ValueError):validate_inputs(p,r)
        p,r=self.fixture();p['inputs'].pop()
        with self.assertRaises(ValueError):validate_inputs(p,r)

    def test_gt_mapping_geometry_and_identity_must_match(self):
        for field,value in [('ground_truth',[{'class_id':0,'xyxy':[1,2,8,9]}]),
                            ('ground_truth',[{'class_id':1,'xyxy':[2,2,8,9]}]),('image_sha256','wrong')]:
            p,r=self.fixture();p['inputs'][0][field]=value
            with self.assertRaises(ValueError):validate_inputs(p,r)

    def test_unlabeled_train_development_or_video_overlap_rejected(self):
        for field,value in [('source_group','train'),('source_group','dev0'),('video_id','train.mp4')]:
            p,r=self.fixture();p['inputs'][4][field]=value
            with self.assertRaises(ValueError):validate_inputs(p,r)

    def test_duplicate_ids_and_pixels_rejected(self):
        for field in ('item_id','image_sha256'):
            p,r=self.fixture();p['inputs'][5][field]=p['inputs'][4][field]
            with self.assertRaises(ValueError):validate_inputs(p,r)


if __name__=='__main__':unittest.main()
