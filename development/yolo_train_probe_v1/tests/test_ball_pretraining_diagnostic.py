import unittest
from development.yolo_train_probe_v1.ball_pretraining_diagnostic_v1 import ball_class_id,map_ball_predictions,state_shape_compatibility


class BallPretrainingTests(unittest.TestCase):
    def test_semantic_class_not_same_numeric_id(self):
        self.assertEqual(ball_class_id({0:'person',32:'sports ball'},'sports ball'),32)
        self.assertEqual(ball_class_id({0:'active_player',1:'ball',2:'rim'},'ball'),1)
    def test_missing_ambiguous_class_rejected(self):
        for names in ({0:'person'},{1:'ball',2:'ball'}):
            with self.assertRaises(ValueError):ball_class_id(names,'ball')
    def test_remapping_preserves_source_class_and_geometry(self):
        p=map_ball_predictions([[1,2,8,9]],[32],[.4],32,'coco')[0]
        self.assertEqual(p['class_id'],1);self.assertEqual(p['source_class_id'],32)
        self.assertEqual(p['xyxy'],[1.,2.,8.,9.])
    def test_wrong_class_invalid_geometry_and_low_conf_rejected(self):
        for box,cls,score in [([1,2,8,9],0,.4),([1,2,8,9],32,.1),([0,0,321,5],32,.4)]:
            with self.subTest(box=box,cls=cls,score=score),self.assertRaises(ValueError):
                map_ball_predictions([box],[cls],[score],32,'coco')
    def test_partial_shape_mismatch_not_whole_head_reset(self):
        source={'conv':(64,3,3,3),'head.box':(64,64,1,1),'head.cls':(80,64,1,1)}
        target={**source,'head.cls':(3,64,1,1)}
        r=state_shape_compatibility(source,target)
        self.assertEqual(r['compatible_tensor_count'],2);self.assertEqual(r['target_tensor_count'],3)
        self.assertEqual(r['incompatible_tensors'][0]['key'],'head.cls')


if __name__=='__main__':unittest.main()
