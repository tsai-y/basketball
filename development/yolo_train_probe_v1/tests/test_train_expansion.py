from copy import deepcopy
import unittest
from development.yolo_train_probe_v1.train_expansion_v1 import check_partitions, training_args, new_training_epoch


def records():
    return [{'image_id': split, 'source_split': 'train', 'source_group': split,
             'partition': split, 'video_id': split+'.mp4', 'image_sha256': split,
             'annotations': [{'id':str(i), 'class_id':i, 'xyxy':[i*10,0,i*10+5,5]} for i in range(3)],
             'human_full_image_confirmation': {'whole_image_confirmed': True,
                 'accepted_counts': {'active_player':1, 'ball':1, 'rim':1}}} for split in ('train', 'val')]


class ExpansionTrainingTests(unittest.TestCase):
    def setUp(self): self.assignments={'train':'detector_train', 'val':'detector_dev_validation'}
    def test_valid_source_separation(self): check_partitions(records(), self.assignments)
    def test_reserved_source_cannot_train(self):
        r=records();r[0]['source_group']='val'
        with self.assertRaisesRegex(ValueError,'Reserved'):check_partitions(r,self.assignments)
    def test_overlap_rejected(self):
        for field in ('video_id','image_sha256'):
            r=records();r[0][field]=r[1][field]
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'overlap'):check_partitions(r,self.assignments)
    def test_original_nontrain_rejected(self):
        for split in ('validation','test'):
            r=records();r[0]['source_split']=split
            with self.subTest(split=split),self.assertRaisesRegex(ValueError,'classification train'):check_partitions(r,self.assignments)
    def test_AI_only_or_count_change_rejected(self):
        for alter in ('approval','count'):
            r=records()
            if alter=='approval':r[0]['human_full_image_confirmation']['whole_image_confirmed']=False
            else:r[0]['annotations'].pop()
            with self.subTest(alter=alter),self.assertRaises(ValueError):check_partitions(r,self.assignments)
    def test_same_train_val_config_not_used(self):
        args=training_args('/dataset','/run',100,False)
        self.assertEqual(args['patience'],20);self.assertEqual(args['device'],0)
        self.assertEqual(args['epochs'],100);self.assertFalse(args['amp'])
    def test_run_bounds_and_smoke(self):
        for epochs,smoke in [(101,False),(0,False),(True,False),(2,True)]:
            with self.subTest(epochs=epochs,smoke=smoke),self.assertRaises(ValueError):training_args('/d','/r',epochs,smoke)
        self.assertEqual(training_args('/d','/r',1,True)['patience'],0)

    def test_final_validation_never_counts_as_training_epoch(self):
        self.assertTrue(new_training_epoch(0,0,{0}))
        self.assertFalse(new_training_epoch(0,1,{0}))
        self.assertFalse(new_training_epoch(1,1,{0}))
        self.assertTrue(new_training_epoch(19,19,set(range(20))))
        self.assertFalse(new_training_epoch(20,20,set(range(20))))

    def test_bias_warmup_does_not_use_large_default_lr(self):
        args=training_args('/d','/r',100,False)
        self.assertEqual(args['warmup_bias_lr'],args['lr0'])


if __name__=='__main__':unittest.main()
