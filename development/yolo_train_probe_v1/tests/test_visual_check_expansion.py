from copy import deepcopy
import unittest
from development.yolo_train_probe_v1.visual_check_expansion_v1 import eligible_samples,count_matches


class VisualExpansionTests(unittest.TestCase):
    def base(self):
        return {'split':'train','samples':[{'sample_id':str(i),'video_id':str(i)+'.mp4',
            'source_group':g,'split':'train','excluded':'0'} for i,g in enumerate(('trained','development','extra'))]}
    def test_excludes_train_and_development_sources(self):
        records=[{'source_group':'trained','video_id':'different.mp4','partition':'train'},
                 {'source_group':'development','video_id':'different2.mp4','partition':'val'}]
        self.assertEqual([x['source_group'] for x in eligible_samples(self.base(),records)],['extra'])
    def test_nontrain_rejected_before_any_image_lookup(self):
        for split in ('validation','test'):
            a=self.base();a['samples'][0]['split']=split
            with self.subTest(split=split),self.assertRaisesRegex(ValueError,'Non-train'):eligible_samples(a,[])
    def test_duplicate_video_rejected(self):
        a=self.base();a['samples'][1]['video_id']=a['samples'][0]['video_id']
        with self.assertRaisesRegex(ValueError,'Duplicate'):eligible_samples(a,[])
    def test_prediction_presence_is_not_a_match(self):
        gt=[{'class_id':1,'xyxy':[0,0,10,10]}]
        pred=[{'class_id':1,'xyxy':[50,50,60,60],'confidence':.9}]
        result=count_matches(gt,pred)['ball']
        self.assertEqual(result,{'ground_truth':1,'predicted':1,'matched':0,'missed':1,'extra':1})
    def test_matching_is_class_specific_and_one_to_one(self):
        gt=[{'class_id':1,'xyxy':[0,0,10,10]}]
        pred=[{'class_id':0,'xyxy':[0,0,10,10],'confidence':.99},
              {'class_id':1,'xyxy':[0,0,10,10],'confidence':.9},
              {'class_id':1,'xyxy':[0,0,10,10],'confidence':.8}]
        result=count_matches(gt,pred)
        self.assertEqual(result['ball']['matched'],1);self.assertEqual(result['ball']['extra'],1)
        self.assertEqual(result['active_player']['matched'],0)


if __name__=='__main__':unittest.main()
