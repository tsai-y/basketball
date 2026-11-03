from copy import deepcopy
import unittest

from development.temporal_selection_v1.probe import select_samples


class ProbeGuardTests(unittest.TestCase):
    def fixture(self):
        rows=[{'sample_id':f'Y{i:03d}','video_id':f'v{i}.mp4','video_path':f'v{i}.mp4',
               'source_group':f'g{i}','original_class':'2p0','binary_label':'0','split':'train','excluded':'0'} for i in range(1,21)]
        return {'split':'train','samples':rows},{r['video_id']:deepcopy(r) for r in rows}

    def test_fixed20_canonical_train_only(self):
        m,a=self.fixture();r=select_samples(m,a)
        self.assertEqual(len(r),20);self.assertEqual(r[0]['sample_id'],'Y001')

    def test_nontrain_unknown_and_path_mismatch_fail_before_video_open(self):
        for field,value in [('split','test'),('excluded','1'),('video_id','unknown'),('video_path','wrong.mp4')]:
            m,a=self.fixture();m['samples'][0][field]=value
            with self.assertRaises(ValueError):select_samples(m,a)

    def test_duplicate_or_missing_source_rejected(self):
        m,a=self.fixture();m['samples'][1]['source_group']=m['samples'][0]['source_group']
        with self.assertRaises(ValueError):select_samples(m,a)
        m,a=self.fixture();m['samples'].pop()
        with self.assertRaises(ValueError):select_samples(m,a)


if __name__=='__main__':unittest.main()
