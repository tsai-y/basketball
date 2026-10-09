"""No real videos: guard, source separation, numeric PSFs and temporal regressions."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_blur_pilot_v2 as v


class PilotV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=v.load_config(v.DEFAULT_CONFIG)
        cls.guard=v.p.TrainGuard()

    def test_source_disjoint_balanced_deterministic(self):
        selected,inv,_=v.choose(self.guard,self.cfg)
        self.assertEqual(len(selected),48);self.assertEqual(len(inv),35)
        self.assertEqual(selected,v.choose(self.guard,self.cfg)[0])
        self.guard.allowed=dict(reversed(list(self.guard.allowed.items())))
        self.assertEqual(selected,v.choose(self.guard,self.cfg)[0])
        groups=[{r['source_group'] for r in selected if r['subset']==s} for s in ('calibration','verification')]
        self.assertFalse(groups[0]&groups[1]);self.assertEqual([len(s) for s in groups],[6,6])
        self.assertIn('v184',groups[0]|groups[1])
        for g in groups[0]|groups[1]:
            rows=[r for r in selected if r['source_group']==g]
            self.assertEqual(sorted(r['binary_label'] for r in rows),['0','0','1','1'])

    def test_every_heldout_id_and_path_before_opener(self):
        train=next(iter(self.guard.allowed.values()))
        with patch.object(v.p.gate,'open_guarded',side_effect=AssertionError('Video access forbidden')):
            for r in v.p.read_csv(v.p.gate.MANIFEST):
                if r['split']=='train':continue
                for vid,path in ((r['video_id'],r['video_path']),(train['video_id'],r['video_path'])):
                    with self.assertRaises(ValueError):self.guard.open(vid,path,'decode')
            for vid,path in [('unknown',None),(train['video_id'],'/tmp/unknown.mp4'),(train['video_id'],'../x.mp4')]:
                with self.assertRaises(ValueError):self.guard.authorize(vid,path)
        self.assertEqual(self.guard.events,[])

    def test_identity_and_v1_score_agreement(self):
        bgr=np.random.default_rng(2).integers(0,256,(40,50,3),dtype=np.uint8)
        gray=v.cv2.cvtColor(bgr,v.cv2.COLOR_BGR2GRAY).astype(float)
        self.assertIs(v.transform(gray,'gaussian',0,None),gray)
        np.testing.assert_allclose(v.score(gray),v.p.score_frame(bgr,v.p.load_config(v.p.DEFAULT_CONFIG)),rtol=1e-12)
        self.assertEqual(v.paired_values(0,0,1e-12),(None,None))

    def test_psf_constant_dc_and_direction(self):
        for k in [3,5,9,13]:
            for a in [0,45,90,135]:
                psf=v.motion_kernel(k,a)
                self.assertAlmostEqual(psf.sum(),1)
                self.assertTrue((psf>=0).all())
                np.testing.assert_allclose(psf,psf[::-1,::-1],atol=1e-15)
                np.testing.assert_allclose(v.transform(np.full((31,31),73.),'motion',k,a),73,atol=1e-12)
        np.testing.assert_allclose(v.motion_kernel(5,0)[2],np.full(5,.2))
        np.testing.assert_allclose(v.motion_kernel(5,90)[:,2],np.full(5,.2))
        impulse=np.zeros((65,65));impulse[32,32]=1
        self.assertLess(v.score(v.transform(impulse,'gaussian',1,None))[0],v.score(impulse)[0])

    def test_positions_rounding_and_spans(self):
        self.assertEqual(v.frame_indices(146,[.1,.3,.5,.7,.9]),[15,44,73,102,131])
        for fps in [23.976,29.97]:
            for duration in [.25,.5,1]:
                lo,hi=v.injection_span(146,fps,duration,.3)
                self.assertGreaterEqual((hi-lo)/fps,duration)
                self.assertLess((hi-lo)/fps,duration+1/fps)
        with self.assertRaises(ValueError):v.injection_span(2,30,1,.5)

    def test_rolling_mad_correct_same_window_center_and_no_oracle(self):
        values=np.array([[1.,2.],[2,4],[4,8],[8,16],[16,32]])
        cfg=dict(self.cfg,temporal=dict(self.cfg['temporal'],rolling_seconds=3))
        z,med,mad,w=v.local_normalize(values,1,cfg)
        self.assertEqual(w,3)
        for i in range(5):
            window=np.log(values[max(0,i-1):i+2]);m=np.median(window,axis=0)
            np.testing.assert_allclose(med[i],m)
            np.testing.assert_allclose(mad[i],np.median(abs(window-m),axis=0))
        base=np.full((150,2),100.);base[60:72]=10
        z,_,_,_=v.local_normalize(base,25,self.cfg)
        self.assertTrue((z[60:72]<-6).all());self.assertTrue((z[:40]==0).all())
        z,_,_,_=v.local_normalize(np.zeros((5,2)),25,self.cfg)
        self.assertTrue(np.isfinite(z).all());self.assertTrue((z==0).all())

    def test_failure_not_replaced(self):
        class EmptyCapture:
            def isOpened(self):return True
            def read(self):return False,None
            def release(self):pass
        row=v.choose(self.guard,self.cfg)[0][0]
        with patch.object(self.guard,'open',return_value=123),patch.object(v.cv2,'VideoCapture',return_value=EmptyCapture()),patch.object(v.os,'close'):
            frames,pts,status=v.decode(row,dict(frame_count=146,fps=23.976),self.guard)
        self.assertEqual(frames,[]);self.assertEqual(status['decode_status'],'frame_count_mismatch')

    def test_fit_rejects_verification_before_scores(self):
        import analyze_blur_pilot_v2 as a
        with self.assertRaises(ValueError):
            a.fit_temporal([({'subset':'verification'},None)],self.cfg)

    def test_detection_iou_and_run_filter(self):
        import analyze_blur_pilot_v2 as a
        z=np.zeros((20,2));z[5:10]=-5;z[15]=-5
        mask=a.detect(z,'both',-3,.1,25)
        self.assertEqual(np.flatnonzero(mask).tolist(),list(range(5,10)))
        stat=a.metrics(mask,5,10,25,.5)
        self.assertEqual(stat['f1'],1);self.assertEqual(stat['best_segment_iou'],1)
        self.assertEqual(stat['onset_error_seconds'],0)
        self.assertEqual(stat['false_positive_segments'],0)
        z[5:10,1]=0
        self.assertFalse(a.detect(z,'both',-3,.1,25).any())
        self.assertEqual(int(a.detect(z,'either',-3,.1,25).sum()),5)

    def test_source_macro_equal_weights(self):
        import analyze_blur_pilot_v2 as a
        import pandas as pd
        df=pd.DataFrame([dict(source_group='a',video_id='x',f1=1.)]*20+
                        [dict(source_group='b',video_id='y',f1=0.)])
        self.assertEqual(a.macro(df,'f1'),.5)


if __name__=='__main__':unittest.main(verbosity=2)
