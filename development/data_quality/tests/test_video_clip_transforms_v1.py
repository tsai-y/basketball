"""Synthetic images + fixed metadata only; no dataset-video access."""
from copy import deepcopy
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'development/data_quality/scripts'))
import audit_visual_preprocessing_pilot_v1 as p
import verify_visual_preprocessing_pilot_v1 as v
from transforms.video_clip_transforms import ClipTransform, STRATEGIES, normalize_tensor, validate_config


class Transforms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = p.load_config()['transform']
        cls.cfg = {**cls.cfg, 'output_height': 24, 'output_width': 32}
        rng = np.random.default_rng(12)
        cls.frame = rng.integers(0, 256, (36,48,3), dtype=np.uint8)
        cls.clip = np.repeat(cls.frame[None], 64, axis=0)

    def test_shared_parameters_no_flicker(self):
        for name in STRATEGIES:
            output, meta = ClipTransform(self.cfg, name)(self.clip, seed=23)
            self.assertTrue(np.all(output == output[:, :1]))
            self.assertTrue(meta['transform']['shared_across_all_frames'])

    def test_same_seed_and_metadata(self):
        for name in STRATEGIES:
            t = ClipTransform(self.cfg, name)
            a, ma = t(self.clip, seed=42)
            b, mb = t(self.clip, seed=42)
            np.testing.assert_array_equal(a,b)
            self.assertEqual(ma,mb)

    def test_different_seed_variation_and_deterministic_controls(self):
        for name in STRATEGIES:
            t = ClipTransform(self.cfg, name)
            a, _ = t(self.clip, seed=42)
            b, _ = t(self.clip, seed=43)
            self.assertEqual(np.array_equal(a,b), name not in ('spatial', 'photometric'))

    def test_shape_dtype_range_rgb(self):
        clip = np.zeros_like(self.clip)
        clip[:,:,:,0] = 255
        a, m = ClipTransform(self.cfg)(clip, seed=0)
        self.assertEqual(a.shape,(3,64,24,32))
        self.assertEqual(a.dtype,np.float32)
        self.assertTrue(a.flags.c_contiguous)
        self.assertTrue(np.all(a[0] == 1) and np.all(a[1:] == 0))
        for name in STRATEGIES:
            a, _ = ClipTransform(self.cfg,name)(self.clip,seed=0)
            self.assertTrue(0 <= a.min() <= a.max() <= 1)

    def test_crop_bounds_and_matching_geometry(self):
        for seed in range(20):
            a, m = ClipTransform(self.cfg,'spatial')(self.clip,seed=seed)
            x,y,w,h = m['transform']['parameters']['crop_xywh']
            self.assertTrue(0 <= x < x+w <= 48 and 0 <= y < y+h <= 36)
            expected = cv2.resize(self.frame[y:y+h,x:x+w],(32,24),interpolation=cv2.INTER_AREA)/255
            np.testing.assert_allclose(a[:,0].transpose(1,2,0),expected,atol=1e-7)

    def test_flip_semantics_and_orientation(self):
        ref,_ = ClipTransform(self.cfg)(self.clip,seed=0)
        flip,meta = ClipTransform(self.cfg,'horizontal_flip')(self.clip,seed=0)
        np.testing.assert_array_equal(ref[:,:,:,::-1],flip)
        self.assertIn('handedness',meta['transform']['flip_semantics'])

    def test_normalization_separate(self):
        a,m = ClipTransform(self.cfg,'photometric')(self.clip,seed=11)
        original = a.copy()
        b,n = normalize_tensor(a,mode='mean_std',mean=[.5,.4,.3],std=[.5,.2,.1])
        np.testing.assert_array_equal(a,original)
        np.testing.assert_allclose(b,(a-np.array([.5,.4,.3],np.float32)[:,None,None,None])/
            np.array([.5,.2,.1],np.float32)[:,None,None,None])
        self.assertEqual(m['transform']['normalization'],'none')
        c,_ = normalize_tensor(a,mode='affine',scale=2,offset=-1)
        np.testing.assert_array_equal(c,2*a-1)
        with self.assertRaises(ValueError):
            normalize_tensor(a,mode='mean_std',mean=[0]*3,std=[0]*3)
        with self.assertRaises(ValueError):
            normalize_tensor(a,mode='identity',scale=2)

    def test_invalid_clips(self):
        t = ClipTransform(self.cfg)
        for bad in ([],self.clip[:0],self.clip[:63],self.clip.astype(np.float32),
                    self.clip[:,:,:,0],np.empty((64,0,48,3),np.uint8),
                    np.empty((64,36,48,4),np.uint8)):
            with self.assertRaises(ValueError): t(bad,seed=0)

    def test_config_validation(self):
        for key,bad in [('spatial_scale',[.2,1]),('brightness',[1.1,.9]),('contrast',[float('nan'),1]),
                        ('output_height',0),('output_width',True),('enhancement_gamma',2),('version','wrong')]:
            cfg = deepcopy(self.cfg); cfg[key] = bad
            with self.assertRaises(ValueError): validate_config(cfg)
        cfg = {**self.cfg,'vertical_flip':True}
        with self.assertRaises(ValueError): validate_config(cfg)
        cfg = p.load_config(); cfg['sample_size'] = 7193
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'bad.yaml'; f.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError): p.load_config(f)

    def test_heldout_augmentation_rejected(self):
        for split in ('validation','test'):
            for name in STRATEGIES[1:]:
                with self.assertRaises(ValueError):
                    ClipTransform(self.cfg,name)(self.clip,seed=0,split=split)

    def test_sampler_metadata_preserved(self):
        cfg=p.load_config(); records,_=p.build_manifest(cfg)
        rec=next(r for r in records if r['fallback'])
        _,m=ClipTransform(self.cfg)(self.clip,seed=1,sample_metadata=rec)
        self.assertEqual(m['sample'],rec)
        m['sample']['seed']=0
        self.assertNotEqual(m['sample'],rec)


class SafetyAndEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=p.load_config()
        cls.records,_=p.build_manifest(cls.cfg)

    def test_manifest_train_only_balanced_and_reproducible(self):
        actual,_=p.validate_manifest(p.MANIFEST,self.cfg)
        self.assertEqual(actual,self.records)
        self.assertEqual(len(actual),48)
        self.assertTrue(all(r['split']=='train' for r in actual))
        self.assertEqual(Counter(r['label'] for r in actual),{'Make':24,'Miss':24})
        self.assertEqual(Counter(r['fps_rational'] for r in actual),{'2997/100':24,'2997/125':24})

    def test_all_fallbacks_contiguous_minimum_overlap(self):
        summaries, segments,_=p.qa.load_sidecars(p.qa.DEFAULT_MANIFEST,p.qa.DEFAULT_SEGMENTS,p.qa.DEFAULT_SUMMARY)
        wanted={r['video_id'] for r in summaries.values() if r['has_clean_64']=='false'}
        self.assertEqual({r['video_id'] for r in self.records if r['fallback']},wanted)
        self.assertEqual(sum(r['candidate_overlap'] for r in self.records),20)
        for r in self.records:
            self.assertEqual(r['end_frame_exclusive']-r['start_frame'],64)
            overlap=p.qa.candidate_overlap(r['start_frame'],r['end_frame_exclusive'],segments.get(r['video_id'],[]))
            self.assertEqual(overlap,r['candidate_overlap'])
            self.assertEqual(r['epoch'],0)
            self.assertEqual(r['policy_version'],'quality_aware_sampler_v1')
            self.assertTrue(r['seed_provenance'])

    def test_heldout_and_nonpilot_rejected_before_filesystem(self):
        guard=p.PilotGuard(self.records)
        ids={r['video_id'] for r in self.records}
        rows=p.qa.read_csv(p.qa.DEFAULT_MANIFEST)
        probes=[next(r for r in rows if r['split']==split)['video_id'] for split in ('validation','test')]
        probes.append(next(r['video_id'] for r in rows if r['split']=='train' and r['video_id'] not in ids))
        with patch.object(p.scanner.Guard,'authorize',side_effect=AssertionError('Filesystem guard reached')):
            for vid in probes:
                with self.assertRaises(ValueError): guard.identity(vid)
                with self.assertRaises(ValueError): guard.open(vid,vid,'test')

    def test_decoder_exact_indices_and_rgb(self):
        from unittest.mock import MagicMock
        rec=next(r for r in self.records if r['start_frame'] > 0)
        guard=MagicMock()
        guard.identity.return_value={'unchanged':True}
        guard.open.return_value=99
        guard.allowed={rec['video_id']:{'video_path':rec['video_id']}}
        cap=MagicMock()
        frames=[]
        for index in range(rec['end_frame_exclusive']):
            frame=np.zeros((8,8,3),np.uint8); frame[:,:,2]=index
            frames.append((True,frame))
        cap.read.side_effect=frames
        num,den=map(int,rec['fps_rational'].split('/'))
        cap.get.side_effect=[index*1000*den/num for index in range(rec['start_frame'],rec['end_frame_exclusive'])]
        with patch.object(p.cv2,'VideoCapture',return_value=cap), patch.object(p.os,'close'):
            clip,meta=p.decode_clip(rec,guard)
        self.assertEqual(cap.read.call_count,rec['end_frame_exclusive'])
        np.testing.assert_array_equal(clip[:,0,0,0],np.arange(rec['start_frame'],rec['end_frame_exclusive']))
        self.assertTrue(np.all(clip[:,:,:,1:]==0))
        self.assertEqual(meta['retained_frames'],64)

    def test_forged_manifest_rejected(self):
        rows=p.qa.read_csv(p.qa.DEFAULT_MANIFEST)
        for split in ('validation','test'):
            held=next(r for r in rows if r['split']==split)
            forged=deepcopy(self.records)
            forged[0]['video_id']=held['video_id']
            forged[0]['split']='train'
            with tempfile.TemporaryDirectory() as tmp:
                f=Path(tmp)/'manifest.csv'; f.write_text(p.manifest_text(forged))
                with self.assertRaises(ValueError): p.validate_manifest(f,self.cfg)

    def test_trace_fail_closed(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'development/data_quality/artifacts') as tmp:
            f=Path(tmp)/'trace'
            f.write_text('openat(AT_FDCWD, "/unknown.mp4", O_RDONLY) = 3\n+++ exited with 0 +++\n')
            with self.assertRaises(ValueError): v.trace_evidence(f,[],False)
            f.write_text('+++ exited with 0 +++\n')
            with self.assertRaises(ValueError): v.trace_evidence(f,['missing'],True)

    def test_json_markdown_same_statistics(self):
        # Once published, verify exact regeneration from JSON, without decoding.
        report=v.REPORT.with_suffix('.json')
        if report.exists():
            data=json.loads(report.read_text())
            self.assertEqual(v.render_report(data),v.REPORT.with_suffix('.md').read_text())
        else:
            fixture={k:i+1 for i,k in enumerate(('videos','clips','retained_frames','decoder_read_frames',
                'prefix_frames','sources','fallback_clips','candidate_overlap','same_seed_checks'))}
            fixture.update(labels={'Make':24,'Miss':24},fps={'2997/100':24,'2997/125':24},
                label_fps={},categories={},config=self.cfg,diagnostics={},
                visual_review={'observations':[]},verification=dict(unique_video_paths=48,
                    validation_video_access=0,test_video_access=0,non_pilot_video_access=0,
                    video_write_attempts=0,sha256='fixture',trace='fixture'),
                validation={'summary':'fixture'},contact_sheets=[],artifact_directory='fixture')
            text=v.render_report(json.loads(json.dumps(fixture)))
            for key in ('videos','clips','retained_frames','decoder_read_frames','prefix_frames','sources'):
                match=__import__('re').search(r'\| '+key+r' \| (\d+) \|',text)
                self.assertEqual(int(match.group(1)),fixture[key])


if __name__=='__main__': unittest.main()
