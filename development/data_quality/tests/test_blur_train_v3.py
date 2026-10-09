"""No dataset video I/O: independent math, access guards, atomic commits and recovery."""
from contextlib import redirect_stdout
import copy
import csv
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_blur_train_v3 as s


class ScannerV3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = s.load_config(s.DEFAULT_CONFIG)
        cls.guard = s.Guard()
        cls.rows = s.choose_smoke(cls.guard, cls.cfg['seed'])
        cls.ctx = s.context(cls.cfg, s.DEFAULT_CONFIG, cls.guard, cls.rows[:2], 'smoke', cls.cfg['seed'])

    def fake_identity(self, vid):
        return dict(video_id=vid, resolved_path=str(s.p.gate.DATA/vid), size=123, mtime_ns=1,
            ctime_ns=1, inode=1, device=1, manifest_sha256=self.guard.scope['input_hashes']['manifest'])

    def fake_process(self, row, guard, cfg, ctx, attempt, failed=False):
        status = 'failed' if failed else 'ok'
        n = 1 if failed else 10
        summary = dict(video_id=row['video_id'], label=row['label'], source_group=row['source_group'],
            split='train', decode_status=status, config_hash=ctx['config_hash'], scanner_version=s.VERSION,
            expected_frame_count=10, processed_frame_count=n, fps_rational='25/1', candidate_frames=0,
            candidate_ratio=0., candidate_segments=[], processing_seconds=.1, longest_candidate_seconds=0.)
        s.atomic_json(attempt/'summary.json', summary)
        s.atomic_text(attempt/'frames.csv', s.csv_text([dict(video_id=row['video_id'], frame_index=i) for i in range(n)]))
        s.atomic_json(attempt/'completion.json', dict(video_id=row['video_id'],
            input_identity=self.fake_identity(row['video_id']), config_hash=ctx['config_hash'], scanner=ctx['scanner'],
            expected_frame_count=10, processed_frame_count=n, status=status, completion_timestamp=s.utc(),
            output_hashes={x:s.digest(attempt/x) for x in ('summary.json','frames.csv')}))
        if failed:
            s.append_event(attempt.parents[2]/'decode_failures.jsonl', video_id=row['video_id'], reason='simulated_decode_failure')
        return summary

    def test_frozen_evidence_and_config(self):
        self.assertEqual(self.cfg['minimum_seconds'], .16)
        self.assertEqual(self.cfg['z_threshold'], -2.)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'bad.yaml'
            path.write_text(s.DEFAULT_CONFIG.read_text().replace('z_threshold: -2.0', 'z_threshold: -2.5'))
            with self.assertRaisesRegex(ValueError, 'z_threshold'):
                s.load_config(path)

    def test_selection_reproducible_balanced_and_order_invariant(self):
        guard = s.Guard()
        guard.allowed = dict(reversed(list(guard.allowed.items())))
        self.assertEqual(self.rows, s.choose_smoke(guard, self.cfg['seed']))
        self.assertEqual(self.rows, s.p.read_csv(s.ROOT/self.cfg['smoke_manifest']))
        self.assertEqual(len(self.rows), 20)
        self.assertEqual(sum(r['label']=='Make' for r in self.rows), 10)
        self.assertEqual(len({r['source_group'] for r in self.rows}), 20)
        self.assertTrue({'2997/125','2997/100'} <= {r['fps_hint'] for r in self.rows})

    def test_all_heldout_and_wrong_paths_rejected_before_any_io(self):
        row = self.rows[0]
        with patch.object(s.p.gate, 'open_guarded', side_effect=AssertionError('No opener')), \
             patch.object(s.Guard, 'identity', side_effect=AssertionError('No stat')):
            for other in s.p.read_csv(s.p.gate.MANIFEST):
                if other['split']=='train':
                    continue
                for vid, path in ((other['video_id'], other['video_path']), (row['video_id'], other['video_path'])):
                    for op in ('probe','decode'):
                        with self.assertRaises(ValueError):
                            self.guard.open(vid, path, op)
            for vid, path in [('unknown', None), (row['video_id'], '/tmp/x.mp4'), (row['video_id'], '../x.mp4')]:
                with self.assertRaises(ValueError):
                    self.guard.open(vid, path, 'decode')
            for item in self.rows:
                self.assertEqual(self.guard.authorize(item['video_id'], item['video_path']), s.p.gate.DATA/item['video_path'])

    def test_positive_open_persists_attempt_and_open_events(self):
        from types import SimpleNamespace
        row = self.rows[0]
        with tempfile.TemporaryDirectory() as d:
            guard = s.Guard(Path(d)/'access.jsonl')
            info = SimpleNamespace(st_dev=1, st_ino=1, st_size=123, st_mtime_ns=1)
            with patch.object(guard, 'identity', side_effect=self.fake_identity), \
                 patch.object(s.p.gate, 'open_guarded', return_value=123), \
                 patch.object(s.os, 'fstat', return_value=info):
                self.assertEqual(guard.open(row['video_id'], row['video_path'], 'probe'), 123)
            events = [json.loads(line) for line in guard.ledger.read_text().splitlines()]
            self.assertEqual([e['phase'] for e in events], ['authorized_attempt','opened'])
            self.assertTrue(all(e['path']==str(s.p.gate.DATA/row['video_path']) for e in events))

    def test_tampered_split_and_symlink_rejected(self):
        guard = s.Guard(); row = self.rows[0]
        guard.allowed[row['video_id']] = dict(guard.allowed[row['video_id']], split='test')
        with self.assertRaises(ValueError):
            guard.authorize(row['video_id'])
        with patch.object(Path, 'is_symlink', return_value=True), patch.object(Path, 'stat', side_effect=AssertionError('No stat')):
            with self.assertRaises(ValueError):
                self.guard.identity(row['video_id'])

    def test_dry_run_never_touches_videos_or_output(self):
        with patch.object(s.Guard, 'identity', side_effect=AssertionError('No stat')), \
             patch.object(s.Guard, 'open', side_effect=AssertionError('No open')), \
             patch.object(s, 'context', side_effect=AssertionError('No ffprobe version either')), redirect_stdout(io.StringIO()):
            self.assertEqual(s.main(['--dry-run']), 0)
        for args in [['--scope','full-train'], ['--max-videos','21','--dry-run'],
                     ['--video-id','unknown','--dry-run'], ['--manifest','/tmp/other.csv','--dry-run']]:
            with self.assertRaises(ValueError):
                s.main(args)

    def test_local_z_independent_recalculation(self):
        scores = np.exp(np.random.default_rng(9).normal(size=(190,2)))
        for fps in (23.976, 29.97):
            z, _, _, quality = s.summarize_scores(scores, fps, self.cfg)
            width = int(math.ceil(3*fps)) | 1
            ref = []
            for i in range(len(scores)):
                values = np.log(np.maximum(scores[max(0,i-width//2):i+width//2+1], 1e-12))
                med = np.median(values, axis=0)
                scale = np.maximum(1.4826*np.median(abs(values-med), axis=0), .05)
                ref.append((np.log(scores[i])-med)/scale)
            np.testing.assert_allclose(z, ref, rtol=0, atol=1e-12)
            self.assertEqual(quality['rolling_window_frames'], width)

    def test_segment_min_duration_gap_and_tenengrad_independence(self):
        for fps, minimum in ((23.976,4),(29.97,5)):
            scores = np.full((180,2), 1000.)
            scores[20:20+minimum-1,0] = 1
            scores[60:60+minimum,0] = 1
            scores[100:100+minimum,0] = 1
            scores[101+minimum:101+2*minimum,0] = 1
            z, raw, keep, q = s.summarize_scores(scores, fps, self.cfg)
            self.assertFalse(keep[20:20+minimum-1].any())
            self.assertEqual(int(keep.sum()), 3*minimum)
            self.assertEqual(len(q['candidate_segments']), 3)  # never bridge one-frame gap
            scores[:,1] = np.random.default_rng(4).uniform(.01,10000,180)
            np.testing.assert_array_equal(keep, s.summarize_scores(scores, fps, self.cfg)[2])
            self.assertTrue(q['noncandidate_windows']['32']['exists'])

    def test_constant_low_is_diagnostic_not_candidate(self):
        _, _, keep, q = s.summarize_scores(np.zeros((146,2)), 23.976, self.cfg)
        self.assertFalse(keep.any())
        self.assertTrue(q['persistent_absolute_low_diagnostics']['laplacian_variance']['entire_video_below_reference'])

    def test_atomic_failure_leaves_no_final_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'result.json'
            with patch.object(s.os, 'rename', side_effect=OSError('Simulated crash')):
                with self.assertRaises(OSError): s.atomic_json(path, {'ok':1})
            self.assertFalse(path.exists())
            self.assertEqual(len(list(Path(d).glob('*.tmp-*'))), 1)
            s.atomic_json(path, {'ok':2})
            with self.assertRaises(ValueError): s.atomic_json(path, {'ok':3})
            self.assertEqual(s.read_json(path), {'ok':2})

    def test_resume_interrupt_corruption_retry_rebuild_and_config_mismatch(self):
        with tempfile.TemporaryDirectory() as d, \
             patch.object(s.Guard, 'identity', side_effect=self.fake_identity), redirect_stdout(io.StringIO()):
            out = Path(d)/'run'
            count = 0
            def interrupted(*args):
                nonlocal count
                count += 1
                if count == 2:
                    s.atomic_text(args[-1]/'partial.txt', 'interrupted')
                    raise KeyboardInterrupt()
                return self.fake_process(*args)
            with patch.object(s, 'process_video', side_effect=interrupted):
                self.assertEqual(s.run(out, self.ctx, self.cfg), 130)
            first = s.latest_attempt(out, self.rows[0]['video_id'])
            hashes = {f.name:s.digest(f) for f in first.iterdir()}
            with patch.object(s, 'process_video', side_effect=AssertionError('No retry without flag')):
                self.assertEqual(s.run(out, self.ctx, self.cfg, resume=True), 2)
            with patch.object(s, 'process_video', side_effect=self.fake_process) as scan:
                self.assertEqual(s.run(out, self.ctx, self.cfg, resume=True, retry=True), 0)
                self.assertEqual(scan.call_count, 1)
            self.assertEqual(hashes, {f.name:s.digest(f) for f in first.iterdir()})
            self.assertTrue((out/'errors.jsonl').is_file())
            expected = s.aggregate(out, self.ctx)
            with patch.object(s.Guard, 'identity', side_effect=AssertionError('No video stat on rebuild')):
                self.assertEqual(s.run(out, self.ctx, self.cfg, resume=True, rebuild=True), 0)
            self.assertEqual(expected, s.aggregate(out, self.ctx))
            altered = copy.deepcopy(self.ctx); altered['config_hash'] = 'different'
            with self.assertRaises(ValueError): s.run(out, altered, self.cfg, resume=True)
            # Corrupt an existing score file. Preserve it; never mark it complete or overwrite it.
            with (first/'frames.csv').open('a') as f: f.write('corrupt\n')
            bad_hash = s.digest(first/'frames.csv')
            with patch.object(s, 'process_video', side_effect=AssertionError('No automatic overwrite')):
                self.assertEqual(s.run(out, self.ctx, self.cfg, resume=True), 2)
            self.assertEqual(s.digest(first/'frames.csv'), bad_hash)
            with patch.object(s, 'process_video', side_effect=self.fake_process):
                self.assertEqual(s.run(out, self.ctx, self.cfg, resume=True, retry=True), 0)
            self.assertEqual(s.digest(first/'frames.csv'), bad_hash)
            with self.assertRaises(ValueError): s.run(out, self.ctx, self.cfg)

    def test_single_failure_continues_and_retry_ledger_preserved(self):
        with tempfile.TemporaryDirectory() as d, patch.object(s.Guard,'identity',side_effect=self.fake_identity), redirect_stdout(io.StringIO()):
            out = Path(d)/'run'
            def one_fails(*args):
                return self.fake_process(*args, failed=args[0]['video_id']==self.rows[0]['video_id'])
            with patch.object(s,'process_video',side_effect=one_fails):
                self.assertEqual(s.run(out,self.ctx,self.cfg),2)
            aggregate = s.aggregate(out,self.ctx)
            self.assertEqual((aggregate['ok'],aggregate['failed']), (1,1))
            original = (out/'decode_failures.jsonl').read_bytes()
            with patch.object(s,'process_video',side_effect=self.fake_process) as scan:
                self.assertEqual(s.run(out,self.ctx,self.cfg,resume=True,retry=True),0)
                self.assertEqual(scan.call_count,1)
            self.assertEqual((out/'decode_failures.jsonl').read_bytes(),original)

    def test_actual_process_short_decode_records_failure(self):
        class EmptyCapture:
            def isOpened(self): return True
            def read(self): return False, None
            def release(self): pass
        with tempfile.TemporaryDirectory() as d:
            out=Path(d); attempt=out/'videos'/'id'/'attempt-000001';attempt.mkdir(parents=True)
            rec=dict(fps=23.976,fps_rational='2997/125',r_frame_rate='2997/125',frame_count=146,
                duration_seconds=6.089423,resolution='320x240')
            with patch.object(s.Guard,'identity',side_effect=self.fake_identity), \
                 patch.object(s.p,'metadata',return_value=rec), patch.object(self.guard,'open',return_value=123), \
                 patch.object(s.cv2,'VideoCapture',return_value=EmptyCapture()), patch.object(s.os,'close'):
                summary=s.process_video(self.rows[0],self.guard,self.cfg,self.ctx,attempt)
            self.assertEqual(summary['decode_status'],'failed')
            self.assertIsNone(summary['candidate_frames'])
            self.assertEqual(s.inspect_attempt(attempt,self.rows[0],self.ctx)[0],'failed')
            self.assertEqual(len((out/'decode_failures.jsonl').read_text().splitlines()),1)

    def test_completion_identity_count_and_version_validation(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);row=self.rows[0]
            self.fake_process(row,self.guard,self.cfg,self.ctx,path)
            self.assertEqual(s.inspect_attempt(path,row,self.ctx,self.fake_identity(row['video_id']))[0],'ok')
            different=self.fake_identity(row['video_id']);different['size']=999
            self.assertEqual(s.inspect_attempt(path,row,self.ctx,different)[0],'suspect')
            record=s.read_json(path/'completion.json');record['processed_frame_count']=9
            (path/'completion.json').write_text(json.dumps(record))
            self.assertEqual(s.inspect_attempt(path,row,self.ctx)[0],'suspect')

    def test_existing_lock_prevents_second_writer(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'run'
            with s.locked_run(out,self.ctx,False):
                with self.assertRaises(BlockingIOError):
                    with s.locked_run(out,self.ctx,True): pass


if __name__ == '__main__':
    unittest.main(verbosity=2)
