"""Finite CPU tests for formal boundary/identity; no real decoding or GPU."""
import ast,copy,inspect,json,tempfile,textwrap,time,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from . import formal_train as formal, training_runtime as rt
from .engine import EngineeringLoop
from .test_engineering import Tiny
from development.aca_eight_class_data_v1 import loop_core as core
from development.aca_eight_class_data_v1.gpu_readiness import equal

class FormalBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    def plan(self):
        return {'schema':rt.PLAN_SCHEMA,'variant':'aca','test_enabled':False,'seed':20261024,
                'max_epochs':30,'patience':5,'effective':16,'checkpoint_every':25,
                'train_ids':[f't{i}' for i in range(7192)],'validation_ids':[f'v{i}' for i in range(1547)]}
    def test_formal_run_loop_and_restore_remain_tested_code(self):
        self.assertIs(rt.TrainingLoop._run,EngineeringLoop._run)
        def tree(f):return ast.dump(ast.parse(textwrap.dedent(inspect.getsource(f))),include_attributes=False)
        self.assertEqual(tree(rt.TrainingLoop.resume),tree(EngineeringLoop.resume))
        self.assertIs(rt.TrainingLoop._save,rt.base.TrainingLoop._save)
        self.assertIs(rt.TrainingLoop._batch,rt.base.TrainingLoop._batch)
    def test_resume_exact_and_reject_foreign_or_engineering_checkpoint(self):
        plan=self.plan();x=torch.randn(1,3,4,3,4,generator=torch.Generator().manual_seed(6))
        def fetch(split,index):
            return {'tensor':x,'targets':torch.tensor([index%8]),
                    'records':[{'split':split,'video_id':plan[split+'_ids'][index]}]}
        with tempfile.TemporaryDirectory() as d:
            def loop(name):
                torch.manual_seed(81);return rt.TrainingLoop(Tiny(),plan,fetch,Path(d)/name,'fixed')
            full=loop('full.pt');full.run(lambda p,s,n:s['updates']==2);expected=full._payload()
            first=loop('part.pt');first.run(lambda p,s,n:s['updates']==1)
            good=torch.load(first.checkpoint,weights_only=True)
            resumed=loop('part.pt');resumed.resume();resumed.run(lambda p,s,n:s['updates']==2)
            self.assertTrue(equal(expected,resumed._payload()))
            self.assertEqual(expected['model']['aca.channel.bn.num_batches_tracked'].item(),32)
            self.assertEqual(expected['model']['bn.num_batches_tracked'].item(),0)
            for patch_value in ({'schema':'aca_reconstruction_v2_engineering_only'},{'run_id':'other'}):
                bad={**good,**patch_value};core.atomic_save(bad,first.checkpoint)
                fresh=loop('part.pt');before=fresh._payload()
                with self.assertRaises(ValueError):fresh.resume()
                self.assertTrue(equal(before,fresh._payload()))
    def test_scope_rejects_other_seed_baseline_test_and_shortened_protocol(self):
        for change in ({'seed':20261025},{'variant':'baseline'},{'test_enabled':True},{'max_epochs':31},
                       {'effective':8},{'train_ids':['t']},{'schema':'aca_reconstruction_v2_engineering_only'}):
            with self.assertRaises(ValueError):rt.TrainingLoop(Tiny(),{**self.plan(),**change},None,'/tmp/unwritten.pt','x')
    def test_hash_and_plan_drift_fail_before_initialization(self):
        p={**self.plan(),'source_root':'unused'};h=core.plan_digest(p)
        with patch.object(formal,'build_plan',side_effect=AssertionError('should not build')):
            with self.assertRaises(ValueError):formal.verify_plan(p,'bad')
        with patch.object(formal,'build_plan',return_value={**p,'effective':8}):
            with self.assertRaises(ValueError):formal.verify_plan(p,h)
    def test_predecessor_order_is_required(self):
        with patch.object(formal,'completed_result') as completed,patch.object(formal,'build_plan',return_value={}) as build:
            formal.require_predecessor({'variant':'lsta','source_root':'x'});completed.assert_not_called()
            formal.require_predecessor({'variant':'tsci','source_root':'x'})
            self.assertEqual(completed.call_args.args[0],formal.run_path('lsta'))
            formal.require_predecessor({'variant':'aca','source_root':'x'})
            self.assertEqual(completed.call_args.args[0],formal.run_path('tsci'))
        with self.assertRaises(ValueError):formal.run_path('baseline')
    def test_completed_baseline_summary_and_history_are_consistent(self):
        p=formal.read(formal.BASELINE_RUN/'plan.json');r=formal.completed_result(formal.BASELINE_RUN,p)
        self.assertEqual(r['best_epoch'],7);self.assertEqual(r['plan_sha256'],formal.BASELINE_HASH)
        original=formal.read
        def changed(path):
            value=original(path)
            if Path(path).name=='summary.json':value['best_epoch']=8
            return value
        with patch.object(formal,'read',side_effect=changed):
            with self.assertRaises(ValueError):formal.completed_result(formal.BASELINE_RUN,p)

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--report',type=Path,required=True);args=ap.parse_args()
    if args.report.exists():raise ValueError('Do not overwrite evidence')
    start=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FormalBoundaryTests))
    args.report.write_text(json.dumps({'status':'passed' if result.wasSuccessful() else 'failed','tests':result.testsRun,
        'elapsed_seconds':time.monotonic()-start,'cuda_initialized':torch.cuda.is_initialized(),
        'real_video_decodes':0,'formal_training_started':False},indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
