"""CPU synthetic and fail-closed regression tests; no real training or test video."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from development.aca_net_v1 import train as base
from development.aca_net_v1.readiness import ToyDataset, toy_model
from . import engine, experiment as exp, train
from .preflight import synthetic_resume


class HighResTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base.configure_runtime(20261025, cuda=False)

    def plan(self):
        return {'config': asdict(exp.HighResConfig(epochs=2, patience=2, workers=0)),
                'train_count': 19, 'validation_count': 5, 'engineering_only_synthetic_toy': True}

    def test_config_and_budget(self):
        self.assertEqual(exp.config_from().epochs, 15)
        self.assertEqual(exp.config_from().patience, 5)
        for change in ({'variant':'aca_dense_candidate'}, {'micro_batch':2}, {'effective_batch':8}, {'lr':.001}, {'seed':1}):
            with self.assertRaises(ValueError): exp.HighResConfig(**change)

    def test_default_is_only_plan(self):
        with patch.object(exp,'build_plan',return_value={'mode':'plan_only'}), patch.object(train,'run_training',side_effect=AssertionError('train')):
            self.assertEqual(train.main([]), {'mode':'plan_only'})

    def test_prepare_never_calls_formal_entry(self):
        with patch.object(exp,'prepare',return_value={'formal_training_started':False}), patch.object(train,'run_training',side_effect=AssertionError('train')):
            self.assertFalse(train.main(['--prepare'])['formal_training_started'])

    def test_start_needs_both_confirmations_before_gpu(self):
        for accepted, confirmed in ((False,False),(True,False),(False,True)):
            with patch.object(exp,'gpu_idle',side_effect=AssertionError('GPU inspected')):
                with self.assertRaises(ValueError):exp.launch_prerequisites(accepted,confirmed)

    def test_no_implicit_resume_or_confirmation(self):
        for args in (['--resume','/tmp/old'], ['--confirm-formal-training'], ['--start-training']):
            with self.assertRaises(ValueError):train.main(args)

    def test_test_dataset_refused_before_path_access(self):
        with patch.object(exp,'ResolutionDataset',side_effect=AssertionError('path access')):
            with self.assertRaises(ValueError):exp.dataset('/forbidden','test')

    def test_only_highres_input_and_correct_split(self):
        plan={'input_shape_per_video':[3,50,192,256]}
        for tensor, split in ((torch.zeros(1,3,50,120,160),'train'), (torch.zeros(1,3,50,192,256),'test')):
            with self.assertRaises(ValueError):list(engine.checked_loader([{'tensor':tensor,'split':[split]}],plan,'train'))

    def test_resume_bitwise_same_including_last_partial_batch(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertTrue(synthetic_resume(Path(folder))['bitwise_equal'])

    def test_checkpoint_family_and_plan_rejected(self):
        plan=self.plan(); model=toy_model('cpu'); opt=torch.optim.Adam(model.parameters(),lr=1e-4)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'checkpoint.pt'
            engine.save_checkpoint(path,model,opt,base.initial_progress(),plan)
            original=torch.load(path,weights_only=True)
            for change in ({'schema':'aca_dense_candidate_resume_v1'}, {'kind':'formal_aca_candidate'}, {'engineering_only':False}, {'plan_sha256':'wrong'}):
                torch.save({**original,**change},path)
                with self.assertRaises(ValueError):engine.load_checkpoint(path,model,opt,plan)

    def test_invalid_progress_cannot_be_saved(self):
        plan=self.plan()
        for change in ({'cursor':1}, {'optimizer_steps':1}, {'bad_epochs':1}, {'finished':True}, {'best_loss':.4}, {'epoch':True}):
            progress=base.initial_progress();progress.update(change)
            with self.assertRaises(ValueError):engine.validate_progress(progress,plan)

    def test_strict_min_ce_and_early_stop_ties(self):
        cfg=exp.HighResConfig(epochs=5,patience=2,workers=0)
        plan={**self.plan(),'config':asdict(cfg)}
        progress=base.initial_progress()
        def metrics(n,loss):return {'count':n,'loss':loss}
        losses=(.5,.4,.4,.4)
        for epoch, loss in enumerate(losses,1):
            progress['history'].append({'epoch':epoch,'train':metrics(19,1.),'validation':metrics(5,loss),
                                        'selected_as_best':epoch<=2})
        progress.update(epoch=4,optimizer_steps=8,best_loss=.4,bad_epochs=2,finished=True)
        engine.validate_progress(progress,plan)
        self.assertEqual(engine.final_summary('/tmp',progress,cfg)['best_epoch'],2)
        bad=copy.deepcopy(progress);bad['history'][2]['selected_as_best']=True
        with self.assertRaises(ValueError):engine.validate_progress(bad,plan)

    def test_cost_probe_cannot_unlock_formal_start(self):
        with self.assertRaises(ValueError):exp.validate_readiness(exp.COST/'report.json',{})

    def test_bn_mode_freezes_new_and_old_bn(self):
        from development.aca_net_v1.baseline_preflight import freeze_batchnorm, training_mode
        model=torch.nn.Sequential(torch.nn.Linear(3,3),torch.nn.BatchNorm1d(3))
        freeze_batchnorm(model);training_mode(model)
        self.assertFalse(model[1].training)
        self.assertTrue(all(not p.requires_grad for p in model[1].parameters()))


if __name__=='__main__':unittest.main()
