"""One CPU model and one synthetic full-size forward; no video or optimizer."""
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from . import adapter as a
from . import model_adapter as m

EVIDENCE = {}


def state_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class ModelAdapterTests(unittest.TestCase):
    def test_fixed_seed_rejected_before_checkpoint_access(self):
        with patch.object(m, 'verified_checkpoint_bytes', side_effect=AssertionError('No weight read')):
            for seed in (0, True, 20261025):
                with self.assertRaises(ValueError): m.initialized_baseline8(seed)

    def test_bad_weight_bytes_and_missing_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'wrong.pt'; p.write_bytes(b'wrong')
            with self.assertRaises(ValueError):
                m.verified_checkpoint_bytes(p, {'pretrained_size_bytes':5, 'pretrained_sha256':'0'*64})
            with self.assertRaises(ValueError):
                m.verified_checkpoint_bytes(p, {'pretrained_size_bytes':6, 'pretrained_sha256':'0'*64})
            with self.assertRaises(ValueError):
                m.verified_checkpoint_bytes(Path(tmp)/'missing.pt', {})

    def test_pinned_dependency_change_rejected(self):
        original = Path.read_bytes
        def read(path):
            data = original(path)
            return data+b' ' if path == m.SPEC_PATH else data
        with patch.object(Path, 'read_bytes', read), self.assertRaisesRegex(ValueError, 'spec changed'):
            m.load_model_contract()

    def test_one_cpu_model_full_input_to_eight_class_loss(self):
        # Any accidental raw-video path or old binary initializer is a failure.
        with patch.object(a.f.baseline, 'decode_uniform50', side_effect=AssertionError('No videos')), \
             patch('development.aca_net_v1.weights.initialized_baseline', side_effect=AssertionError('No binary initializer')):
            model, initialization = m.initialized_baseline8()
            self.assertEqual(initialization['backbone_tensors_equal_to_source'],265)
            self.assertEqual(initialization['head_shape'],[8,2048])
            self.assertEqual(initialization['parameters'], {'total':23524424,'trainable':23471304,'frozen_bn_affine':53120})
            bn_parameters = {id(p) for module in model.modules()
                             if isinstance(module, nn.modules.batchnorm._BatchNorm)
                             for p in module.parameters()}
            self.assertEqual(sum(isinstance(x,nn.modules.batchnorm._BatchNorm) for x in model.modules()),53)
            for name, p in model.named_parameters():
                self.assertEqual(p.requires_grad, id(p) not in bn_parameters, name)
            self.assertTrue(all(isinstance(block.aca, nn.Identity)
                                for stage in ('layer1','layer2','layer3','layer4') for block in getattr(model,stage)))
            model.eval(); m.baseline_training_mode(model)
            self.assertTrue(model.training)
            self.assertTrue(all(not x.training for x in model.modules()
                                if isinstance(x,nn.modules.batchnorm._BatchNorm)))
            before = state_digest(model)
            # Bounded deterministic tensor with temporal variation, no image file input.
            tensor = torch.linspace(-1,1,50).view(1,50,1,1).expand(3,50,120,160).contiguous()
            item = {'tensor':tensor,'class_id':6,'binary_label':0,'original_class':'mp0',
                    'video_id':'synthetic_only','source_group':'synthetic','split':'train',
                    'metadata':{'input_contract_sha256':a.f.CONTRACT_SHA256}}
            batch = a.collate_eight_class([item])
            self.assertEqual(batch['targets'].tolist(),[6])
            with torch.no_grad():
                logits = model(batch['tensor'])  # The only full model forward in this suite.
                loss = a.eight_class_ce(logits, batch['targets'])
            self.assertEqual(tuple(logits.shape),(1,8))
            self.assertTrue(torch.isfinite(logits).all().item())
            oracle = torch.logsumexp(logits.double()[0],dim=0)-logits.double()[0,6]
            self.assertAlmostEqual(loss.item(),oracle.item(),places=5)
            metrics = a.EightClassMetrics();metrics.add(logits,batch['targets'])
            self.assertEqual(metrics.result()['count'],1)
            self.assertAlmostEqual(metrics.result()['ce'],oracle.item(),places=10)
            after = state_digest(model)
            self.assertEqual(before,after)
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            EVIDENCE.update(initialization=initialization,input_shape=list(batch['tensor'].shape),
                            output_shape=list(logits.shape),target_class_id=6,
                            ce_matches_independent_logsumexp=True,
                            state_sha256_before=before,state_sha256_after=after,
                            all_parameter_grads_none=True, bn_modules=53,
                            model_forward_count=1,model_instances=1,
                            optimizer_updates=0,real_video_decodes=0,
                            cuda_initialized=torch.cuda.is_initialized(),
                            synthetic_logits_not_accuracy_evidence=True)


if __name__ == '__main__':
    unittest.main()
