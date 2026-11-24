"""CPU-only constructor/optimizer acceptance; no video decode or formal updates."""
from copy import deepcopy
import gc
import hashlib
from pathlib import Path
import tempfile
import torch
from . import formal_train as f


def signature(model):
    h=hashlib.sha256()
    for name,t in model.state_dict().items():
        h.update(name.encode());h.update(str((tuple(t.shape),t.dtype)).encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    torch.set_num_threads(4)
    f.need(not f.RUN.exists(),'Candidate run already exists')
    f.need(not (f.PREP/'cpu_acceptance.json').exists(),'Acceptance already exists; do not overwrite')
    plan=f.draft_plan();old=f.parent.build_plan()
    model,_=f.parent.initialized()
    reference=signature(model);rng=f.core.capture_rng()
    del model;gc.collect()
    model,init=f.initialized()
    f.need(reference==signature(model) and f.ORIGINAL.equal(rng,f.core.capture_rng()),'Initialization/RNG differ')
    with tempfile.TemporaryDirectory(prefix='aca_lr_cpu_') as tmp:
        def no_fetch(*args):
            raise AssertionError('CPU acceptance must not decode or fetch data')
        base=f.parent.TrainingLoop(model,old,no_fetch,Path(tmp)/'unused_parent.pt','cpu-parent')
        base_groups=deepcopy(base.optimizer.state_dict()['param_groups']);del base
        loop=f.TrainingLoop(model,plan,no_fetch,Path(tmp)/'unused_candidate.pt','cpu-candidate')
        groups=deepcopy(loop.optimizer.state_dict()['param_groups'])
        normalized=deepcopy(groups);normalized[0]['lr']=0.0001
        f.need(normalized==base_groups and groups[0]['lr']==f.LR and loop.optimizer.defaults['lr']==f.LR,
               'Actual Adam differs beyond LR')
        f.need(reference==signature(model) and f.ORIGINAL.equal(rng,f.core.capture_rng()),'Constructor changed model/RNG')
        payload=loop._payload()
        f.need(payload['optimizer']['param_groups']==groups and payload['plan_sha256']==f.core.plan_digest(plan),
               'Checkpoint would record wrong LR/plan')
        f.need(loop._run.__func__ is f.runtime.TrainingLoop._run and f.TrainingLoop.resume is f.runtime.TrainingLoop.resume,
               'Numerical loop/checkpoint implementation differs')
        f.need(not list(Path(tmp).iterdir()) and loop.state['updates']==0,'Unexpected checkpoint/update')
        del payload,loop
    bad=deepcopy(plan);bad['optimizer']['lr']=0.0001
    try:
        f.TrainingLoop(model,bad,lambda *a:None,Path('/tmp/unused_lr_reject.pt'),'reject')
    except (ValueError,AssertionError):
        pass
    else:
        raise AssertionError('Old LR accepted')
    # Separate scalar fixture proves actual Adam step uses new LR, without model training.
    p=torch.nn.Parameter(torch.tensor([1.],dtype=torch.float32))
    opt=torch.optim.Adam([p],lr=0.0001,betas=(.9,.999),eps=1e-8,weight_decay=0.,amsgrad=False,foreach=False,fused=False)
    f.set_fresh_learning_rate(opt,plan);p.grad=torch.ones_like(p);opt.step()
    expected=1.-f.LR/(1.+1e-8)
    f.need(abs(p.item()-expected)<1e-7,'Scalar Adam LR mismatch')
    try:
        f.set_fresh_learning_rate(opt,plan)
    except (ValueError,AssertionError):
        pass
    else:
        raise AssertionError('Post-update LR override accepted')
    f.PREP.mkdir(parents=True,exist_ok=True)
    result=dict(status='passed',draft_plan_sha256=f.core.plan_digest(plan),
        initial_model_buffers_sha256=reference,initial_parameters_buffers_rng_equal=True,
        actual_optimizer_lr=groups[0]['lr'],optimizer_other_fields_equal=True,
        same_numerical_loop=True,checkpoint_lr_verified=True,wrong_lr_rejected=True,
        post_update_override_rejected=True,scalar_fixture_updates=1,scalar_fixture_parameter=p.item(),
        video_decodes=0,formal_optimizer_updates=0,gpu_training_started=False,
        gpu_preflight_performed=False,initialization=init,
        code_sha256={k:v for k,v in plan['code_sha256'].items() if k.startswith(f.PREFIX)})
    f.runtime.atomic_json(f.PREP/'cpu_acceptance.json',result)
    final=f.build_plan();f.verify_plan(final,f.core.plan_digest(final))
    f.runtime.atomic_json(f.PREP/'plan.json',final)
    f.runtime.atomic_json(f.PREP/'review.json',dict(status='prepared_not_started',
        plan_sha256=f.core.plan_digest(final),run_directory=str(f.RUN),
        only_numerical_change='Adam lr 0.0001 -> 0.00005',
        gpu_preflight='Deferred to manual launch; unchanged original two GPU checks',
        automation='Not activated',test_run=False))
    print('PREPARED_NOT_STARTED',f.core.plan_digest(final),flush=True)

if __name__=='__main__':main()
