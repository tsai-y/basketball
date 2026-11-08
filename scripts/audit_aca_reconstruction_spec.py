"""Read-only, standard-library audit of the frozen ACA reconstruction spec.

No torch imports, models, datasets, GPU calls, checkpoints, or training entrypoints.
Run: python3 scripts/audit_aca_reconstruction_spec.py
"""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = 'docs/research/specs/aca_dense_reference_v1.json'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(relative):
    return json.loads((ROOT / relative).read_text())


def audit():
    spec = read_json(SPEC)
    require(spec['spec_id'] == 'aca_dense_reference_v1', 'Unexpected specification')
    require(spec['exact_author_reproduction_confirmed'] is False and
            spec['executable_training_config'] is False and
            spec['training_authorized_by_this_spec'] is False, 'Claim or scope changed')
    for path, expected in spec['source_sha256'].items():
        require(file_sha256(ROOT / path) == expected, f'Pinned source changed: {path}')

    tree = ast.parse((ROOT / 'development/aca_net_v1/modules.py').read_text())
    config_class = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                        and n.name == 'ReconstructionConfig')
    defaults = {n.target.id: ast.literal_eval(n.value) for n in config_class.body
                if isinstance(n, ast.AnnAssign)}
    require(defaults == spec['configuration'], 'Dataclass defaults disagree with specification')
    experiment = ast.parse((ROOT / 'development/aca_four_arm_v1/experiment.py').read_text())
    parameter_constant = next(ast.literal_eval(n.value) if isinstance(n.value, ast.Dict)
                              else {k.arg:ast.literal_eval(k.value) for k in n.value.keywords}
                              for n in experiment.body if isinstance(n, ast.Assign)
                              and any(isinstance(t, ast.Name) and t.id == 'PARAMETERS' for t in n.targets))

    stages = []
    sites = []
    incoming = 64
    height, width = 30, 40
    for stage, (channels, blocks) in enumerate(zip((64,128,256,512), (3,4,6,3)), 1):
        backbone = affine = 0
        for block in range(blocks):
            stride = 2 if stage > 1 and block == 0 else 1
            projection = incoming != channels * 4 or stride != 1
            # Count every conv weight and BN gamma/beta independently.
            main = incoming*channels + channels*channels*9 + channels*(4*channels)
            block_affine = 2*channels + 2*channels + 2*(4*channels)
            project = incoming*(4*channels) if projection else 0
            if projection:
                block_affine += 2*(4*channels)
            insertion_shape = ['N',channels,50,height,width]
            height = (height + 2 - 3) // stride + 1
            width = (width + 2 - 3) // stride + 1
            sites.append(dict(name=f'layer{stage}.{block}',input_channels=incoming,channels=channels,
                              insertion_shape=insertion_shape, output_shape=['N',4*channels,50,height,width],
                              conv2_stride=stride,projection=projection,
                              backbone_parameters=main+project+block_affine,
                              backbone_frozen_BN_affine=block_affine))
            backbone += main + project + block_affine
            affine += block_affine
            incoming = 4*channels
        hidden = channels // 16
        gate = channels*channels + 2*(27*channels*channels)
        short = channels*hidden*3 + hidden + hidden*channels*3 + channels
        dynamic = 50*12 + 12 + 12*3 + 3
        tsci = 3*(2*1*7*7 + 2)
        stages.append(dict(stage=f'layer{stage}',channels=channels,blocks=blocks,short_hidden=hidden,
                           gating_per_block=gate,short_per_block=short,dynamic_per_block=dynamic,
                           tsci_per_block=tsci,module_per_block=gate+short+dynamic+tsci,
                           module_stage=blocks*(gate+short+dynamic+tsci),backbone_stage=backbone,
                           backbone_frozen_BN_affine=affine))
    require(sites == spec['sites'], '16 insertion sites/stride/shape or backbone counts differ')
    require(stages == spec['stages'], 'Per-stage counts differ')
    require([s['backbone_stage'] for s in stages] == [215808,1219584,7098368,14964736],
            'Unexpected ResNet50 stage counts')
    stem = 3*64*7*7 + 2*64
    components = dict(backbone=stem+sum(s['backbone_stage'] for s in stages),
                      gating=sum(s['blocks']*s['gating_per_block'] for s in stages),
                      short_term=sum(s['blocks']*s['short_per_block'] for s in stages),
                      dynamic_kernel=16*651,tsci=16*300)
    require(components == spec['components'], 'Component totals differ')
    frozen_backbone = 128 + sum(s['backbone_frozen_BN_affine'] for s in stages)
    require(frozen_backbone == spec['bn_policy']['backbone_frozen_affine'] == 53120,
            'Backbone BN parameter budget differs')
    require(spec['bn_policy']['tsci_frozen_affine'] == 16*3*2 == 96, 'TSCI BN budget differs')

    old_audit = read_json('docs/research/results/aca_architecture_audit_v1.json')
    readiness = read_json('docs/research/results/aca_four_arm_readiness_v1.json')
    matched = []
    for row in spec['totals']:
        arm, classes = row['variant'], row['classes']
        comp = dict(components, classifier=classes*(2048+1))
        if arm not in ('lsta','aca'):
            for key in ('gating','short_term','dynamic_kernel'):
                comp[key] = 0
        if arm not in ('tsci','aca'):
            comp['tsci'] = 0
        frozen = frozen_backbone + (96 if arm in ('tsci','aca') else 0)
        require(row['components'] == comp and row['total'] == sum(comp.values()) and
                row['frozen_affine'] == frozen and row['trainable'] == row['total']-frozen,
                f'Total/trainable parameter mismatch: {arm}/{classes}')
        old = next(r for r in old_audit['parameters']['rows'] if r['variant']==arm and r['classes']==classes)
        require(old['components'] == comp and old['total'] == row['total'], 'Existing meta audit disagrees')
        if classes == 2:
            require(row['total'] == parameter_constant[arm], 'Trainer parameter declaration disagrees')
            ready_rows = [r for r in readiness['rows'] if r['variant']==arm]
            require(len(ready_rows) == 3 and all(r['total_parameters']==row['total'] and
                    r['trainable_parameters']==row['trainable'] for r in ready_rows),
                    'Saved three-seed engineering readiness counts disagree')
        matched.append(f'{arm}/{classes}')
    require(len(matched) == len(set(matched)) == 8, 'Expected four arms times two heads')
    # Permuting height/width must restore the same symbolic axes.
    axes = ['NT','C','H','W']
    for branch in spec['tsci']['branches']:
        permuted = [axes[i] for i in branch['permutation']]
        require([permuted[i] for i in branch['inverse']] == axes, 'TSCI inverse permutation differs')
    return dict(schema='aca_reconstruction_spec_arithmetic_audit_v1',status='passed',
                spec_sha256=file_sha256(ROOT/SPEC),source_pins_verified=len(spec['source_sha256']),
                configuration_checked_by_AST=True,insertion_sites_verified=len(sites),
                arithmetic_matches_existing_meta_rows=matched,
                historical_readiness_count_rows_verified=len(readiness['rows']),
                TSCI_inverse_permutations_verified=3,components=components,
                scope='integer arithmetic, AST, SHA and existing JSON only; not a new forward/gradient test',
                models_instantiated=0,weights_loaded=0,GPU_jobs_started=0,
                inference_runs=0,optimizer_updates=0,test_runs=0)


if __name__ == '__main__':
    print(json.dumps(audit(),ensure_ascii=False,indent=2))
