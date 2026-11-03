"""Explicit, isolated one-time test of the already-frozen three-model fusion.

Original training/validation guards are not changed. Run this file directly so
the original mainline package is loaded before the reproduction-only packages.
"""
import argparse
from collections import Counter
import csv
import gc
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[1]
SOURCE = Path('/home/tsai/projects/basketball')
sys.path.insert(0, str(SOURCE))
import development
# A stable list avoids namespace-path recalculation dropping an appended path
# when a dependency updates sys.path. Original mainline stays first.
development.__path__ = [str(SOURCE / 'development'), str(ROOT / 'development')]

import cv2
import numpy as np
import torch
from development.aca_net_v1.data import (
    VideoScope, Basketball50Dataset, decode_uniform50, normalize_rgb,
    file_sha256, require, select_smoke_ids, _identity,
)
from development.aca_net_v1.train import canonical_hash, configure_runtime, runtime_identity
from development.aca_validation_v1.analyze import (
    metadata, RUNS, protected_files, verify_protected, load_model as load_aca,
    tensor_digest, summarize, write_new, read_json,
)
from development.aca_mainline_fusion_v1.checkpoint import inspect as inspect_mainline, json_equivalent
from development.data_pipeline.pipeline import load_gate_config, SequentialRGBDecoder, gate1a
from development.data_pipeline.multiwindow_v1 import BasketballThreeWindowDataset, select_covering_windows
from development.data_quality.transforms.video_clip_transforms import ClipTransform, derived_seed, normalize_tensor
from development.partial_finetune_v1.initialization import load_initialized_model

CANDIDATE = ROOT / 'development/aca_mainline_fusion_v1/artifacts/mainline_fusion_4Lbbzt/candidate.json'
FROZEN_CANDIDATE_SHA = '72334469211b12620d6655974ec6b390b7e79ece2cbb189918572f436ae1e107'
ARTIFACTS = FOLDER / 'artifacts'
CLAIM = ARTIFACTS / 'three_equal_original_test.claim.json'
NAMES = ('dense', 'dw3d', 'mainline')


def process_starttime():
    return int(Path(f'/proc/{os.getpid()}/stat').read_text().split(') ', 1)[1].split()[19])


class EvaluationScope:
    """New explicit allowlist; test paths require the persistent owner claim."""
    def __init__(self, split, ids=None, *, claim=None, protocol_sha=None):
        require(split in ('validation', 'test'), 'Inference split only')
        pinned = VideoScope(source_root=SOURCE, split='validation')
        manifest = ROOT / 'data/processed/dataset_manifest.csv'
        with manifest.open(newline='', encoding='utf-8') as stream:
            all_rows = list(csv.DictReader(stream))
        require(file_sha256(manifest) == pinned.hashes['manifest'], 'Manifest changed')
        groups = {s: {r['source_group'] for r in all_rows if r['split'] == s}
                  for s in ('train', 'validation', 'test')}
        require(not groups['test'] & (groups['train'] | groups['validation']), 'Group leakage')
        self.split, self.claim, self.protocol_sha = split, claim, protocol_sha
        self.hashes = pinned.hashes
        self.allowed = {r['video_id']: dict(r) for r in all_rows if r['split'] == split}
        wanted = sorted(self.allowed) if ids is None else list(ids)
        require(len(wanted) == len(set(wanted)) and all(i in self.allowed for i in wanted),
                'Duplicate/missing/foreign IDs')
        self.rows = tuple(self.allowed[i] for i in wanted)
        if split == 'test':
            require(ids is None and len(self.rows) == 1539 and
                    Counter(r['binary_label'] for r in self.rows) == {'0': 731, '1': 808},
                    'Full original test cohort required; no exclusions/relabeling')

    def authorize(self, video_id):
        require(video_id in self.allowed, 'Cross-split/unknown ID rejected before path access')
        if self.split == 'test':
            require(self.claim == CLAIM and self.protocol_sha is not None, 'Explicit test claim required')
            record = read_json(self.claim)
            require(record['protocol_sha256'] == self.protocol_sha and record['pid'] == os.getpid()
                    and record['proc_starttime'] == process_starttime(), 'Foreign/stale test claim')
        row = dict(self.allowed[video_id])
        relative = gate1a.safe_relative(row['video_path'])
        return row, SOURCE / 'archive/Basketball_51 dataset' / relative

    def identity(self, video_id):
        _, path = self.authorize(video_id)
        return _identity(path.stat())

    def open(self, video_id, video_path, purpose):
        row, path = self.authorize(video_id)
        require(video_path == row['video_path'] and purpose == 'decode_contiguous_rgb_clip',
                'Unexpected decoder request')
        return gate1a.open_guarded(path)


class Inputs:
    def __init__(self, scope, gate):
        self.scope, self.gate = scope, gate

    def aca(self, video_id):
        rgb, meta, row = decode_uniform50(self.scope, video_id)
        return normalize_rgb(rgb), meta

    def mainline(self, video_id):
        # Same frozen decoder/selection/reference transform as the original
        # dataset, but authorization is exclusively the new explicit test scope.
        row = self.scope.allowed[video_id]
        decoder = SequentialRGBDecoder()
        middle, _, selected, n, fps = decoder(row, self.scope, selection=None,
                                             expected_frame_count=None, expected_fps_rational=None)
        windows = select_covering_windows(n)
        require((selected['start_frame'], selected['end_frame_exclusive']) == windows[1], 'Middle mismatch')
        transform = ClipTransform({k: v for k, v in self.gate.raw['transform'].items() if k != 'policy'}, 'reference')
        tensors = []
        for i, (start, end) in enumerate(windows):
            clip = middle if i == 1 else decoder(
                row, self.scope, selection={'start_frame': start, 'end_frame_exclusive': end},
                expected_frame_count=n, expected_fps_rational=None)[0]
            array, _ = transform(clip, seed=derived_seed(self.gate.raw['seed'], video_id, 0, i, 'reference'),
                                 split=self.scope.split,
                                 sample_metadata={'start_frame': start, 'end_frame_exclusive': end})
            array, _ = normalize_tensor(array, mode='identity')
            tensors.append(torch.from_numpy(array))
        return torch.stack(tensors), {'frame_count': n, 'fps': fps, 'windows': windows}


def prediction(name, logits, source):
    require(torch.isfinite(logits).all().item(), 'Nonfinite logits')
    logits = logits.detach().cpu()
    y = int(source['binary_label'])
    if name == 'mainline':
        require(tuple(logits.shape) == (1,), 'Mainline output shape changed')
        z = logits[0]
        p, pred = z.sigmoid().item(), int(z >= 0)
        logs = torch.nn.functional.logsigmoid(torch.stack((-z, z)).double()).tolist()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, torch.tensor(float(y))).item()
    else:
        require(tuple(logits.shape) == (1, 2), 'ACA output shape changed')
        p, pred = logits.double().softmax(-1)[0, 1].item(), logits.argmax(-1).item()
        logs = logits.double().log_softmax(-1)[0].tolist()
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([y])).item()
    return {'video_id': source['video_id'], 'split': source['split'], 'label': y,
            'original_class': source['original_class'], 'shot_type': source['original_class'][:-1],
            'source_group': source['source_group'], 'logits': logits.flatten().tolist(),
            'p_make': p, 'prediction': pred, 'log_probabilities': logs, 'ce_loss': loss}


def fuse(rows):
    require(len(rows) == 3, 'Exactly three frozen components required')
    base = rows[0]
    keys = ('video_id', 'split', 'label', 'original_class', 'shot_type', 'source_group')
    require(all(all(r[k] == base[k] for k in keys) for r in rows), 'Misaligned fusion')
    require(all(math.isfinite(r['p_make']) and 0 <= r['p_make'] <= 1 and
                len(r['log_probabilities']) == 2 and all(map(math.isfinite, r['log_probabilities']))
                for r in rows), 'Invalid component probabilities')
    # Match the frozen JS implementation's ordered arithmetic, not logit averaging.
    p = 0.0
    logs = []
    for r in rows:
        p += (1 / 3) * r['p_make']
        logs.append(math.log(1 / 3) + r['log_probabilities'][base['label']])
    high = max(logs)
    loss = -(high + math.log(sum(math.exp(z - high) for z in logs)))
    return {**{k: base[k] for k in keys}, 'p_make': p, 'prediction': int(p > .5),
            'ce_loss': loss, 'component_p_make': {n: r['p_make'] for n, r in zip(NAMES, rows)}}


def load_mainline(info):
    model, initialization = load_initialized_model()
    require(initialization == info['config']['initialization'], 'Initialization identity mismatch')
    checkpoint = torch.load(info['mainline_run'] + '/best.pt', map_location='cpu', weights_only=True)
    require(checkpoint['engineering_only'] is False and checkpoint['epoch'] == 0 and
            json_equivalent(checkpoint['config'], info['config']), 'Mainline best mismatch')
    state = checkpoint['model_state_dict']
    require(all(not t.is_floating_point() or (t.dtype == torch.float32 and torch.isfinite(t).all().item())
                for t in state.values()), 'Invalid mainline state')
    model.load_state_dict(state, strict=True)
    return model.requires_grad_(False).eval(), tensor_digest(model.state_dict())


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]


def check_gpu_idle():
    out = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    desktops = {'gnome-remote-desktop-daemon', 'Discord', 'nautilus', 'ptyxis',
                'snapd-desktop-integration', 'gnome-shell', 'Xwayland', 'chrome', 'codex', 'Codex'}
    for value in out.splitlines():
        if value.strip().isdigit() and int(value) != os.getpid():
            exe = Path(os.readlink(f'/proc/{int(value)}/exe')).name
            require(exe in desktops, f'Other GPU computation active: {value} {exe}; not stopped')


def prepare():
    require(file_sha256(CANDIDATE) == FROZEN_CANDIDATE_SHA, 'Frozen candidate changed')
    candidate = read_json(CANDIDATE)
    require(candidate['selected_method'] == 'three_equal' and tuple(candidate['components']) == NAMES,
            'Wrong frozen primary')
    for comp in candidate['components'].values():
        require(comp['weight'] == 1 / 3, 'Weights changed')
        for key in ('checkpoint', 'config'):
            require(file_sha256(comp[key]) == comp[key + '_sha256'], f'Frozen {key} changed')
    specs = {n: metadata(n, p) for n, p in RUNS.items()}
    info = inspect_mainline()  # Must happen before CUDA initialization.
    protected = protected_files(specs)
    protected.update(info['protected_files_sha256'])
    protected[str(CANDIDATE)] = FROZEN_CANDIDATE_SHA
    for path in FOLDER.glob('*.py'):
        protected[str(path)] = file_sha256(path)
    gate = load_gate_config()
    require(gate.sha256 == info['config']['gate_config_sha256'], 'Frozen transform changed')
    for path in (gate.path, *gate.paths.values()):
        protected[str(path)] = file_sha256(path)
    for path in (ROOT / 'data/processed/dataset_manifest.csv', ROOT / 'data/processed/split_summary.json'):
        protected[str(path)] = file_sha256(path)
    test = EvaluationScope('test')  # Metadata only, cannot authorize a video.
    old = []
    for path in sorted((SOURCE / 'runs').glob('*/test_metrics.json')):
        # Metadata establishes prior holdout exposure, never model selection.
        record = read_json(path)
        if record.get('manifest_sha256') == test.hashes['manifest']:
            old.append({'path': str(path), 'sha256': file_sha256(path),
                        'same_manifest': True, 'n': record.get('n')})
    protocol = {'schema': 'frozen_three_equal_one_time_test_v1', 'candidate_sha256': FROZEN_CANDIDATE_SHA,
                'primary': 'three_equal', 'appendix': list(NAMES), 'weights': {n: 1 / 3 for n in NAMES},
                'cohort': 'all_original_1539_test_videos_no_exclusions_no_relabeling',
                'input_hashes': test.hashes, 'ordered_ids_sha256': canonical_hash([r['video_id'] for r in test.rows]),
                'threshold': 'fusion p_make > 0.5, ties Miss; single original tie rules retained',
                'selection': 'frozen_validation_selected_checkpoints_and_fusion; no_test_selection',
                'mainline_selected_epoch': 0, 'aca_not_verified_author_original': True,
                'test_history': old, 'globally_unseen_test': not bool(old),
                'validation_parity_tolerance': {'dense': 1e-6, 'dw3d': 1e-6, 'mainline': 1e-5},
                'optimizer_updates': 0, 'training': False, 'test_tuning': False,
                'protected_files_sha256': protected}
    return candidate, specs, info, gate, protocol


def parity(specs, info, gate, output):
    ids = select_smoke_ids(VideoScope(source_root=SOURCE, split='validation'))
    scope = EvaluationScope('validation', ids)
    inputs = Inputs(scope, gate)
    original_aca = Basketball50Dataset(source_root=SOURCE, split='validation', video_ids=ids)
    original_main = BasketballThreeWindowDataset(gate, split='validation', video_ids=ids, epoch=0)
    for i, video_id in enumerate(ids):
        torch.testing.assert_close(inputs.aca(video_id)[0], original_aca[i]['tensor'], rtol=0, atol=0)
        torch.testing.assert_close(inputs.mainline(video_id)[0], original_main[i]['tensor'], rtol=0, atol=0)
    saved = ROOT / 'development/aca_validation_v1/artifacts/validation_compare_vykz5xz0'
    all_rows = {n: read_rows(saved / f'{n}_predictions.jsonl') for n in NAMES[:2]}
    all_rows['mainline'] = info['rows']
    expected = {n: {r['video_id']: r for r in rows} for n, rows in all_rows.items()}
    # Reproduce the whole saved validation fusion before opening any test video.
    saved_fusion = read_rows(read_json(CANDIDATE)['selected_predictions'])
    replay = [fuse([expected[n][r['video_id']] for n in NAMES]) for r in saved_fusion]
    require(len(replay) == 1547, 'Validation fusion cohort mismatch')
    for actual, reference in zip(replay, saved_fusion):
        require(actual['prediction'] == reference['prediction'] and
                abs(actual['p_make'] - reference['p_make']) < 1e-14 and
                abs(actual['ce_loss'] - reference['ce_loss']) < 1e-12, 'Frozen fusion arithmetic mismatch')
    results = {}
    for name in NAMES:
        model, before = load_mainline(info) if name == 'mainline' else load_aca(name, specs[name])
        model.to('cuda')
        differences = []
        with torch.inference_mode():
            for source in scope.rows:
                x, _ = inputs.mainline(source['video_id']) if name == 'mainline' else inputs.aca(source['video_id'])
                actual = prediction(name, model(x.unsqueeze(0).to('cuda')), source)
                reference = expected[name][source['video_id']]
                difference = abs(actual['p_make'] - reference['p_make'])
                require(actual['prediction'] == reference['prediction'] and difference <= (1e-5 if name == 'mainline' else 1e-6),
                        f'{name} saved validation inference parity failed: {source["video_id"]} delta={difference}')
                differences.append(difference)
        require(tensor_digest(model.state_dict()) == before, 'State changed in parity')
        results[name] = {'max_probability_absolute_difference': max(differences), 'predictions_identical': True}
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(f'validation parity {name} passed', flush=True)
    report = {'status': 'passed', 'validation_ids': ids, 'input_tensors_bitwise_equal': True,
              'saved_fusion_all_1547_reproduced': True, 'models': results, 'test_videos_opened': 0}
    write_new(output / 'validation_parity.json', report)
    return report


def infer(name, scope, gate, specs, info, output):
    started = time.perf_counter()
    model, before = load_mainline(info) if name == 'mainline' else load_aca(name, specs[name])
    model.to('cuda')
    inputs = Inputs(scope, gate)
    rows = []
    with (output / f'{name}_predictions.jsonl').open('x', encoding='utf-8') as stream, torch.inference_mode():
        for source in scope.rows:
            x, meta = inputs.mainline(source['video_id']) if name == 'mainline' else inputs.aca(source['video_id'])
            row = prediction(name, model(x.unsqueeze(0).to('cuda')), source)
            row['input_metadata'] = meta
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            if len(rows) % 100 == 0 or len(rows) == len(scope.rows):
                stream.flush()
                print(f'test {name} videos={len(rows)}/{len(scope.rows)} seconds={time.perf_counter()-started:.1f}', flush=True)
    require(all(not m.training for m in model.modules()) and
            all(p.grad is None and not p.requires_grad for p in model.parameters()) and
            tensor_digest(model.state_dict()) == before, 'Model state/BN/grad changed')
    write_new(output / f'{name}_verification.json', {'state_unchanged': True, 'videos': len(rows),
               'seconds': time.perf_counter()-started, 'model_state_sha256': before, 'optimizer_updates': 0})
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def render_report(report, output):
    names = {'three_equal': '三模型等權融合（主結果）', 'dense': 'dense ACA 候選',
             'dw3d': 'ACA-DW3D 候選', 'mainline': 'MixNet-I3D＋Transformer（epoch 0）'}
    rows = ''
    for name in ('three_equal', *NAMES):
        m = report['metrics'][name]
        rows += (f'<tr><th>{names[name]}</th><td>{100*m["accuracy"]:.2f}%</td>'
                 f'<td>{m["macro_f1"]:.4f}</td><td>{m["loss"]:.6f}</td>'
                 f'<td>{100*m["miss_recall"]:.2f}%</td><td>{100*m["make_recall"]:.2f}%</td>'
                 f'<td>{html.escape(str(m["confusion_rows_true_columns_predicted"]))}</td></tr>')
    shots = ''.join(f'<tr><th>{s}</th><td>{m["count"]}</td><td>{100*m["accuracy"]:.2f}%</td></tr>'
                    for s, m in report['metrics']['three_equal']['by_shot_type'].items())
    document = f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>凍結三模型 Test 評估</title>
<style>body{{font:17px/1.7 system-ui;max-width:1150px;margin:40px auto;padding:20px;color:#17243a;background:#fafbfe}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{padding:12px;border:1px solid #d7deea;text-align:left}}.note{{background:#fff1ce;padding:18px}}h1{{font-size:30px}}</style>
<h1>原低解析度三模型融合：固定 Test 評估</h1><p>共 1539 部原始 test 影片，Miss=731、Make=808；未刪除、改標籤或依 test 調參。以下為 <b>test 成績，不是 validation</b>。</p>
<table><tr><th>模型</th><th>Accuracy</th><th>Macro-F1</th><th>CE loss</th><th>Miss recall</th><th>Make recall</th><th>混淆矩陣</th></tr>{rows}</table>
<p>矩陣列為真實 [Miss, Make]，欄為預測 [Miss, Make]。單模型為附表用途，不依結果替換主模型。</p>
<h2>主結果依投籃類別</h2><table><tr><th>類別</th><th>影片數</th><th>Accuracy</th></tr>{shots}</table>
<h2>模型與評估條件</h2><p>dense ACA 使用第 8 輪、DW3D 第 4 輪、主線第 0 輪，均在 test 前依 validation 選定。主線第 0 輪表示初始化權重被保留，不代表部分微調改善。三個 Make 機率各占 1/3，融合機率大於 0.5 才判進球；不平均 logits。ACA 候選不是已核實作者原版。</p>
<p>ACA：全影片均勻 50 幀，RGB 160×120，ImageNet normalization。主線：三段連續 64 幀，RGB 256×192，原 reference transform 與模型內 normalization。FP32、eval/inference mode、零 optimizer updates。未使用高解析度修改。</p>
<p class="note">限制：相同 test 切分曾用於歷史其他模型；本次是此凍結三模型融合的首次評估，不能稱為整個專案完全未見的最終測試。Validation 曾反覆用於開發。若以本次 test 結果改版，需另設獨立外部評估；本結果不保證其他資料達 90%。</p>
<p><a href="report.json">完整 report.json</a> · <a href="protocol.json">事前凍結 protocol</a> · <a href="validation_parity.json">前處理與推論一致性檢查</a></p></html>'''
    with (output / 'index.html').open('x', encoding='utf-8') as stream:
        stream.write(document)


def run(execute):
    candidate, specs, info, gate, protocol = prepare()
    if not execute:
        print(json.dumps({k: v for k, v in protocol.items() if k != 'protected_files_sha256'}, indent=2))
        return
    require(not CLAIM.exists(), 'Persistent test claim exists: no automatic repeat/resume')
    check_gpu_idle()
    require(shutil.disk_usage(FOLDER).free > 5 * 1024**3, 'Insufficient free disk')
    ARTIFACTS.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='three_equal_test_', dir=ARTIFACTS))
    write_new(output / 'protocol.json', protocol)
    write_new(output / 'candidate_snapshot.json', candidate)
    for path in FOLDER.glob('*.py'):
        shutil.copyfile(path, output / path.name)
    configure_runtime(20261025)
    print(f'output={output}', flush=True)
    parity(specs, info, gate, output)
    verify_protected(protocol['protected_files_sha256'])
    # Exclusive persistent claim is created only after CPU/GPU validation passes.
    claim = {'protocol_sha256': canonical_hash(protocol), 'candidate_sha256': FROZEN_CANDIDATE_SHA,
             'output': str(output), 'pid': os.getpid(), 'proc_starttime': process_starttime(),
             'one_time': True, 'automatic_retry_allowed': False}
    write_new(CLAIM, claim)
    print('TEST CLAIMED: frozen configuration, no tuning, no retries', flush=True)
    try:
        scope = EvaluationScope('test', claim=CLAIM, protocol_sha=claim['protocol_sha256'])
        rows = {n: infer(n, scope, gate, specs, info, output) for n in NAMES}
        fused = [fuse([rows[n][i] for n in NAMES]) for i in range(len(scope.rows))]
        with (output / 'three_equal_predictions.jsonl').open('x', encoding='utf-8') as stream:
            for row in fused:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        require(all(len(r) == 1539 and [x['video_id'] for x in r] == [s['video_id'] for s in scope.rows]
                    for r in (*rows.values(), fused)), 'Incomplete/unaligned test cohort')
        verify_protected(protocol['protected_files_sha256'])
        report = {'status': 'completed', 'scope': 'test_not_validation', 'primary': 'three_equal',
                  'count': 1539, 'metrics': {**{n: summarize(r) for n, r in rows.items()}, 'three_equal': summarize(fused)},
                  'protocol_sha256': claim['protocol_sha256'], 'protected_files_unchanged': True,
                  'optimizer_updates': 0, 'test_tuning': False, 'globally_unseen_test': protocol['globally_unseen_test'],
                  'test_history': protocol['test_history'], 'runtime': runtime_identity(),
                  'test_video_decodes': {'dense': 1539, 'dw3d': 1539, 'mainline': 3 * 1539}}
        render_report(report, output)
        write_new(output / 'report.json', report)
        print(json.dumps({'status': 'completed', 'output': str(output), 'metrics': report['metrics']}, ensure_ascii=False), flush=True)
    except BaseException as exc:
        write_new(output / 'failure.json', {'status': 'failed_no_automatic_retry', 'error': repr(exc),
                   'claim_preserved': True, 'pid': os.getpid()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-once', action='store_true', help='Explicitly authorize validation parity then one frozen test')
    run(parser.parse_args().execute_once)
