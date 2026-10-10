#!/usr/bin/env python3
"""Verify derived visual-pilot evidence and publish/check matching Markdown/JSON; never open videos."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import audit_visual_preprocessing_pilot_v1 as p

REPORT = p.ROOT/'docs/data_quality/visual_preprocessing_pilot_v1'


def trace_evidence(path, selected_ids, require_videos=True):
    text = path.read_text()
    p.need('+++ exited with 0 +++' in text and '+++ killed' not in text, 'Incomplete/failed syscall trace')
    # %file traces all path-based access attempts, including failed opens/stats.
    # -yy also renders the source path for /proc/self/fd and descriptor-relative calls.
    video_lines = [line for line in text.splitlines() if '.mp4' in line]
    observed, opens, writes = set(), 0, []
    for line in video_lines:
        ids = re.findall(r'/archive/Basketball_51 dataset/([^\s<>"\)]+\.mp4)', line)
        p.need(ids, 'Unrecognized video path in syscall trace')
        observed.update(ids)
        if re.search(r'\bopen(at|at2)?\(', line):
            opens += 1
        if any(flag in line for flag in ('O_WRONLY', 'O_RDWR', 'O_TRUNC', 'O_CREAT')):
            writes.append(line)
    p.need(not writes, 'Video write attempt in trace')
    p.need(observed <= set(selected_ids), 'Non-pilot video access in trace')
    if require_videos:
        p.need(observed == set(selected_ids) and opens >= len(selected_ids), 'Missing selected video opens')
    else:
        p.need(not observed, 'Metadata/test command unexpectedly accessed videos')
    rows = p.qa.read_csv(p.qa.DEFAULT_MANIFEST)
    val = {r['video_id'] for r in rows if r['split'] == 'validation'}
    test = {r['video_id'] for r in rows if r['split'] == 'test'}
    return dict(trace=str(path.relative_to(p.ROOT)), sha256=p.sha256(path),
        trace_scope='strace -f -yy -e trace=%file; all path attempts; descriptor paths annotated',
        unique_video_paths=len(observed), video_path_syscalls=len(video_lines), video_open_syscalls=opens,
        validation_video_access=len(observed & val), test_video_access=len(observed & test),
        non_pilot_video_access=len(observed-set(selected_ids)), video_write_attempts=0)


def verify(out, trace):
    summary = json.loads((out/'summary.json').read_text())
    samples = json.loads((out/'samples.json').read_text())
    p.need(p.sha256(out/'samples.json') == summary['samples_sha256'], 'Samples digest mismatch')
    cfg = p.load_config()
    records, _ = p.validate_manifest(p.MANIFEST, cfg)
    p.need(summary['config'] == cfg and len(samples) == 48, 'Config/count mismatch')
    p.need([s['selection'] for s in samples] == records, 'Sampler metadata mismatch')
    for key, digest in summary['provenance'].items():
        p.need(p.sha256(p.ROOT/key) == digest, f'Provenance changed: {key}')
    aggregate = p.aggregate(samples)
    p.need(aggregate == summary['diagnostics'], 'Aggregate mismatch')
    expected_counts = dict(videos=48, clips=48, retained_frames=3072,
        decoder_read_frames=sum(r['end_frame_exclusive'] for r in records),
        prefix_frames=sum(r['start_frame'] for r in records),
        fallback_clips=sum(r['fallback'] for r in records),
        candidate_overlap=sum(r['candidate_overlap'] for r in records),
        sources=len({r['source_group'] for r in records}), same_seed_checks=240)
    for key, value in expected_counts.items():
        p.need(summary[key] == value, f'Count mismatch: {key}')
    for key, field in [('labels', 'label'), ('categories', 'category'), ('fps', 'fps_rational'), ('source_counts', 'source_group')]:
        p.need(summary[key] == dict(Counter(r[field] for r in records)), f'Group mismatch: {key}')
    p.need(len(summary['access_events']) == 48 and
        {e['video_id'] for e in summary['access_events']} == {r['video_id'] for r in records} and
        all(e['split'] == 'train' and e['read_only'] for e in summary['access_events']), 'Access ledger mismatch')
    for s in samples:
        rec, decoding = s['selection'], s['decoding']
        p.need(decoding['retained_frames'] == 64 and decoding['prefix_frames'] == rec['start_frame'] and
            decoding['decoder_read_frames'] == rec['end_frame_exclusive'] and
            decoding['source_unchanged'] and decoding['timestamps_match_indices'], 'Decode evidence mismatch')
        for strategy in p.STRATEGIES:
            candidate = s['candidates'][strategy]
            meta = candidate['metadata']
            t = meta['transform']
            p.need(meta['sample'] == rec and candidate['same_seed_equal'], 'Lost sampler metadata/reproducibility')
            p.need(t['shared_across_all_frames'] and t['normalization'] == 'none', 'Transform contract mismatch')
            p.need(t['seed'] == p.derived_seed(cfg['seed'], rec['video_id'], cfg['epoch'], 0, strategy), 'Seed mismatch')
            p.need(t['input_shape'] == decoding['input_shape'] and
                t['output_shape'] == [3,64,cfg['transform']['output_height'],cfg['transform']['output_width']], 'Shape mismatch')
            x,y,w,h = t['parameters']['crop_xywh']
            _,ih,iw,_ = t['input_shape']
            p.need(0 <= x < x+w <= iw and 0 <= y < y+h <= ih, 'Crop bounds')
            p.need(t['parameters']['horizontal_flip'] == (strategy == 'horizontal_flip'), 'Flip mismatch')
            d = candidate['diagnostics']
            p.need(all(np.isfinite(v) for v in d.values()) and 0 <= d['value_min'] <= d['value_max'] <= 1,
                   'Invalid diagnostic values')
    for name in summary['contact_sheets']:
        p.need((out/name).is_file(), 'Missing contact sheet')
    evidence = trace_evidence(trace, [r['video_id'] for r in records])
    # The publication contains aggregates/provenance, not per-frame data or local source identities.
    result = {k: v for k,v in summary.items() if k not in ('access_events', 'command')}
    result['command'] = ['python', '-B', str(Path(p.__file__).relative_to(p.ROOT))]
    result['artifact_directory'] = str(out.relative_to(p.ROOT))
    result['verification'] = evidence
    result['verifier_sha256'] = p.sha256(Path(__file__))
    review = out/'review.json'
    p.need(review.is_file(), 'Review evidence required before publication')
    result['visual_review'] = json.loads(review.read_text())
    result['review_sha256'] = p.sha256(review)
    validation = out/'validation.json'
    p.need(validation.is_file(), 'Validation evidence required before publication')
    result['validation'] = json.loads(validation.read_text())
    result['validation_sha256'] = p.sha256(validation)
    return result


def render_report(s):
    rows = ['# Basketball-51 Visual Preprocessing / Augmentation Pilot v1', '',
        '本輪完成可重現的 train-only Pilot；正式 enhancement／augmentation 政策尚未定案。', '',
        '## 組成與解碼', '', '| 項目 | 數值 |', '| --- | --- |']
    for key in ('videos','clips','retained_frames','decoder_read_frames','prefix_frames','sources',
                'fallback_clips','candidate_overlap','same_seed_checks'):
        rows.append(f'| {key} | {s[key]} |')
    rows += ['', f"- Label：`{json.dumps(s['labels'], sort_keys=True)}`。",
        f"- FPS：`{json.dumps(s['fps'], sort_keys=True)}`（23.976／29.97）。",
        f"- Category：`{json.dumps(s['categories'], sort_keys=True)}`。",
        f"- Label × FPS：`{json.dumps(s['label_fps'], sort_keys=True)}`。",
        f"- Pilot seed={s['config']['seed']}；sampler seed={s['config']['sampler_seed']}；epoch={s['config']['epoch']}。",
        '- 所有 5 支 fallback 都保留原始連續 64 格，總 overlap 20；其他 43 支窗口 overlap=0。',
        '- 先納入 5 支 fallback，再以 label × FPS 最少格、交替 category、優先未用來源、SHA-256 破同分選樣。',
        '- 扣除 5 支 fallback 後剩 43 個位置，因此 affected-clean／unaffected 為 22／21；不是偷偷替換。',
        '- 涵蓋 35 個可用 train 來源中的 32 個；逐步優先未用來源的貪婪策略不保證全域最大來源覆蓋。來源代碼不等於已確認比賽 ID；配額取樣不能估計完整 train 品質比例。',
        '- 依序解碼至 end，必要前綴只供精確 frame 定位、即讀即丟；僅保留選中窗口，不 decode 窗口後影格。',
        '- 原始連續 index 與 decoder timestamps 誤差均小於半個 frame；不跳格、拼接或插格。', '',
        '## 四類處理', '',
        '1. Deterministic preprocessing：guarded OpenCV/FFmpeg decode、BGR→RGB、最大 4:3 中心裁切、INTER_AREA resize 到 192×256、uint8 THWC→float32 CTHW [0,1]。',
        '2. Image enhancement：固定 gamma=0.95，公式 x^gamma；無自動逐幀估參，不是模糊修復或 data augmentation。',
        '3. Training augmentation：brightness／contrast／saturation 各 [0.95,1.05]；另試線性 crop scale [0.92,1]。每個 clip 共用參數。水平翻轉為獨立方向敏感選項。',
        '4. Quality-aware sampling：沿用 frozen sampler v1 選低 candidate overlap 的連續窗口；candidate 不是人工確認的模糊真值。', '',
        '## Pixel 與時間診斷', '',
        '下表為每 clip 指標的等權平均；完整 mean/min/max 在同名 JSON。Brightness 為 RGB Rec.709 權重；contrast 為各幀 luma std 的平均；endpoint/new clip 以 RGB channel elements 計。', '',
        '| 策略 | Mean | Std | Brightness | Contrast | Saturation | Endpoint % | New clip % | Laplacian | Tenengrad | Δbrightness | Δ殘差 MAE |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    keys = ['mean','std','brightness','contrast','saturation','endpoint_channel_fraction',
            'newly_clipped_channel_fraction','laplacian_variance','tenengrad','frame_brightness_abs_delta',
            'brightness_delta_residual_mae_vs_reference']
    for name, data in s['diagnostics'].items():
        values = [data[k]['mean']*(100 if 'fraction' in k else 1) for k in keys]
        rows.append('| '+name+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
    rows += ['', '- 所有輸出 float32、[3,64,192,256]、range [0,1]，無 padding；保留比例與 aspect rounding 見 JSON。',
        '- crop、flip、色彩參數共用一份 metadata，240 次同 seed 完整 tensor bitwise 比較通過。',
        '- 合成重複 frame 測試檢查不新增參數性 flicker；不同 seed 只驗證預期的 photometric／spatial 變化。',
        '- 真實影片逐幀 brightness delta 與對 reference 的 delta residual 包含動作、鏡頭切換及 crop 內容改變，不能單憑它宣稱無感知閃爍。',
        '- Laplacian／Tenengrad 在相同輸出尺寸、luma×255 上計算，只是診斷；分數增加不等於品質或準確率改善。', '',
        '## 視覺檢查與決策', '', *['- '+v for v in s['visual_review']['observations']], '',
        '- 保留：reference 作為研究對照；clip 共用參數、seed 與 metadata 契約。',
        '- 暫留候選：mild spatial，仍需邊緣人物／籃框與人工時序檢查。Photometric 僅保留研究實作，當前參數因 clipping 個案而暫緩採用。',
        '- 尚待決定：gamma enhancement 是否有可見性收益；horizontal flip 改變左右與慣用手，Make/Miss label 不變性尚未實證。',
        '- 拒絕作為本輪正式預設：當前 photometric 參數、自動 enhancement／水平翻轉、sharpness score 單獨決策，以及需求禁止的時序／重度幾何策略。', '',
        '## 介面、安全與驗證', '',
        '- `ClipTransform(config, strategy)(rgb_uint8_THWC, seed=..., split="train", sample_metadata=selection)` → `(float32_CTHW, metadata)`。',
        '- `normalize_tensor` 獨立接受 explicit identity／mean_std／affine；未指定 checkpoint，因此不預設 I3D mean/std 或 [-1,1]。可由後續 DataLoader 使用 `torch.from_numpy`，本輪沒有 DataLoader/model 實作。',
        '- reference 介面可供未來固定評估使用；所有候選遇 validation/test split 都拒絕。本輪沒有讀取其影片。',
        f"- Syscall trace：{s['verification']['unique_video_paths']} 支 Pilot，validation={s['verification']['validation_video_access']}、test={s['verification']['test_video_access']}、名單外={s['verification']['non_pilot_video_access']}、write={s['verification']['video_write_attempts']}。",
        f"- Trace SHA-256：`{s['verification']['sha256']}`；檔案位於 `{s['verification']['trace']}`。",
        '- 驗證：'+s['validation']['summary'],
        '- JSON／Markdown 由同一 verified summary 產生並逐 byte check；完整驗證記錄見 JSON validation。', '',
        '## 產物與重現', '',
        '- [方法、定義與重現命令](visual_preprocessing_method_v1.md)',
        '- [結構化摘要](visual_preprocessing_pilot_v1.json)',
        '- [固定 Pilot manifest](../../development/data_quality/manifests/visual_preprocessing_pilot_manifest_v1.csv)',
        '- [設定](../../development/data_quality/configs/visual_preprocessing_pilot_v1.yaml)',
        f"- Ignored contact sheets（{len(s['contact_sheets'])} 張）：`{s['artifact_directory']}/contact_sheets/`；包含全部 fallback、2 個 unaffected、2 個 affected-clean。", '',
        '## 研究限制與停止點', '',
        '48 支、單一 epoch、每支一個窗口，不代表完整 train 或未來 checkpoint 的最佳輸入。Contact sheets 為 4 個時間點的稀疏靜態檢查，未完成人工連續播放審查；沒有 NR-IQA 指標或分類 accuracy。', '',
        '本輪只使用 train 影片；未修改原始影片、label、固定 manifest/split、門檻、sidecar、既有 sampler 結果或 frozen A1/A2/B/C。未完成正式影像增強政策，未執行 I3D／YOLO／其他模型訓練或評估，未下載 checkpoint，未證明任何策略改善 Make/Miss 準確率。停止於 full-video Pilot；不是 pre-release prediction。沒有 git add、commit 或 push。', '']
    return '\n'.join(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path, default=p.OUTPUT)
    ap.add_argument('--trace', type=Path, default=p.OUTPUT.parent/'visual_preprocessing_pilot_v1_runtime/run.strace')
    ap.add_argument('--publish', action='store_true')
    args = ap.parse_args(argv)
    result = verify(args.output_dir, args.trace)
    outputs = {REPORT.with_suffix('.json'): p.dumps(result), REPORT.with_suffix('.md'): render_report(result)}
    if args.publish:
        p.need(all(not path.exists() for path in outputs), 'Refusing existing publication')
        for path, content in outputs.items():
            p.write_new(path, content)
    else:
        for path, content in outputs.items():
            p.need(path.read_text() == content, f'Publication mismatch: {path.name}')
    print('VERIFY_PASS: metadata, provenance, sampler, aggregate, syscall scope, JSON/Markdown')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(2)
