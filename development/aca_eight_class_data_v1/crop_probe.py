"""Bounded fixed16 input-retention probe; never a training/data-loader entrypoint.

One conservative rule family, fixed before candidate outputs: 5percent per
spatial edge and floor(F/20) frames per temporal edge. No target detector,
label-conditioned decisions, parameter search, validation or test access.
"""
from datetime import datetime
from html import escape
import hashlib
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from . import foundation as f

ROOT = f.ROOT
BASE = ROOT / 'development/aca_eight_class_data_v1/artifacts/input_foundation_20261115_v1'
OUT = ROOT / 'development/aca_eight_class_data_v1/artifacts/crop_retention_fixed16_20261116_v1'
VARIANTS = ('base', 'space', 'time', 'both')
KEYS = (0, 8, 16, 21, 24, 27, 33, 41, 49)


def now():
    return datetime.now(ZoneInfo('Asia/Taipei')).isoformat()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def candidate(native, *, spatial, temporal):
    f.need(isinstance(native, np.ndarray) and native.dtype == np.uint8 and
           native.ndim == 4 and native.shape[1:] == (240, 320, 3), 'Invalid native RGB source')
    count = len(native)
    trim = count // 20 if temporal else 0
    available = count - 2 * trim
    indices = tuple(trim + i for i in f.baseline.uniform_indices(available))
    xyxy = (16, 12, 304, 228) if spatial else (0, 0, 320, 240)
    x0, y0, x1, y1 = xyxy
    rgb = np.stack([cv2.resize(native[i, y0:y1, x0:x1], (160, 120),
                               interpolation=cv2.INTER_AREA) for i in indices])
    return rgb, dict(source_indices=list(indices), crop_xyxy=list(xyxy),
                     source_frame_count=count, trim_each_edge_frames=trim,
                     selected_source_span_frames=indices[-1]-indices[0],
                     output_shape=[50, 120, 160, 3])


def native_decode(scope, video_id):
    _, path = scope.authorize(video_id)
    fd = f.baseline.gate1a.open_guarded(path)
    cap = None
    try:
        before = os.fstat(fd)
        cap = cv2.VideoCapture(f'/proc/self/fd/{fd}', cv2.CAP_FFMPEG)
        f.need(cap.isOpened(), 'Cannot decode source')
        raw_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        f.need(np.isfinite(raw_count) and raw_count == int(raw_count) and raw_count >= 50
               and np.isfinite(fps) and fps > 0, 'Invalid source metadata')
        frames = []
        for _ in range(int(raw_count)):
            ok, bgr = cap.read()
            f.need(ok and isinstance(bgr, np.ndarray) and bgr.dtype == np.uint8 and
                   bgr.shape == (240, 320, 3), 'Truncated or unexpected source frame')
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        f.need(not cap.read()[0], 'Source frame count mismatch')
        after = os.fstat(fd)
        f.need(f.baseline._identity(before) == f.baseline._identity(after), 'Source changed')
        return np.stack(frames), fps
    finally:
        if cap is not None:
            cap.release()
        os.close(fd)


def draw_contact(path, rgb, meta, fps, font):
    sheet = Image.new('RGB', (800, 1450), '#20242b')
    pen = ImageDraw.Draw(sheet)
    for slot, frame in enumerate(rgb):
        x, y = (slot % 5)*160, (slot // 5)*145
        index = meta['source_indices'][slot]
        pen.text((x+2, y+2), f'{slot:02d} f{index} ~{index/fps:.2f}s', fill='white', font=font)
        sheet.paste(Image.fromarray(frame), (x, y+25))
    sheet.save(path)


def draw_comparison(path, variants, metadata, fps, font):
    sheet = Image.new('RGB', (1280, len(KEYS)*270), '#20242b')
    pen = ImageDraw.Draw(sheet)
    for row, slot in enumerate(KEYS):
        for col, name in enumerate(VARIANTS):
            x, y = col*320, row*270
            index = metadata[name]['source_indices'][slot]
            pen.text((x+3, y+3), f'{name} slot{slot} f{index} ~{index/fps:.2f}s', fill='white', font=font)
            sheet.paste(Image.fromarray(variants[name][slot]).resize((320,240), Image.Resampling.NEAREST), (x,y+30))
    sheet.save(path)


def draw_boundaries(path, native, fps, indices, font, crop_overlay=False):
    sheet = Image.new('RGB', (1600, ((len(indices)+4)//5)*270), '#20242b')
    for k, index in enumerate(indices):
        frame = Image.fromarray(native[index])
        if crop_overlay:
            ImageDraw.Draw(frame).rectangle((16,12,303,227), outline='#ff3b30', width=2)
        x, y = (k%5)*320, (k//5)*270
        sheet.paste(frame, (x,y+30))
        ImageDraw.Draw(sheet).text((x+3,y+3), f'f{index} ~{index/fps:.3f}s', fill='white', font=font)
    sheet.save(path)


def build():
    f.need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Explicit CPU-only environment required')
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    records = json.loads((BASE/'representative16/records.json').read_text())
    base_plan = json.loads((BASE/'plan.json').read_text())
    f.need(len(records)==16 and [r['video_id'] for r in records] ==
           [r['video_id'] for r in base_plan['selected_16']], 'Fixed16 identity changed')
    release_path = ROOT/'docs/research/results/input_processing_foundation_release_v1.json'
    release = json.loads(release_path.read_text())
    protected = dict(release['source_files_sha256'])
    protected['docs/research/results/input_processing_foundation_release_v1.json'] = f.baseline.file_sha256(release_path)
    for path in BASE.rglob('*'):
        if path.is_file():
            protected[str(path.relative_to(ROOT))] = f.baseline.file_sha256(path)
    for relative, expected in protected.items():
        f.need(f.baseline.file_sha256(ROOT/relative) == expected, 'Frozen baseline changed')
    OUT.mkdir(exist_ok=False)
    plan = dict(schema='fixed16_crop_retention_probe_v1', started_at_taiwan=now(),
        status='candidate_generation', selected_ids=[r['video_id'] for r in records],
        split='train_only', variants=list(VARIANTS),
        spatial_rule='Central90percent: nativeRGB[12:228,16:304], then160x120 INTER_AREA',
        temporal_rule='Trim floor(F/20) source frames at each edge, then endpoint-uniform50 from remaining range',
        temporal_and_spatial_parameters_fixed_before_candidate_outputs=True,
        sample_previously_seen=True, comparison_is_exploratory_not_blind=True,
        selection_by_label_or_predicted_outcome=False, models_loaded=False,
        tensor_shape=[3,50,120,160], augmentation='none',
        decision_rule=['Reject formal default if introduced loss of essential shooter/ball/rim/court cues is confirmed or unresolved.',
                       'Do not adopt for compute savings: same50xRGBx120x160 model input.',
                       'Assess whether known graphics/closeups/occlusion remain; no accuracy inference.',
                       'No tuning/replacement sample/automatic fallback after seeing results.'],
        no_global_activation=True, no_training=True, test_video_access=False,
        protected_baseline_files_sha256=protected)
    write(OUT/'plan.json', plan)
    scope = f.FoundationScope(source_root='/home/tsai/projects/basketball', split='train',
                              video_ids=plan['selected_ids'])
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    results = []
    try:
        for index, record in enumerate(records):
            folder = OUT/f'{index:02d}'
            folder.mkdir()
            native, fps = native_decode(scope,record['video_id'])
            source_sha = f.baseline.guarded_source_sha256(scope,record['video_id'])
            f.need(source_sha == record['source_sha256'], 'Fixed16 source hash changed')
            variants, metadata = {}, {}
            for name in VARIANTS:
                variants[name], metadata[name] = candidate(native,spatial=name in ('space','both'),
                                                            temporal=name in ('time','both'))
                tensor = f.baseline.normalize_rgb(variants[name])
                f.need(tuple(tensor.shape)==(3,50,120,160) and tensor.dtype==torch.float32 and
                       tensor.is_contiguous() and torch.isfinite(tensor).all().item(), 'Invalid candidate input')
                metadata[name]['tensor_sha256'] = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
                metadata[name]['tensor_elements'] = tensor.numel()
                draw_contact(folder/f'{name}.png', variants[name],metadata[name],fps,font)
            f.need(metadata['base']['tensor_sha256']==record['tensor_sha256'], 'Frozen baseline parity failed')
            f.need(metadata['base']['source_indices']==record['source_frame_indices'], 'Baseline timing changed')
            draw_comparison(folder/'comparison.png', variants,metadata,fps,font)
            draw_boundaries(folder/'head_native.png',native,fps,list(range(15)),font)
            draw_boundaries(folder/'tail_native.png',native,fps,list(range(len(native)-15,len(native))),font)
            draw_boundaries(folder/'crop_overlay.png',native,fps,metadata['base']['source_indices'],font,True)
            result = dict(index=index,video_id=record['video_id'],source_group=record['source_group'],
                          split='train',source_frame_count=len(native),fps=fps,
                          source_sha256=source_sha,variants=metadata,baseline_parity_passed=True,
                          source_decode_count=1,video_open_count_for_decode=1,
                          landmark_or_event_annotation=None,visual_review='pending')
            write(folder/'record.json',result)
            results.append(result)
            write(OUT/'records.json',results)
            print(f'PROGRESS {index+1}/16 {record["video_id"]}: all four inputs valid, baseline parity passed',flush=True)
        for relative, expected in protected.items():
            f.need(f.baseline.file_sha256(ROOT/relative)==expected, 'Frozen baseline changed after probe')
        f.need(not torch.cuda.is_initialized(), 'Unexpected CUDA context')
        plan['status']='technical_passed_visual_review_pending'
        plan['technical_completed_at_taiwan']=now()
        write(OUT/'plan.json',plan)
        cards = []
        for result in results:
            i=result['index']
            cards.append(f'<article id="case{i:02d}"><h2>{i:02d} {escape(result["video_id"])}</h2>'
                f'<p><a href="{i:02d}/comparison.png">四路關鍵影格</a> · <a href="{i:02d}/head_native.png">原片前15幀</a> · '
                f'<a href="{i:02d}/tail_native.png">原片後15幀</a> · <a href="{i:02d}/crop_overlay.png">原圖裁切範圍</a></p>'
                f'<img src="{i:02d}/comparison.png" loading="lazy" alt="四路對照">'
                + '<details><summary>各版本全部50幀</summary>'
                + ''.join(f'<p>{name}</p><img loading="lazy" src="{i:02d}/{name}.png" alt="{name}50幀">' for name in VARIANTS)
                + '</details></article>')
        (OUT/'index.html').write_text('<!doctype html><html lang="zh-Hant"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>固定16支時間／裁切資訊保留檢查</title><style>'
            'body{font:16px/1.7 system-ui;max-width:1280px;margin:24px auto;padding:0 16px;color:#20242b}'
            'img{max-width:100%;height:auto}article{border-top:1px solid #ccc;margin-top:24px}'
            '.notice{background:#fff3d6;padding:12px}summary{cursor:pointer}</style>'
            '<h1>固定16支：時間／裁切資訊保留檢查</h1>'
            '<p class="notice">所有版本皆50幀RGB、160×120。base＝原基準；space＝四周各裁5%；time＝頭尾各去掉約5%；both＝兩者合併。'
            '四欄同一slot在時間裁切後可能對應不同來源幀，請看標示的f編號；時間是索引／名義FPS。放大僅供觀看。'
            '這是看過樣本後的探索性資訊保留檢查，不是盲式標籤審查或準確率比較。未啟動訓練，未讀validation／test。</p>'
            + '<p><a href="decision.json">最終採用判斷</a> · <a href="plan.json">規則與驗收條件</a></p>'
            + ''.join(cards)+'</html>\n')
        print(f'OUTPUT: {OUT}',flush=True)
        return OUT
    except Exception as error:
        write(OUT/'failure.json',dict(status='stopped_on_problem',error_type=type(error).__name__,error=str(error),
                                      completed=len(results),at_taiwan=now()))
        raise


if __name__=='__main__':
    build()
