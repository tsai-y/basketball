"""Compare exact input RGB at native size; no inference, label edits or test."""
import argparse
from collections import Counter
import hashlib
from html import escape
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import yaml

from development.aca_net_v1.data import VideoScope, Basketball50Dataset, preview_rgb, uniform_indices
from development.aca_review_v1.build import FUSION, SOURCE, verify, source_path
from development.data_pipeline.multiwindow_v1 import select_covering_windows
from development.data_quality.transforms.video_clip_transforms import ClipTransform, derived_seed
from scripts import audit_i3d_temporal_feasibility as gate

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[1]
REVIEW = ROOT / "development/aca_review_v1/artifacts/review24_ckaqp3hd"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def digest_rgb(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def decode_original(scope, video_id):
    _, path = scope.authorize(video_id)
    fd = gate.open_guarded(path)
    cap = None
    try:
        before = identity(os.fstat(fd))
        cap = cv2.VideoCapture(f"/proc/self/fd/{fd}", cv2.CAP_FFMPEG)
        require(cap.isOpened(), "Cannot decode authorized review source")
        raw, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
        require(math.isfinite(raw) and raw >= 66 and abs(raw-round(raw)) < 1e-6 and math.isfinite(fps) and fps > 0,
                "Invalid frame count/FPS")
        frames = []
        for index in range(round(raw)):
            ok, bgr = cap.read()
            require(ok and bgr.dtype == np.uint8 and bgr.shape == (240, 320, 3), f"Invalid original frame {index}")
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        require(not cap.read()[0] and identity(os.fstat(fd)) == before, "Source count/identity changed")
        return np.stack(frames), float(fps)
    finally:
        if cap is not None:
            cap.release()
        os.close(fd)


def reconstruct_mainline(source, windows, raw_config, video_id):
    transform = ClipTransform({k: v for k, v in raw_config["transform"].items() if k != "policy"}, "reference")
    frames = []
    for index, (start, end) in enumerate(windows):
        tensor, _ = transform(source[start:end], split="validation",
            seed=derived_seed(raw_config["seed"], video_id, 0, index, "reference"),
            sample_metadata={"start_frame": start, "end_frame_exclusive": end})
        unit = tensor.transpose(1, 2, 3, 0)
        frames.append(np.rint(unit*255).astype(np.uint8))
    return np.stack(frames)


def timing(n, fps, human_seconds):
    require(type(n) is int and n >= 66 and math.isfinite(fps) and fps > 0, "Invalid timing inputs")
    require(type(human_seconds) in (int, float) and math.isfinite(human_seconds) and human_seconds >= 0,
            "Expected human approximate time")
    aca = list(uniform_indices(n))
    windows = select_covering_windows(n)
    index = max(0, min(n-1, int(math.floor(human_seconds*fps+.5))))
    nearest = min(aca, key=lambda i: (abs(i/fps-human_seconds), i))
    return {"source_frame_count": n, "fps": fps, "last_frame_seconds": (n-1)/fps,
            "human_time_seconds_approximate": human_seconds,
            "human_time_outside_frame_timestamp_range": human_seconds > (n-1)/fps,
            "nearest_source_frame": index, "nearest_source_seconds": index/fps,
            "aca_indices": aca, "aca_times_seconds": [i/fps for i in aca],
            "aca_nearest_frame": nearest, "aca_nearest_time_seconds": nearest/fps,
            "aca_time_distance_seconds": abs(nearest/fps-human_seconds),
            "mainline_windows_half_open": [list(w) for w in windows],
            "mainline_windows_containing_nearest_source_frame": [j+1 for j,(start,end) in enumerate(windows) if start<=index<end],
            "mainline_union_covers_all_source_frames": set(range(n)) == set(i for a,b in windows for i in range(a,b)),
            "semantic_key_event_presence_confirmed": None,
            "warning": "Timestamp coverage does not establish ball/rim visibility, outcome, or target attempt"}


def contact_pages(frames, indices, fps, output, prefix, *, columns, per_page, approximate_time):
    require(len(frames) == len(indices) and len(frames)>0, "Contact sheet length mismatch")
    font = ImageFont.truetype(FONT, 13)
    height, width = frames.shape[1:3]
    pages = []
    for start in range(0, len(frames), per_page):
        end = min(len(frames), start+per_page)
        cols = min(columns, end-start)
        rows = math.ceil((end-start)/cols)
        canvas = Image.new("RGB", (cols*(width+4), rows*(height+26)), "#e8ecf1")
        draw = ImageDraw.Draw(canvas)
        for local, pos in enumerate(range(start,end)):
            x,y = (local%cols)*(width+4), (local//cols)*(height+26)
            canvas.paste(Image.fromarray(frames[pos]), (x+2,y+24))
            t = indices[pos]/fps
            draw.text((x+3,y+3), f"f{indices[pos]:03d}  {t:.3f}s", font=font, fill="#111827")
            if abs(t-approximate_time)<=.5:
                draw.rectangle((x,y,x+width+3,y+height+25),outline="#ba7a13",width=2)
        filename = f"{prefix}_{len(pages)+1:02d}.png"
        canvas.save(output/filename, compress_level=3)
        pages.append({"file":filename,"source_indices":list(indices[start:end]),"native_frame_size":[width,height]})
    return pages


def key_panel(source, aca, main, info, path):
    """Native-sized RGB centered on 320x240 cells; never invent image detail."""
    offsets = (-.25,0,.25)
    fps,n = info["fps"],len(source)
    width,height,caption = 336,240,46
    canvas = Image.new("RGB", (3*width,3*(height+caption)),"#edf0f4")
    draw = ImageDraw.Draw(canvas); font=ImageFont.truetype(FONT,14)
    windows=info["mainline_windows_half_open"]
    for col,offset in enumerate(offsets):
        target=max(0,min((n-1)/fps,info["human_time_seconds_approximate"]+offset))
        index=max(0,min(n-1,int(math.floor(target*fps+.5))))
        aca_pos=min(range(50),key=lambda j:(abs(info["aca_indices"][j]/fps-target),j))
        wi=next(j for j,(a,b) in enumerate(windows) if a<=index<b)
        panels=(("Original 320x240",index,source[index]),
                ("ACA 160x120",info["aca_indices"][aca_pos],aca[aca_pos]),
                (f"Mainline W{wi+1} 256x192",index,main[wi,index-windows[wi][0]]))
        for row,(name,frame,rgb) in enumerate(panels):
            x,y=col*width,row*(height+caption)
            draw.text((x+5,y+4),name,font=font,fill="#111827")
            draw.text((x+5,y+23),f"f{frame:03d}  {frame/fps:.3f}s",font=font,fill="#111827")
            canvas.paste(Image.fromarray(rgb),(x+(width-rgb.shape[1])//2,y+caption+(height-rgb.shape[0])//2))
    canvas.save(path,compress_level=3)


def render_case(case, info, pages):
    blocks=[]
    titles={"source":"原片全部影格（320×240）","aca":"ACA 實際抽樣 50 張（160×120）",
            "window1":"主線視窗1：連續64張（256×192）","window2":"主線視窗2：連續64張（256×192）",
            "window3":"主線視窗3：連續64張（256×192）"}
    for key,items in pages.items():
        images="".join(f"<p>第{i+1}頁；原片影格 {p['source_indices'][0]}–{p['source_indices'][-1]}</p><a href='{p['file']}'><img loading='lazy' src='{p['file']}' alt='{key} page {i+1}'></a>" for i,p in enumerate(items))
        blocks.append(f"<details><summary>{titles[key]}</summary>{images}</details>")
    flags="待複核：標籤或目標投籃尚未確認。" if case["review_id"] in ("R02","R24") else "原標籤與預測保持不變。"
    return f"""<!doctype html><html lang='zh-Hant'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>{case['review_id']} 模型輸入對照</title><style>body{{font:16px/1.65 system-ui;max-width:1100px;margin:24px auto;padding:0 18px;color:#20242a}}img{{max-width:100%;height:auto}}details{{border:1px solid #ccd3dd;padding:12px;margin:16px 0}}summary{{cursor:pointer;font-weight:600}}.notice{{background:#fff5dc;padding:12px}}code{{overflow-wrap:anywhere}}a{{color:#1857a4}}</style>
<a href='../index.html'>回到24部總覽</a><h1>{case['review_id']}　{escape(case['selection_reason'])}</h1>
<p><code>{escape(case['video_id'])}</code>；<a href='{escape(case['source_url'],quote=True)}'>開啟原片</a></p>
<p class='notice'>{flags} 人工時間約 {info['human_time_seconds_approximate']} 秒，不是精確事件標註。橘框只代表影格時間位於該時間前後0.5秒，不代表已確認進／沒進。</p>
<p>使用者紀錄：{escape(case['human_review']['notes'])}</p>
<p>原片 {info['source_frame_count']} 張，FPS={info['fps']:.6f}，最後影格時間 {info['last_frame_seconds']:.3f} 秒。ACA 最接近人工時間的影格：f{info['aca_nearest_frame']}（{info['aca_nearest_time_seconds']:.3f}秒），時間差 {info['aca_time_distance_seconds']:.3f}秒。</p>
<p>主線三視窗（左含右不含）：{escape(str(info['mainline_windows_half_open']))}；原片全部影格皆由至少一個視窗涵蓋：{info['mainline_union_covers_all_source_frames']}。這不代表網路下採樣後仍能辨識球框關係。</p>
<h2>人工時間附近：前0.25秒／當下／後0.25秒</h2><p>各格維持原本像素大小，沒有放大、銳化、裁切ROI或生成影像。點圖可看PNG原尺寸。</p>
<a href='key_comparison.png'><img src='key_comparison.png' alt='native resolution input comparison'></a>
{''.join(blocks)}<p><a href='comparison.json'>完整影格索引、時間、逐像素核對與來源指紋</a>。本頁顯示模型輸入反正規化的RGB，不顯示normalized負值張量；沒有模型推論、標籤修正或重算成績。</p></html>"""


def build():
    require(not torch.cuda.is_initialized(), "CPU-only frame comparison")
    torch.set_num_threads(2); cv2.setNumThreads(1)
    review=read_json(REVIEW/"review_template.json")
    cases=review["cases"]
    require(len(cases)==24 and len({c['video_id'] for c in cases})==24 and all(c['human_review']['reviewed'] for c in cases),
            "Expected completed user first-pass for 24 unique cases")
    summary=read_json(REVIEW/"sampling_summary.json")
    protected=dict(summary['protected_files_sha256'])
    candidate=read_json(FUSION/"candidate.json")
    main_cfg_path=Path(candidate['components']['mainline']['config'])
    main_cfg=read_json(main_cfg_path)
    raw_path=SOURCE/"development/data_pipeline/configs/data_pipeline_gate_v1.yaml"
    require(gate.digest(raw_path)==main_cfg['gate_config_sha256'],"Mainline transform config changed")
    raw_config=yaml.safe_load(raw_path.read_text())
    for relative,expected in main_cfg['source_sha256'].items():
        original=SOURCE/relative
        require(gate.digest(original)==expected,"Mainline source changed")
        protected[str(original)]=expected
    for relative in ('development/data_pipeline/multiwindow_v1.py','development/data_pipeline/pipeline.py',
                     'development/data_quality/transforms/video_clip_transforms.py'):
        require(gate.digest(ROOT/relative)==gate.digest(SOURCE/relative),'Worktree/source transform differs')
        protected[str(ROOT/relative)]=gate.digest(ROOT/relative)
    protected.update({str(p):gate.digest(p) for p in [*REVIEW.iterdir(),*FOLDER.glob('*.py'),main_cfg_path,raw_path]
                      if p.is_file()})
    verify(protected)
    scope=VideoScope(source_root=SOURCE,split='validation',video_ids=[c['video_id'] for c in cases])
    allowed={r['video_id']:r for r in scope.rows}
    for case in cases:
        source_path(case,allowed)
    artifacts=FOLDER/'artifacts';artifacts.mkdir(exist_ok=True)
    output=Path(tempfile.mkdtemp(prefix='frame_compare_',dir=artifacts))
    print(f'OUTPUT={output}',flush=True)
    write_new(output/'provenance.json',{'protected_files_sha256':protected,'review_source':str(REVIEW),
              'selected_ids':[c['video_id'] for c in cases],'case_count':24,'test_videos_opened':0,
              'pixel_comparison':'original loaders vs independent reconstruction, exact uint8 RGB equality',
              'images':'lossless PNG, native frame sizes; annotation outside RGB pixels; no generated content'})
    ordered=sorted(cases,key=lambda c:(c['review_id'] not in ('R02','R24'),c['review_id']))
    results=[]
    try:
        aca_dataset=Basketball50Dataset(source_root=SOURCE,split='validation',video_ids=[c['video_id'] for c in ordered])
        for index,case in enumerate(ordered):
            case_dir=output/case['review_id'];case_dir.mkdir()
            source_file=source_path(case,allowed)
            source_sha=gate.digest(source_file)
            source,fps=decode_original(scope,case['video_id'])
            info=timing(len(source),fps,case['human_review']['key_result_time_seconds'])
            item=aca_dataset[index]
            aca=preview_rgb(item['tensor'])
            indices=info['aca_indices']
            expected=np.stack([cv2.resize(source[i],(160,120),interpolation=cv2.INTER_AREA) for i in indices])
            require(item['video_id']==case['video_id'] and np.array_equal(aca,expected), 'ACA loader pixel parity failed')
            subprocess.run([sys.executable,'-B',str(FOLDER/'mainline_worker.py'),'--review-id',case['review_id'],
                            '--output',str(case_dir)],cwd=SOURCE,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},check=True)
            main_meta=read_json(case_dir/'mainline_dataset.json')
            with np.load(case_dir/'mainline_rgb.npz',allow_pickle=False) as saved:
                main=saved['rgb']
            require(main_meta['frame_count']==len(source) and abs(main_meta['fps']-fps)<1e-9 and
                    main_meta['windows']==info['mainline_windows_half_open'],'Mainline source timing mismatch')
            reconstructed=reconstruct_mainline(source,info['mainline_windows_half_open'],raw_config,case['video_id'])
            require(np.array_equal(main,reconstructed) and digest_rgb(main)==main_meta['display_rgb_sha256'],
                    'Mainline loader pixel parity failed')
            pages={'source':contact_pages(source,list(range(len(source))),fps,case_dir,'original',columns=6,per_page=24,
                                         approximate_time=case['human_review']['key_result_time_seconds']),
                   'aca':contact_pages(aca,indices,fps,case_dir,'aca50',columns=10,per_page=50,
                                      approximate_time=case['human_review']['key_result_time_seconds'])}
            for wi,(a,b) in enumerate(info['mainline_windows_half_open']):
                pages[f'window{wi+1}']=contact_pages(main[wi],list(range(a,b)),fps,case_dir,f'window{wi+1}',columns=8,
                     per_page=32,approximate_time=case['human_review']['key_result_time_seconds'])
            key_panel(source,aca,main,info,case_dir/'key_comparison.png')
            require(gate.digest(source_file)==source_sha,'Original source bytes changed')
            evidence={'review_id':case['review_id'],'video_id':case['video_id'],'split':'validation','timing':info,
                      'original_source_sha256':source_sha,'original_source_unchanged':True,'aca_rgb_sha256':digest_rgb(aca),
                      'mainline_rgb_sha256':digest_rgb(main),'aca_pixel_parity':True,'mainline_pixel_parity':True,
                      'source_frames_rendered':len(source),'aca_frames_rendered':50,'mainline_frames_rendered':[64,64,64],
                      'pages':pages,'human_semantic_checks_modified':False,'model_instantiated':False,
                      'optimizer_updates':0,'test_videos_opened':0}
            write_new(case_dir/'comparison.json',evidence)
            with (case_dir/'index.html').open('x',encoding='utf-8') as stream:
                stream.write(render_case(case,info,pages))
            results.append({'review_id':case['review_id'],'video_id':case['video_id'],'timing':info,
                            'aca_pixel_parity':True,'mainline_pixel_parity':True})
            print(f"case={case['review_id']} done={len(results)}/24 frames={len(source)} fps={fps:.3f} ACA_delta={info['aca_time_distance_seconds']:.3f}s pixel_parity=passed",flush=True)
        verify(protected)
        require(not torch.cuda.is_initialized(),'Unexpected CUDA initialization')
        links=''.join(f"<tr><td><a href='{c['review_id']}/index.html'>{c['review_id']}</a></td><td>{escape(c['selection_reason'])}</td><td>{c['human_review']['key_result_time_seconds']}秒（約）</td><td>{'待複核' if c['review_id'] in ('R02','R24') else '沿用人工紀錄，尚未判讀模型輸入'}</td></tr>" for c in ordered)
        with (output/'index.html').open('x',encoding='utf-8') as stream:
            stream.write(f"<!doctype html><html lang='zh-Hant'><meta charset='utf-8'><title>24部影片：模型輸入抽幀對照</title><style>body{{font:16px/1.65 system-ui;max-width:1050px;margin:28px auto;padding:0 18px}}table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;border-bottom:1px solid #ccd3dd;padding:10px}}a{{color:#1857a4}}</style><h1>24部影片：原片／ACA／主線實際輸入對照</h1><p>全部24部逐像素核對通過；原片全部影格、ACA50張、主線3×64張皆提供無損PNG及影格時間。優先查看R02與R24。</p><p>這是輸入重建與時間對齊，不是模型成績或自動標籤判定；未更動人工紀錄、標籤、模型、validation成績，未讀test。人工時間為約略定位；主線時間涵蓋全片不代表視覺訊息足夠。</p><table><thead><tr><th>案例</th><th>類型</th><th>人工時間</th><th>待處理</th></tr></thead><tbody>{links}</tbody></table><p><a href='report.json'>完整產出與核對報告</a></p></html>")
        report={'status':'completed','case_count':24,'cases':results,'protected_files_unchanged':True,
                'all_actual_loader_pixels_reproduced':True,'all_original_sources_unchanged':True,
                'semantic_event_presence_not_automatically_inferred':True,'human_review_modified':False,
                'labels_modified':False,'validation_scores_recomputed':False,'model_instantiated':False,
                'optimizer_updates':0,'test_videos_opened':0}
        write_new(output/'report.json',report)
        print(json.dumps({'status':'completed','output':str(output),'case_count':24,
                          'all_actual_loader_pixels_reproduced':True,'human_review_modified':False},ensure_ascii=False),flush=True)
    except Exception as exc:
        write_new(output/'failure.json',{'status':'failed','error':repr(exc),'completed_cases':len(results),'no_automatic_retry':True})
        raise
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-comparisons',action='store_true')
    args=parser.parse_args()
    if args.build_comparisons:
        build()
    else:
        print(json.dumps({'mode':'plan_only','review_source':str(REVIEW),'case_count':24,'no_training':True,'no_test':True}))
