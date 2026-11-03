"""Fixed v4 -> three-window input smoke; original train only, offline eval, no cache writes."""
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
import argparse
import hashlib
import html
import json
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
import torch

from development.data_pipeline.pipeline import load_gate_config
from development.data_pipeline.multiwindow_v1 import BasketballThreeWindowDataset, select_covering_windows
from development.partial_finetune_v1.initialization import load_initialized_model, EXTRACTOR, HEAD
from development.mixnet_i3d.extract_cache_v1 import tensor_sha
from development.yolo_train_probe_v1.probe import read, save, sha, require
from .probe import ROOT, FOLDER, ARTIFACTS
from .frozen_probe_v2 import decode_new
from .selector_v4 import BoundaryPolicy
from .window_adapter_v1 import DiagnosticThreeWindowAdapter, IntervalRequest

PARENT=ARTIFACTS/'boundary_refine_v4_se8f2bo6'
CACHE=ROOT/'development/mixnet_i3d/artifacts/feature_cache_v1/67b6400e0982bb6d'
CASE_IDS=('T004','Y012','Y019','T001','T008')


def cache_inventory():
    rows=[]
    for p in sorted(CACHE.rglob('*')):
        if p.is_file():
            s=p.stat();rows.append((str(p.relative_to(CACHE)),s.st_size,s.st_mtime_ns))
    require(rows,'Missing original cache inventory')
    return {'file_count':len(rows),'metadata_sha256':hashlib.sha256(json.dumps(rows).encode()).hexdigest()}


def make_plan():
    report=read(PARENT/'report.json');require(report['status']=='completed','Incomplete fixed v4 run')
    require(asdict(BoundaryPolicy())==read(PARENT/'policy.json'),'v4 settings changed')
    for name in ('selector.py','selector_v2.py','selector_v3.py','selector_v4.py'):
        require(sha(FOLDER/name)==sha(PARENT/'inputs'/name),'Frozen selection source changed: '+name)
    by={r['sample_id']:r for r in report['videos']}
    require(set(CASE_IDS).issubset(by),'Missing fixed diagnostic cases')
    clarification=read(ARTIFACTS/'temporal_probe_v2_xafv9o3e/inputs/Y012_human_target_clarification.json')
    require(clarification['sample_id']=='Y012' and clarification['target_event_identity_human_confirmed'],
            'Y012 target identity confirmation missing')
    uncertain=read(ARTIFACTS/'event_retention_v1_vqoqjui0/human_Y019_start_uncertainty.json')
    require(uncertain['normalized_answer']=='uncertain','Y019 human uncertainty changed')
    requests={
        'T004':IntervalRequest((0,146),provenance='unchanged full-source normal control; no target selection'),
        'Y012':IntervalRequest((37,142),review_scope='explicit_input_smoke_only',target_identity_confirmed=True,
            event_completeness_verified=False,provenance='human target identity + AI endpoint review; candidate mechanics only, NOT human full-event truth'),
        'Y019':IntervalRequest((0,89),target_identity_confirmed=True,uncertainty_reasons=('source_start_uncertain',),
            provenance='direct user uncertain at source start; preserve original pipeline'),
        'T001':IntervalRequest(uncertainty_reasons=('multiple_events_target_unconfirmed',),
            provenance='replay plus graphic plus later court; never choose longest scene'),
        'T008':IntervalRequest((140,146),uncertainty_reasons=('known_closeup_not_target',),
            provenance='deliberate negative-control request from real6-frame short closeup; not selected shot'),
    }
    require(any(s['start_frame']==37 and s['end_frame_exclusive']==142 for s in by['Y012']['v4']['contiguous_scene_candidates']),
            'Y012 requested candidate changed')
    require(any(s['start_frame']==140 and s['end_frame_exclusive']==146 for s in by['T008']['v4']['all_segments_including_short']),
            'Short negative control changed')
    return report,[by[s] for s in CASE_IDS],requests


def preview_window(tensor,path,fps):
    require(not path.exists(),'Do not overwrite window preview')
    rgb=np.rint(tensor.permute(1,2,3,0).numpy()*255).clip(0,255).astype(np.uint8)
    n,h,w,_=rgb.shape;require(n==64,'Preview must contain one contiguous window')
    p=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}',
        '-r',fps,'-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','veryfast','-crf','20',
        '-pix_fmt','yuv420p','-movflags','+faststart','-n',str(path)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        p.stdin.write(rgb.tobytes());p.stdin.close();error=p.stderr.read().decode(errors='replace')
        require(p.wait(timeout=30)==0,'Preview encode failed: '+error[-400:])
    finally:
        if p.poll() is None:p.kill();p.wait()
    cap=cv2.VideoCapture(str(path))
    try:
        require(cap.isOpened() and round(cap.get(cv2.CAP_PROP_FRAME_COUNT))==64,'Preview count changed')
        require(abs(cap.get(cv2.CAP_PROP_FPS)-float(Fraction(fps)))<1e-4,'Preview FPS changed')
    finally:cap.release()


def panel(item,path,title):
    tiles=[]
    for i,(a,b) in enumerate(item['windows']):
        row=[]
        for j in (0,31,63):
            rgb=np.rint(item['tensor'][i,:,j].permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8)
            bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            tile=np.zeros((216,256,3),np.uint8);tile[24:]=bgr
            cv2.putText(tile,f'w{i} [{a},{b}) f{a+j}',(4,17),cv2.FONT_HERSHEY_SIMPLEX,.39,(255,255,255),1)
            row.append(tile)
        tiles.append(np.hstack(row))
    header=np.zeros((32,768,3),np.uint8)
    cv2.putText(header,title,(5,22),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1)
    require(cv2.imwrite(str(path),np.vstack([header,*tiles])),'Input panel failed')


def independent_rgb_check(item,frames):
    """Independent pixel path: native BGR -> explicit RGB -> frozen resize/float.

    Does not reuse the adapter decoder or ClipTransform. All64 indices checked
    in every window, including channel order and spatial conversion.
    """
    require(tuple(item['tensor'].shape)==(3,3,64,192,256),'Input shape changed')
    require(item['tensor'].dtype==torch.float32 and torch.isfinite(item['tensor']).all().item(),'Invalid input values')
    require(0<=float(item['tensor'].min())<=float(item['tensor'].max())<=1,'Input not unit range')
    for wi,(a,b) in enumerate(item['windows']):
        require(0<=a<b<=len(frames) and b-a==64,'Out-of-source or non64 window')
        expected=[]
        for raw in frames[a:b]:
            rgb=cv2.cvtColor(raw,cv2.COLOR_BGR2RGB)
            expected.append(cv2.resize(rgb,(256,192),interpolation=cv2.INTER_AREA).astype(np.float32)/255.)
        manual=torch.from_numpy(np.ascontiguousarray(np.stack(expected).transpose(3,0,1,2)))
        require(torch.equal(manual,item['tensor'][wi]),'Independent native RGB/window/resize mismatch')


def run_forward(model,item,provenance,device):
    x=item['tensor'].unsqueeze(0).to(device)
    mean=torch.tensor(provenance['mean'],dtype=torch.float32,device=device).view(1,1,3,1,1,1)
    std=torch.tensor(provenance['std'],dtype=torch.float32,device=device).view(1,1,3,1,1,1)
    expected=((x-mean)/std).reshape(3,3,64,192,256)
    checks=[]
    def normalization_hook(module,args):
        require(torch.equal(args[0],expected),'Model does not receive exact single RGB mean/std normalization')
        checks.append(True)
    handle=model.backbone.conv_stem.register_forward_pre_hook(normalization_hook)
    try:
        with torch.inference_mode():
            features=model.forward_features(x);logits=model.head(features)
            require(tuple(features.shape)==(1,3,8,1536) and torch.isfinite(features).all().item(),'Invalid feature output')
            require(tuple(logits.shape)==(1,) and torch.isfinite(logits).all().item(),'Invalid logit output')
            require(not features.requires_grad and not logits.requires_grad,'Gradient graph unexpectedly created')
            result={'feature_shape':list(features.shape),'logits_shape':list(logits.shape),
                    'features_finite':True,'logits_finite':True,'single_normalization_verified':checks==[True],
                    'feature_sha256':tensor_sha(features[0].cpu().contiguous()),'logit':float(logits[0].cpu())}
            require(result['single_normalization_verified'],'Missing normalization check')
            return result
    finally:handle.remove()


def execute(dry_run=False):
    prior,cases,requests=make_plan();config=load_gate_config();ids=[c['video_id'] for c in cases]
    # Both objects still use the unchanged original data loader/decoder/transform.
    baseline=BasketballThreeWindowDataset(config,split='train',video_ids=ids,epoch=0)
    adapter=DiagnosticThreeWindowAdapter(config,split='train',video_ids=ids,
        requests={c['video_id']:requests[c['sample_id']] for c in cases},epoch=0)
    plan={'cases':[{k:c[k] for k in ('sample_id','video_id','source_group','frame_count')}|
                  {'request':asdict(requests[c['sample_id']])} for c in cases],
          'candidate_scope':'Explicit input-mechanics smoke only, not human complete-event truth or training approval',
          'selection_thresholds_frozen':True,'classification_accuracy_computed':False,'training_started':False}
    if dry_run:return {**plan,'source_opens':0,'writes':False,'forward_started':False}
    require(torch.cuda.is_available(),'CUDA unavailable: no silent runtime/device change')
    output=Path(tempfile.mkdtemp(prefix='adapter_smoke_v1_',dir=ARTIFACTS))
    for name in ('inputs','previews','panels'):(output/name).mkdir()
    protected={**prior['protected_sha256'],str(PARENT/'report.json'):sha(PARENT/'report.json'),
               str(PARENT/'policy.json'):sha(PARENT/'policy.json'),str(EXTRACTOR):sha(EXTRACTOR),str(HEAD):sha(HEAD)}
    for p in [FOLDER/'window_adapter_v1.py',FOLDER/'smoke_windows_v1.py',
              ROOT/'development/data_pipeline/pipeline.py',ROOT/'development/data_pipeline/multiwindow_v1.py',
              ROOT/'development/data_quality/transforms/video_clip_transforms.py']:
        protected[str(p)]=sha(p);shutil.copy2(p,output/'inputs'/p.name)
    for sid,p in [('Y012',ARTIFACTS/'temporal_probe_v2_xafv9o3e/inputs/Y012_human_target_clarification.json'),
                  ('Y019',ARTIFACTS/'event_retention_v1_vqoqjui0/human_Y019_start_uncertainty.json')]:
        protected[str(p)]=sha(p);shutil.copy2(p,output/'inputs'/f'{sid}_human_scope.json')
    cache_before=cache_inventory();save(output/'inputs/cache_inventory_before.json',cache_before)
    save(output/'plan.json',plan);save(output/'run_state.json',{'status':'running','samples_completed':0})
    print('OUTPUT:',output,flush=True)
    torch.set_num_threads(4);cv2.setNumThreads(1);torch.manual_seed(20261028)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    model,provenance=load_initialized_model();model.requires_grad_(False).eval();model=model.cuda()
    before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    torch.cuda.reset_peak_memory_stats();rows=[];cards=[]
    try:
        for index,case in enumerate(cases):
            sid=case['sample_id'];original=baseline[index];adapted=adapter[index]
            require(original['video_id']==adapted['video_id']==case['video_id'],'Video identity changed')
            require(original['binary_label']==adapted['binary_label']==int(baseline.base.rows[index]['binary_label']),'Label changed')
            require(original['frame_count']==adapted['frame_count']==case['frame_count'],'Source frame count changed')
            require(original['windows']==select_covering_windows(case['frame_count']),'Original windows changed')
            # Independent full native decode uses the same guarded train allowlist,
            # but not the adapter's RGB decoder/transform path.
            frames,fps,identity=decode_new({**baseline.base.rows[index],'sample_id':sid},baseline.base.guard)
            require(identity==case['source_identity'],'Source identity differs from fixed diagnostic')
            independent_rgb_check(original,frames);independent_rgb_check(adapted,frames)
            same=torch.equal(original['tensor'],adapted['tensor'])
            mode=adapted['adaptation']['mode']
            if mode!='diagnostic_candidate':
                for key,value in original.items():
                    other=adapted[key]
                    require(torch.equal(value,other) if torch.is_tensor(value) else value==other,
                            'Fallback/identity altered an original field: '+key)
                require(same,'Fallback input differs bitwise')
            else:
                require(not same,'Crop candidate unexpectedly identical to original')
                a,b=adapted['adaptation']['effective_interval']
                require({i for x,y in adapted['windows'] for i in range(x,y)}==set(range(a,b)),'Selected interval incompletely covered')
            old_forward=run_forward(model,original,provenance,'cuda')
            new_forward=run_forward(model,adapted,provenance,'cuda')
            if mode!='diagnostic_candidate':
                require(old_forward==new_forward,'Equal fallback inputs did not reproduce outputs')
            panels=[]
            for name,item in (('original',original),('adapted',adapted)):
                rel=f'panels/{sid}_{name}.png';panel(item,output/rel,f'{sid} {name} RGB model input');panels.append(rel)
            old_videos=[];new_videos=[]
            for wi in range(3):
                rel=f'previews/{sid}_original_w{wi}.mp4';preview_window(original['tensor'][wi],output/rel,fps);old_videos.append(rel)
                if same:new_videos.append(rel)
                else:
                    rel=f'previews/{sid}_adapted_w{wi}.mp4';preview_window(adapted['tensor'][wi],output/rel,fps);new_videos.append(rel)
            row={'sample_id':sid,'video_id':case['video_id'],'source_group':case['source_group'],
                 'binary_label':original['binary_label'],'source_frame_count':case['frame_count'],'fps_rational':fps,
                 'source_identity':identity,'original_windows':original['windows'],'adapted_windows':adapted['windows'],
                 'adaptation':adapted['adaptation'],'input_shape':list(adapted['tensor'].shape),
                 'input_dtype':'float32','input_layout':'windows,C,T,H,W','input_color':'RGB','input_range':[float(adapted['tensor'].min()),float(adapted['tensor'].max())],
                 'original_input_sha256':tensor_sha(original['tensor']),'adapted_input_sha256':tensor_sha(adapted['tensor']),
                 'inputs_bitwise_equal':same,'independent_native_rgb_all_pixels_verified':True,
                 'original_forward':old_forward,'adapted_forward':new_forward,'panels':panels,
                 'original_window_videos':old_videos,'adapted_window_videos':new_videos,
                 'quality_weighting_used':False,'training_use_approved':False}
            rows.append(row)
            videos=lambda paths:''.join(f'<video controls preload="metadata" src="{p}"></video>' for p in paths)
            cards.append(f'<section id="{sid}"><h2>{sid}：{mode}</h2><p>原因：{html.escape(adapted["adaptation"]["reason"])}<br>原窗口：{original["windows"]}<br>轉接窗口：{adapted["windows"]}；bitwise相同：{same}</p><p>模型輸入 {tuple(adapted["tensor"].shape)}；特徵(1,3,8,1536)、logits(1,)皆有限。不計準確率，不判新舊誰較好。</p><h3>原三窗口</h3><img src="{panels[0]}"><div>{videos(old_videos)}</div><h3>轉接三窗口</h3><img src="{panels[1]}"><div>{videos(new_videos)}</div></section>')
            save(output/'run_state.json',{'status':'running','samples_completed':len(rows)})
            print(sid,mode,adapted['adaptation']['reason'],'input_equal=',same,'RGB/windows/normalization/forward: PASS',flush=True)
            del original,adapted,frames
        require(all(torch.equal(before[k],v.detach().cpu()) for k,v in model.state_dict().items()),'Model parameters or buffers changed')
        require(all(p.grad is None and not p.requires_grad for p in model.parameters()),'Unexpected gradients/trainable parameters')
        require(all(not mod.training for mod in model.modules()),'Module left in train mode')
        require(all(sha(p)==h for p,h in protected.items()),'Protected source/model/history changed')
        cache_after=cache_inventory();require(cache_after==cache_before,'Original feature-cache metadata inventory changed')
        stats=[baseline.base.ledger.statistics(),adapter.original.base.ledger.statistics()]
        access={split:sum(s[split]['opens'] for s in stats) for split in ('train','validation','test')}
        require(access['validation']==access['test']==0,'Unexpected nontrain original access')
        checks={'absolute_window_coordinates_and_coverage':True,'all_pixels_rgb_reference_match':True,
                'fallback_and_identity_inputs_bitwise_equal':all(r['inputs_bitwise_equal'] for r in rows if r['adaptation']['mode']!='diagnostic_candidate'),
                'single_mean_std_normalization':all(r[k]['single_normalization_verified'] for r in rows for k in ('original_forward','adapted_forward')),
                'finite_feature_and_logit_shapes':True,'model_parameters_buffers_unchanged':True,
                'original_feature_cache_inventory_unchanged':True,'no_automatic_training_approval':True}
        report={'status':'completed','rows':rows,'checks':checks,'model':provenance,
                'precision':'float32_no_amp_no_tf32','device':torch.cuda.get_device_name(0),
                'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                'protected_sha256':protected,'cache_inventory_before':cache_before,'cache_inventory_after':cache_after,
                'original_video_opens':access,'baseline_ledger':stats[0],'adapter_ledger':stats[1],
                'classification_accuracy_computed':False,'optimizer_created':False,'training_started':False,
                'old_feature_cache_written':False,'TANet_read_or_modified_by_this_probe':False,
                'selection_thresholds_modified':False,'new_model_modules_added':False,'automatic_training_use_videos':0,
                'candidate_mechanical_test_not_full_event_truth':True,'forward_pairs':len(rows)}
        save(output/'report.json',report)
        page='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>固定v4→三窗口：工程串接測試</title><style>body{font:18px system-ui;line-height:1.8;max-width:1200px;margin:30px auto;padding:0 20px}img{max-width:100%}video{width:360px;max-width:100%}section{border-top:1px solid #ccd;padding:20px 0}section div{display:flex;gap:16px;flex-wrap:wrap}.warn{background:#fff0cf;padding:18px}</style><h1>固定v4→三窗口：只測輸入契約，不訓練</h1><p class="warn">Y012只作明示的候選座標／輸入分支測試，不代表完整事件已確認或批准訓練。Y019保留開頭不確定；T001多事件不選最長；T008的6幀特寫是刻意給入的短片段負例、不是選定投籃。這些均回退原政策。T004整片不變作正常對照。</p><p>原資料載入器、v4與模型均未修改。回退先呼叫原流程並保留其所有原欄位／tensor；另一條原流程獨立產生參考，要求逐值相同。另以原生BGR解碼→明示RGB→相同空間縮放，核對每窗口全部64幀、全部像素；送入既有MixNet-I3D＋Transformer前也核對只做一次mean/std。</p><p>用既有凍結擷取器與seed20261024 head，離線載入、全部eval、inference_mode，無新增模型模組／optimizer。只看形狀與有限值，不計分類準確率或比較預測優劣。未讀／改TANet、未寫原快取或歷史訓練目錄，分類validation/test原片開啟0。</p><p><a href="plan.json">固定案例與試驗範圍</a> · <a href="report.json">逐筆契約／輸出檢查／SHA</a></p><p>圖中每列為一個連續64幀窗口，展示首／中／末幀；下方影片是模型輸入RGB縮放後的64幀有損預覽，不作訓練或特徵輸入。回退兩側共用相同預覽檔以明示資料一致。</p>'''+''.join(cards)+'</html>'
        (output/'index.html').write_text(page,encoding='utf-8')
        save(output/'run_state.json',{'status':'completed','samples_completed':len(rows),'checks':checks})
        print('ALL CONTRACTS PASS; no training/cache/TANet changes',flush=True);return output
    except BaseException as exc:
        save(output/'run_state.json',{'status':'failed','samples_completed':len(rows),'error':f'{type(exc).__name__}: {exc}'})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dry-run',action='store_true')
    args=p.parse_args();print(execute(args.dry_run))
