"""Frozen v5 -> independent three-window contract test. Offline eval only."""
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import argparse
import html
import shutil
import tempfile

import cv2
import torch

from development.data_pipeline.pipeline import load_gate_config
from development.data_pipeline.multiwindow_v1 import BasketballThreeWindowDataset,select_covering_windows
from development.mixnet_i3d.extract_cache_v1 import tensor_sha
from development.partial_finetune_v1.initialization import load_initialized_model,EXTRACTOR,HEAD
from development.yolo_train_probe_v1.probe import read,save,sha,require
from .probe import ROOT,FOLDER,ARTIFACTS
from .selector_v5 import ExposurePolicyV5
from .window_adapter_v1 import IntervalRequest
from .window_adapter_v5 import V5DiagnosticThreeWindowAdapter
from .frozen_probe_v2 import decode_new
from .smoke_windows_v1 import independent_rgb_check,run_forward,panel,preview_window,cache_inventory

PARENT=ARTIFACTS/'local_exposure_v5_of7zq9p2'
CASE_IDS=('T004','Y012','Y019','T001','T008','S026','S021','S030')
HUMAN_Y012=ARTIFACTS/'temporal_probe_v2_xafv9o3e/inputs/Y012_human_target_clarification.json'
HUMAN_Y019=ARTIFACTS/'event_retention_v1_vqoqjui0/human_Y019_start_uncertainty.json'


def make_plan():
    prior=read(PARENT/'report.json')
    require(prior['status']=='completed' and all(prior['regression_checks'].values()),'v5 regression not complete')
    require(asdict(ExposurePolicyV5())==read(PARENT/'policy_v5.json'),'Frozen v5 policy changed')
    for name in ('selector.py','selector_v2.py','selector_v3.py','selector_v4.py','selector_v5.py'):
        require(sha(FOLDER/name)==sha(PARENT/'inputs'/name),'Frozen source changed: '+name)
    target=read(HUMAN_Y012);uncertain=read(HUMAN_Y019)
    require(target['sample_id']=='Y012' and target['target_event_identity_human_confirmed'], 'Missing human target identity')
    require(uncertain['normalized_answer']=='uncertain','Human uncertainty changed')
    by={r['sample_id']:r for r in prior['videos']};require(set(CASE_IDS)<=by.keys(),'Missing fixed cases')
    requests={
        'T004':IntervalRequest((0,146),provenance='Full-source normal identity control'),
        'Y012':IntervalRequest((37,142),review_scope='explicit_input_smoke_only',target_identity_confirmed=True,
            event_completeness_verified=False,provenance='Previously reviewed coordinates and human later-FT identity; mechanical candidate test ONLY, not event/result truth'),
        'Y019':IntervalRequest((0,89),target_identity_confirmed=True,uncertainty_reasons=('source_start_uncertain',),
            provenance='Preserve direct human uncertainty at source start'),
        'T001':IntervalRequest(uncertainty_reasons=('multiple_events_target_unconfirmed',),
            provenance='Replay plus later court; never select by length'),
        'T008':IntervalRequest((140,146),uncertainty_reasons=('short_closeup_negative_control_not_target',),
            provenance='Existing6-frame closeup for short-input fallback, not a proposed shot'),
        'S026':IntervalRequest(uncertainty_reasons=('v5_local_exposure_recovery_unconfirmed',),
            provenance='Frame77 pulse not confirmed recovered; retain original policy'),
        'S021':IntervalRequest((0,146),provenance='v5 exposure-wing fix full-source identity; no crop needed'),
        'S030':IntervalRequest((34,146),uncertainty_reasons=('short_earlier_camera_view_may_contain_release',),
            provenance='Later long view is not proof of complete event; preserve earlier34 frames'),
    }
    cases=[by[s] for s in CASE_IDS]
    plan={'source_v5_run':str(PARENT),'source_v5_report_sha256':sha(PARENT/'report.json'),
          'cases':[{k:r[k] for k in ('sample_id','video_id','source_group','frame_count')}|
                   {'request':asdict(requests[r['sample_id']])} for r in cases],
          'thresholds_frozen':True,'new_semantic_or_human_event_annotation_added':False,
          'Y012_scope':'Explicit nonidentity input branch only; not complete-event/result verification or production crop adoption',
          'classification_accuracy_computed':False,'training_started':False,'production_new_crop_adoptions':0}
    return prior,cases,requests,plan


def render(output,rows):
    cards=[]
    for r in rows:
        d=r['adaptation'];sid=r['sample_id']
        media=lambda paths:''.join(f'<video controls preload="metadata" src="{p}"></video>' for p in paths)
        cards.append(f'<section id="{sid}"><h2>{sid}：{d["mode"]}</h2><p>{html.escape(r["video_id"])}；標籤：{r["binary_label"]}<br>請求區間：{d["originating_request"]["interval"]}；有效區間：{d["effective_interval"]}<br>原因：{html.escape(d["reason"])}<br>原窗口：{r["original_windows"]}<br>轉接窗口：{r["adapted_windows"]}<br>輸入逐值相同：{r["inputs_bitwise_equal"]}；回退沿用同一tensor：{d["fallback_same_tensor_object"]}</p><p>輸入(3,3,64,192,256)，RGB float32；特徵(1,3,8,1536)、logits(1,)皆有限，僅核對契約，不評分。完整事件確認：false；production裁段採用：false。</p><h3>原三窗口</h3><img src="{r["panels"][0]}"><div>{media(r["original_window_videos"])}</div><h3>轉接三窗口</h3><img src="{r["panels"][1]}"><div>{media(r["adapted_window_videos"])}</div></section>')
    page='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>固定v5→三窗口：獨立轉接測試</title><style>body{font:18px system-ui;line-height:1.8;max-width:1250px;margin:30px auto;padding:0 20px}img{max-width:100%}video{width:360px;max-width:100%}section{border-top:1px solid #ccd;padding:20px 0}section div{display:flex;gap:16px;flex-wrap:wrap}.warn{background:#fff1ce;padding:18px}</style><h1>固定v5→三窗口：只測輸入契約</h1><p class="warn">Y012的[37,142)只作非identity的工程測試；雖先前人確認目標是後段罰球，不代表完整出手與結果、端點已成為真值或可採用訓練。其他7筆原輸入不變。多事件、S026恢復未確認、Y019開頭未知、短片段與S030可能丟掉早段出手的情況均回退，不選最長場景。</p><p>獨立train-only轉接器使用凍結v5紀錄，裁段請求必須精確對應其乾淨場景提案。保留原資料載入器與舊v4轉接器；沿用已測過的絕對座標解碼／RGB空間轉換機制，不插幀、不跨鏡拼接。每個原／新窗口全部64幀與全部像素另以原生BGR→RGB核對；原片座標與標籤不變，stem hook核對只做一次mean/std。</p><p>既有權重離線載入、全凍結eval／inference_mode。只核對形狀與有限值，未比較準確率／裁剪效果；不寫特徵快取、不訓練、不開分類validation/test、不改TANet。這不批准全資料自動裁片。影片只是模型輸入的有損64幀預覽，回退兩側共用相同預覽。</p><p><a href="plan.json">固定案例與請求來源</a> · <a href="report.json">逐筆座標／RGB／標籤／正規化／前向檢查</a></p>'''+''.join(cards)+'</html>'
    (output/'index.html').write_text(page,encoding='utf-8')


def execute(dry_run=False):
    prior,cases,requests,plan=make_plan();config=load_gate_config();ids=[c['video_id'] for c in cases]
    baseline=BasketballThreeWindowDataset(config,split='train',video_ids=ids,epoch=0)
    adapter=V5DiagnosticThreeWindowAdapter(config,split='train',video_ids=ids,epoch=0,
        v5_records={c['video_id']:c for c in cases},source_report_sha256=plan['source_v5_report_sha256'],
        requests={c['video_id']:requests[c['sample_id']] for c in cases})
    if dry_run:return {'case_ids':list(CASE_IDS),'source_video_opens':0,'files_written':False,'forward_started':False,
                       'source_v5_report_sha256':plan['source_v5_report_sha256'],'production_new_crop_adoptions':0}
    require(torch.cuda.is_available(),'CUDA unavailable; do not silently change runtime')
    output=Path(tempfile.mkdtemp(prefix='adapter_smoke_v5_',dir=ARTIFACTS))
    for name in ('inputs','previews','panels'):(output/name).mkdir()
    protected=dict(prior['protected_sha256'])
    for p in (PARENT/'report.json',PARENT/'policy_v5.json',PARENT/'AI_review.json',PARENT/'verification.json',EXTRACTOR,HEAD,HUMAN_Y012,HUMAN_Y019):
        protected[str(p)]=sha(p)
    for name in ('window_adapter_v1.py','window_adapter_v5.py','smoke_windows_v1.py','smoke_windows_v5.py','selector_v5.py'):
        p=FOLDER/name;protected[str(p)]=sha(p);shutil.copy2(p,output/'inputs'/name)
    for p in (FOLDER/'tests/test_window_adapter_v5.py',ROOT/'development/data_pipeline/multiwindow_v1.py',
              ROOT/'development/data_pipeline/pipeline.py',ROOT/'development/data_quality/transforms/video_clip_transforms.py'):
        protected[str(p)]=sha(p);shutil.copy2(p,output/'inputs'/p.name)
    shutil.copy2(HUMAN_Y012,output/'inputs/Y012_human_identity_only.json')
    shutil.copy2(HUMAN_Y019,output/'inputs/Y019_human_uncertainty.json')
    save(output/'plan.json',plan);save(output/'policy_v5.json',asdict(ExposurePolicyV5()))
    cache_before=cache_inventory();save(output/'inputs/cache_inventory_before.json',cache_before)
    save(output/'run_state.json',{'status':'running','samples_completed':0});print('OUTPUT:',output,flush=True)
    torch.set_num_threads(4);cv2.setNumThreads(1);torch.manual_seed(20261028)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    rows=[]
    try:
        model,provenance=load_initialized_model();model.requires_grad_(False).eval();model=model.cuda()
        before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};torch.cuda.reset_peak_memory_stats()
        for index,case in enumerate(cases):
            sid=case['sample_id'];old=baseline[index];new=adapter[index];d=new['adaptation']
            require(old['video_id']==new['video_id']==case['video_id'],'Video ID changed')
            require(old['binary_label']==new['binary_label']==int(baseline.base.rows[index]['binary_label']),'Label changed')
            require(old['frame_count']==new['frame_count']==case['frame_count'],'Source count changed')
            require(old['windows']==select_covering_windows(case['frame_count']),'Original window policy changed')
            require(d['source_report_sha256']==plan['source_v5_report_sha256'],'Wrong v5 provenance')
            frames,fps,identity=decode_new(baseline.base.rows[index],baseline.base.guard)
            require(identity==case['source_identity'],'Source identity changed')
            independent_rgb_check(old,frames);independent_rgb_check(new,frames)
            same=torch.equal(old['tensor'],new['tensor']);candidate=d['mode']=='diagnostic_candidate'
            if not candidate:
                require(same and d['fallback_same_tensor_object'],'Fallback tensor changed')
                for k,v in old.items():require(torch.equal(v,new[k]) if torch.is_tensor(v) else v==new[k],'Original field changed: '+k)
            else:
                require(sid=='Y012' and not same and d['requested_interval_matches_frozen_clean_span'],'Unplanned crop branch')
                a,b=d['effective_interval']
                require({f for x,y in new['windows'] for f in range(x,y)}==set(range(a,b)),'Crop coverage incomplete')
                require(not d['event_completeness_verified'] and not d['production_crop_adopted'],'Candidate promoted to truth')
            original_forward=run_forward(model,old,provenance,'cuda');adapted_forward=run_forward(model,new,provenance,'cuda')
            if not candidate:require(original_forward==adapted_forward,'Identical fallback inputs changed outputs')
            panels=[]
            for name,item in (('original',old),('adapted',new)):
                rel=f'panels/{sid}_{name}.png';panel(item,output/rel,f'{sid} {name} RGB input / fixed v5');panels.append(rel)
            original_videos=[];adapted_videos=[]
            for wi in range(3):
                rel=f'previews/{sid}_original_w{wi}.mp4';preview_window(old['tensor'][wi],output/rel,fps);original_videos.append(rel)
                if same:adapted_videos.append(rel)
                else:
                    rel=f'previews/{sid}_adapted_w{wi}.mp4';preview_window(new['tensor'][wi],output/rel,fps);adapted_videos.append(rel)
            rows.append({'sample_id':sid,'video_id':case['video_id'],'source_group':case['source_group'],
                'binary_label':old['binary_label'],'source_frame_count':old['frame_count'],'source_identity':identity,
                'original_windows':old['windows'],'adapted_windows':new['windows'],'adaptation':d,
                'input_shape':list(new['tensor'].shape),'input_dtype':'float32','input_color':'RGB',
                'original_input_sha256':tensor_sha(old['tensor']),'adapted_input_sha256':tensor_sha(new['tensor']),
                'inputs_bitwise_equal':same,'independent_native_rgb_all_pixels_verified':True,
                'original_forward':original_forward,'adapted_forward':adapted_forward,
                'panels':panels,'original_window_videos':original_videos,'adapted_window_videos':adapted_videos,
                'production_crop_adopted':False,'quality_weighting_used':False})
            save(output/'run_state.json',{'status':'running','samples_completed':len(rows)})
            print(sid,d['mode'],d['reason'],'input_equal=',same,'RGB/label/normalization/forward: PASS',flush=True)
            del old,new,frames
        require(all(torch.equal(before[k],v.detach().cpu()) for k,v in model.state_dict().items()),'Model state changed')
        require(all(p.grad is None and not p.requires_grad for p in model.parameters()),'Gradients unexpectedly enabled')
        require(all(not m.training for m in model.modules()),'Module left in train mode')
        require(all(sha(p)==h for p,h in protected.items()),'Protected source/model/history changed')
        cache_after=cache_inventory();require(cache_before==cache_after,'Original cache inventory changed')
        ledgers=[baseline.base.ledger.statistics(),adapter.original.base.ledger.statistics()]
        access={s:sum(v[s]['opens'] for v in ledgers) for s in ('train','validation','test')}
        require(access=={'train':59,'validation':0,'test':0},'Unexpected original source access')
        checks={'frozen_v5_source_and_policy_verified':True,'absolute_coordinates_contiguous64_and_coverage':True,
                'all_original_and_adapted_pixels_match_independent_RGB':True,'labels_unchanged':True,
                'fallback_identity_input_and_output_exact':True,'single_mean_std_normalization':True,
                'finite_feature_and_logit_shapes':True,'model_parameters_and_buffers_unchanged':True,
                'old_feature_cache_inventory_unchanged':True,'candidate_not_promoted_to_training_truth':True}
        report={'status':'completed','schema':'frozen_v5_three_windows_input_smoke_v1','rows':rows,'checks':checks,
                'mode_counts':dict(Counter(r['adaptation']['mode'] for r in rows)),
                'source_v5_run':str(PARENT),'source_v5_report_sha256':plan['source_v5_report_sha256'],
                'protected_sha256':protected,'model':provenance,'model_precision':'float32_no_amp_no_tf32',
                'model_peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                'device':torch.cuda.get_device_name(0),'forward_pairs':8,'actual_forwards':16,
                'cache_inventory_before':cache_before,'cache_inventory_after':cache_after,'original_video_opens':access,
                'classification_accuracy_computed':False,'optimizer_created':False,'training_started':False,
                'feature_tensors_cached':False,'new_model_modules_added':False,'TANet_modified':False,
                'production_new_crop_adoptions':0,'diagnostic_nonidentity_candidate_tests':1,
                'complete_target_event_and_result_verified':False,'selection_thresholds_modified':False}
        save(output/'report.json',report);render(output,rows)
        save(output/'run_state.json',{'status':'completed','samples_completed':8,'checks':checks})
        print('ALL v5 INPUT CONTRACTS PASS; no crop adoption, training, or cache writes',flush=True);return output
    except BaseException as exc:
        save(output/'run_state.json',{'status':'failed','samples_completed':len(rows),'error':f'{type(exc).__name__}: {exc}'})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dry-run',action='store_true')
    args=p.parse_args();print(execute(args.dry_run))
