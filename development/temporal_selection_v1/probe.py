"""Train-only native-video temporal proposal previews; no training or cache rewrite."""
from collections import Counter
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
import argparse
import html
import json
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

from development.data_pipeline.pipeline import AccessLedger,SplitGuard
from development.data_pipeline.multiwindow_v1 import select_covering_windows
from development.yolo_train_probe_v1.probe import read,save,sha,require
from .selector import Policy,analyze

FOLDER=Path(__file__).resolve().parent
ROOT=FOLDER.parents[1]
ARTIFACTS=FOLDER/'artifacts'
DEFAULT_AUDIT=ROOT/'development/yolo_audit_v1/artifacts/audit_seed20261024_7_aaaj0r'


def select_samples(manifest,allowed):
    require(manifest['split']=='train' and len(manifest['samples'])==20,'Fixed20 original-train diagnostic samples required')
    rows=manifest['samples']
    require(len({r['sample_id'] for r in rows})==20 and len({r['video_id'] for r in rows})==20,'Duplicate samples')
    require(len({r['source_group'] for r in rows})==20,'One video per source required')
    for r in rows:
        require(r['split']=='train' and str(r['excluded'])=='0' and r['video_id'] in allowed,
                'Reject nontrain/excluded ID before any video open')
        official=allowed[r['video_id']]
        require(all(r[k]==official[k] for k in ('video_path','source_group','original_class','binary_label','split')),
                'Sample differs from canonical train manifest')
    return sorted(rows,key=lambda r:r['sample_id'])


def decode(sample,guard):
    before=guard.identity(sample['video_id']);fd=guard.open(sample['video_id'],sample['video_path'],'temporal_scene_proposal_native_decode')
    cap=None;frames=[]
    try:
        cap=cv2.VideoCapture(f'/proc/self/fd/{fd}',cv2.CAP_FFMPEG)
        require(cap.isOpened(),'Cannot open authorized descriptor')
        n=sample['frame_count'];fps=float(Fraction(sample['fps_rational']))
        require(abs(cap.get(cv2.CAP_PROP_FRAME_COUNT)-n)<1e-6 and abs(cap.get(cv2.CAP_PROP_FPS)-fps)<1e-4,'Frame/FPS changed')
        for i in range(n):
            ok,raw=cap.read();require(ok and raw.shape==(240,320,3) and raw.dtype==np.uint8,'Invalid/truncated native frame')
            frames.append(raw)
        require(not cap.read()[0],'Extra decoded frames')
    finally:
        if cap is not None:cap.release()
        os.close(fd)
    after=guard.identity(sample['video_id']);require(before==after,'Original video changed')
    return frames,{'before':before,'after':after}


def encode(frames,path,fps):
    require(frames and not path.exists(),'Empty or existing preview')
    process=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s','320x240',
        '-r',fps,'-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','veryfast','-crf','20',
        '-pix_fmt','yuv420p','-movflags','+faststart','-n',str(path)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        for frame in frames:process.stdin.write(frame.tobytes())
        process.stdin.close();error=process.stderr.read().decode(errors='replace')
        require(process.wait(timeout=30)==0,'Preview encoding failed: '+error[-500:])
    finally:
        if process.poll() is None:process.kill();process.wait()
    cap=cv2.VideoCapture(str(path))
    try:
        require(cap.isOpened() and round(cap.get(cv2.CAP_PROP_FRAME_COUNT))==len(frames),'Preview round trip count mismatch')
        require(abs(cap.get(cv2.CAP_PROP_FPS)-float(Fraction(fps)))<1e-4,'Preview FPS mismatch')
        require(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))==320 and round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))==240,'Preview geometry changed')
    finally:cap.release()


def frame_pages(frames,indices,folder,prefix,marks):
    tiles=[]
    for i in sorted(set(indices)):
        tile=np.zeros((264,320,3),np.uint8);tile[24:]=frames[i]
        label=f'{prefix} frame {i}'+(' CUT?' if i in marks else '')
        cv2.putText(tile,label,(4,17),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1);tiles.append(tile)
    pages=[]
    for start in range(0,len(tiles),8):
        batch=list(tiles[start:start+8])
        while len(batch)<8:batch.append(np.zeros_like(tiles[0]))
        rel=f'{prefix}_{len(pages)+1}.png'
        require(cv2.imwrite(str(folder/rel),np.vstack([np.hstack(batch[:4]),np.hstack(batch[4:])])),'Contact write failed');pages.append(rel)
    return pages


def timeline(analysis,n,path):
    canvas=np.full((160,1200,3),245,np.uint8)
    for m in analysis['per_frame']:
        x=20+round(m['frame_index']*1160/max(1,n-1))
        cv2.line(canvas,(x,95),(x,95-round(m['floor_area']*65)),(60,150,40),1)
    for cut in analysis['cut_frame_candidates']:
        x=20+round(cut*1160/max(1,n-1));cv2.line(canvas,(x,25),(x,105),(20,20,220),2)
    p=analysis['proposed_contiguous_interval']
    if p:
        a=20+round(p['start_frame']*1160/n);b=20+round(p['end_frame_exclusive']*1160/n)
        cv2.rectangle(canvas,(a,112),(b,130),(220,120,30),-1)
    cv2.putText(canvas,'GREEN: wood-like area  RED: possible camera cut  BLUE: unverified interval',(20,17),cv2.FONT_HERSHEY_SIMPLEX,.47,(20,20,20),1)
    cv2.putText(canvas,'frame 0  /  all intervals [start,end)  /  NO semantic shot or result verification',(20,151),cv2.FONT_HERSHEY_SIMPLEX,.45,(20,20,20),1)
    require(cv2.imwrite(str(path),canvas),'Timeline write failed')


def render(output,rows):
    cards=[]
    for row in rows:
        sid=row['sample_id'];p=row['analysis']['proposed_contiguous_interval'];fps=row['fps']
        desc=(f'候選 [{p["start_frame"]}, {p["end_frame_exclusive"]})，{p["length"]}幀；尚未確認投籃與結果' if p else
              '沒有唯一可靠候選；維持原三窗口回退')
        videos=f'<div><label>原片預覽（有損轉碼，只供檢視）<br><video controls preload="metadata" src="{row["original_preview"]}"></video></label>'
        if row['proposal_preview']:videos+=f'<label>連續候選預覽（不跨切鏡拼接）<br><video controls preload="metadata" src="{row["proposal_preview"]}"></video></label>'
        videos+='</div>'
        cuts=row['analysis']['cut_frame_candidates']
        cards.append(f'<section id="{sid}"><h2>{sid} / {html.escape(row["source_group"])}</h2><p>{html.escape(desc)}。原片{row["frame_count"]}幀／{row["frame_count"]/fps:.2f}秒；切鏡候選：{html.escape(str(cuts))}。</p><img src="{row["timeline"]}">{videos}<details><summary>原生逐格總覽／切鏡前後</summary>'+''.join(f'<img loading="lazy" src="{x}">' for x in row['contact_pages'])+'</details></section>')
    page='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>連續片段選擇v1：20支train診斷</title><style>body{font:18px system-ui;line-height:1.8;max-width:1350px;margin:30px auto;padding:0 20px;color:#243042}img{max-width:100%}video{width:480px;max-width:100%;background:#111}section{padding:20px 0;border-top:1px solid #ccd}section div{display:flex;gap:22px;flex-wrap:wrap}h1{font-size:28px}</style><h1>連續片段選擇：先驗證分鏡／球場候選</h1><p>此版本不是YOLO、不是投籃事件辨識器。只用HSV外觀變化、灰階差與木地板色外觀提出切鏡／連續畫面候選；皮膚、圖卡、燈光、快速運鏡皆可能誤導。尚無出手／球到框／結果真值，不能宣稱剪對了。</p><p>20支沿用先前train診斷樣本，不是代表性盲測。固定參數未按模型準確率調整。原始分類validation/test未開啟；沒有重訓、沒改三窗口或特徵快取、沒刪原片。所有候選都必須檢查是否保留出手與結果後才能另立訓練實驗，目前仍回退原三窗口。</p><p>若多段球場畫面或片段不足64幀，標示不確定，不選最長的一段冒稱投籃；不串接不同鏡頭。連續候選保留原長度，尚未決定新的64幀取樣／短片段處理方式。</p><p><a href="report.json">逐幀分數、候選與來源SHA</a> · <a href="human_event_review_draft.json">待確認事件時間點（全數未確認）</a> · <a href="AI_review.html">AI初審（若有）</a></p>'''+''.join(cards)+'</html>'
    (output/'index.html').write_text(page,encoding='utf-8')


def execute(audit=DEFAULT_AUDIT,dry_run=False):
    audit=Path(audit).resolve();require(audit.is_relative_to(ROOT/'development/yolo_audit_v1/artifacts'),'Foreign audit')
    manifest=read(audit/'manifest.json');ledger=AccessLedger();guard=SplitGuard('train',ledger)
    samples=select_samples(manifest,guard.allowed)
    if dry_run:return {'videos':20,'original_video_opens':0,'files_written':False,'training_started':False}
    ARTIFACTS.mkdir(exist_ok=True);output=Path(tempfile.mkdtemp(prefix='temporal_probe_v1_',dir=ARTIFACTS))
    for name in ('previews','contacts','timelines','inputs'):(output/name).mkdir()
    policy=Policy();protected={str(audit/'manifest.json'):sha(audit/'manifest.json')}
    for rel in ('development/data_pipeline/multiwindow_v1.py','development/mixnet_i3d/artifacts/feature_cache_v1/67b6400e0982bb6d/run.pt'):
        p=ROOT/rel;require(p.is_file(),'Missing baseline provenance');protected[str(p)]=sha(p)
    save(output/'inputs/audit_manifest.json',manifest)
    for name in ('selector.py','probe.py'):shutil.copy2(FOLDER/name,output/'inputs'/name)
    save(output/'policy.json',asdict(policy));save(output/'run_state.json',{'status':'running','videos_completed':0})
    print('OUTPUT:',output,flush=True);rows=[];cv2.setNumThreads(1)
    try:
        for sample in samples:
            frames,identity=decode(sample,guard);n=len(frames);analysis=analyze(frames,policy);sid=sample['sample_id']
            original=f'previews/{sid}_original.mp4';encode(frames,output/original,sample['fps_rational'])
            p=analysis['proposed_contiguous_interval'];preview=None
            if p:
                preview=f'previews/{sid}_candidate.mp4'
                encode(frames[p['start_frame']:p['end_frame_exclusive']],output/preview,sample['fps_rational'])
            points={min(n-1,n*q//10) for q in range(10)}|{n-1}
            for c in analysis['cut_frame_candidates']:
                points.update(range(max(0,c-2),min(n,c+3)))
            if p:points.update((p['start_frame'],p['end_frame_exclusive']-1))
            contacts=frame_pages(frames,points,output/'contacts',sid,set(analysis['cut_frame_candidates']))
            time=f'timelines/{sid}.png';timeline(analysis,n,output/time)
            row={'sample_id':sid,'video_id':sample['video_id'],'source_group':sample['source_group'],
                 'original_split':'train','frame_count':n,'fps':float(Fraction(sample['fps_rational'])),
                 'source_identity':identity,'analysis':analysis,'original_preview':original,'proposal_preview':preview,
                 'display_previews_are_lossy_not_feature_inputs':True,'baseline_windows_unchanged':select_covering_windows(n),
                 'timeline':time,'contact_pages':['contacts/'+x for x in contacts]}
            rows.append(row);save(output/'run_state.json',{'status':'running','videos_completed':len(rows)})
            print(f'{len(rows)}/20 {sid} cuts={len(analysis["cut_frame_candidates"])} proposal={None if p is None else (p["start_frame"],p["end_frame_exclusive"])}',flush=True)
        stats=ledger.statistics();require(stats['train']['opens']==20 and stats['validation']['opens']==stats['test']['opens']==0,'Unexpected original access')
        require(all(sha(p)==h for p,h in protected.items()),'Protected baseline/source changed')
        report={'schema':'train20_contiguous_scene_proposal_diagnostic_v1','status':'completed','videos':rows,
                'policy':asdict(policy),'summary':{'videos':20,'single_interval_proposals':sum(bool(r['proposal_preview']) for r in rows),
                'candidate_fallback_videos':sum(not bool(r['proposal_preview']) for r in rows),'automatic_training_use_videos':0},
                'original_video_access':stats,'sequential_frames_decoded':sum(r['frame_count'] for r in rows),
                'protected_input_sha256':protected,'source_annotations_modified':False,'training_started':False,
                'feature_cache_modified':False,'original_videos_deleted':False,'classification_validation_test_used':False,
                'limitations':'Previously inspected20 purposeful train diagnostic videos; handcrafted thresholds not calibrated; camera/court segmentation is not shot-event localization; no release/result ground truth',
                'AI_review_status':'pending'}
        save(output/'report.json',report)
        save(output/'human_event_review_draft.json',{'origin':'unreviewed_not_ground_truth','videos':[
            {'sample_id':r['sample_id'],'video_id':r['video_id'],'release_frame':None,'ball_at_rim_frame':None,
             'result_visible_frame':None,'event_end_frame_exclusive':None,'proposal_preserves_event_and_result':None,
             'human_confirmed':False,'notes':''} for r in rows]})
        render(output,rows);save(output/'run_state.json',{'status':'completed','videos_completed':20})
        print({'output':str(output),'summary':report['summary']},flush=True);return output
    except BaseException as exc:
        save(output/'run_state.json',{'status':'failed','videos_completed':len(rows),'error':f'{type(exc).__name__}: {exc}'});raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit-dir',default=str(DEFAULT_AUDIT));p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    print(execute(a.audit_dir,a.dry_run))


if __name__=='__main__':main()
