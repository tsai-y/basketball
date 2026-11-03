"""Small ball-only review from source-reserved native cached PNGs; never train."""
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import argparse
import html
import math
import shutil
import tempfile

import cv2
import numpy as np

from development.data_pipeline.pipeline import AccessLedger,SplitGuard
from .probe import ARTIFACTS,read,save,sha,require,within
from .ball_only_probe_v1 import check as check_ball
from .prepare_expansion_v1 import validate_native_pair

SCHEMA='source_diverse_ball_only_review_v2'
PROTOCOL='visible_physical_ball_only_complete_image_v2'


def select_sources(samples,assignments,used):
    require(samples and all(s.get('split')=='train' and str(s.get('excluded'))=='0' for s in samples),'Original train sources only')
    require(len({s['video_id'] for s in samples})==len(samples),'Duplicate source video')
    require(len({s['source_group'] for s in samples})==len(samples),'One cached video per group required')
    available=[s for s in samples if s['source_group'] not in used]
    selected=[]
    for original_class in ('2p0','2p1','3p0','3p1'):
        pool=[s for s in available if s['original_class']==original_class and assignments.get(s['source_group'])=='detector_train']
        require(pool,'No unused training source for '+original_class)
        selected.append(min(pool,key=lambda s:s['source_group']))
    held=sorted([s for s in available if assignments.get(s['source_group'])=='detector_dev_validation'],key=lambda s:s['source_group'])
    require(len(held)>=2,'Need two unused pre-reserved development sources')
    return selected+held[:2]


def check_proposal(boxes):
    require(boxes and isinstance(boxes,list),'Visible positive ball boxes required; uncertain is not empty negative')
    for b in boxes:
        require(len(b)==4 and all(type(x) is int for x in b),'Integer native box coordinates required')
        require(0<=b[0]<b[2]<=320 and 0<=b[1]<b[3]<=240,'Invalid ball box')
    require(len({tuple(b) for b in boxes})==len(boxes),'Duplicate ball boxes')


def protect_check(manifest):
    for path,h in manifest['protected_input_sha256'].items():require(sha(path)==h,'Protected source changed: '+path)


def validate_pending(row):
    check_proposal(row['ball_boxes_proposed'])
    require(row['status']=='pending_human_ball_only_full_image_review' and row['human_confirmed'] is False
            and row['visible_ball_completeness_confirmed'] is False and row['training_ready'] is False,
            'AI-only draft cannot be marked human-confirmed/complete/ready')


def check_pending(review_dir):
    review_dir=Path(review_dir).resolve();require(review_dir.is_relative_to(ARTIFACTS.resolve()),'Foreign review')
    d=read(review_dir/'review_draft.json');require(d['schema']==SCHEMA and d['protocol']==PROTOCOL,'Wrong draft protocol')
    pool=Path(d['pool']);m=read(pool/'manifest.json');protect_check(m)
    require(sha(pool/'manifest.json')==d['pool_manifest_sha256'],'Pool manifest changed')
    require(d['human_reply'] is None and d['human_confirmation_events']==[] and d['training_ready_images']==0,
            'Not an unconfirmed AI draft')
    metas={r['image_id']:r for r in m['records']}
    for row in d['images']:
        validate_pending(row);meta=metas[row['image_id']]
        require(all(row[k]==meta[k] for k in meta),'Source metadata differs')
        require(sha(within(pool,row['image']))==row['image_sha256'],'Native image changed')
        require(sha(within(review_dir,row['shown_panel']))==row['shown_panel_sha256'],'Shown proposal changed')
    require(not d['training_started'] and not d['labels_exported'] and not d['new_human_labels_created'],'Wrong phase claim')
    return {'images':len(d['images']),'training_ready_images':0,'labels_exported':False}


def corrected_location_proposal(ai,image_id,boxes,reply):
    require(ai['origin']=='AI_proposals_not_ground_truth' and isinstance(reply,str) and reply.strip(),
            'AI draft and direct human location reply required')
    check_proposal(boxes)
    result=deepcopy(ai);rows=[r for r in result['selected'] if r['image_id']==image_id]
    require(len(rows)==1,'Correction must identify exactly one shown image')
    old=deepcopy(rows[0]['ball_boxes']);rows[0]['ball_boxes']=deepcopy(boxes)
    rows[0]['reason']='依人工提供局部圖更正球的位置；新框邊界與全圖可見球完整性仍待確認。'
    event={'image_id':image_id,'user_reply':reply,'source':'direct_human_chat_reply_with_image',
           'scope':'Ball location correction only; new AI box extent, other images and full-image completeness not confirmed',
           'withdrawn_ball_boxes':old,'new_AI_ball_boxes':deepcopy(boxes),
           'ball_box_boundary_confirmed':False,'whole_image_confirmed':False,'batch_confirmed':False,
           'training_ready_images':0}
    result['human_location_correction']=deepcopy(event)
    return result,event


def stage_location_correction(review_dir,image_id,boxes,reply,evidence):
    review_dir=Path(review_dir).resolve();check_pending(review_dir)
    old=read(review_dir/'review_draft.json');pool=Path(old['pool'])
    evidence=Path(evidence).resolve();require(evidence.is_file(),'User evidence missing')
    ai,event=corrected_location_proposal(read(review_dir/'AI_proposals_snapshot.json'),image_id,boxes,reply)
    inputs=Path(tempfile.mkdtemp(prefix='location_correction_',dir=pool))
    copied=inputs/('user_evidence'+evidence.suffix);shutil.copy2(evidence,copied)
    require(sha(copied)==sha(evidence),'Evidence pixels changed')
    event.update(evidence_path=str(copied),evidence_sha256=sha(copied),
                 parent_review_dir=str(review_dir),parent_review_draft_sha256=sha(review_dir/'review_draft.json'))
    ai['human_location_correction']=deepcopy(event)
    save(inputs/'AI_proposals.json',ai);output=stage(pool,inputs/'AI_proposals.json')
    updated=read(output/'review_draft.json');by_id={r['image_id']:r for r in updated['images']}
    for row in old['images']:
        current=by_id[row['image_id']]
        if row['image_id']!=image_id:
            require(current['ball_boxes_proposed']==row['ball_boxes_proposed'] and
                    current['shown_panel_sha256']==row['shown_panel_sha256'],'Unrelated proposal changed')
        else:
            event.update(previous_shown_panel_sha256=row['shown_panel_sha256'],
                         revised_shown_panel_sha256=current['shown_panel_sha256'],
                         revised_review_dir=str(output))
    save(output/'human_location_correction.json',event);check_pending(output);return output


def accept_other_ball_images(draft,corrected_image_id,reply):
    require(reply=='其餘都正常','Explicit other-images reply required')
    require(draft['protocol']==PROTOCOL and corrected_image_id in {r['image_id'] for r in draft['images']},
            'Known corrected image and ball-only protocol required')
    result=deepcopy(draft);accepted=[]
    for row in result['images']:
        validate_pending(row)
        if row['image_id']==corrected_image_id:continue
        row.update(status='human_confirmed_ball_only_full_image',human_confirmed=True,
                   visible_ball_completeness_confirmed=True,annotation_complete=True,
                   training_ready=row['detector_partition']=='train',
                   annotations=[{'id':f'ball{i}','class_id':0,'xyxy':deepcopy(box),
                                 'origin':'AI_box_accepted_in_direct_human_ball_only_review'}
                                for i,box in enumerate(row['ball_boxes_proposed'],1)])
        accepted.append(row['image_id'])
    require(accepted,'No other images to confirm')
    event={'source':'direct_human_chat_reply','user_reply':reply,'accepted_image_ids':accepted,
           'explicitly_not_accepted_image_ids':[corrected_image_id],
           'confirmation_scope':'Other displayed images: visible physical ball boxes and full-image ball completeness only; corrected image excluded; no player/rim/role approval',
           'protocol':PROTOCOL,'batch_fully_confirmed':False}
    result['human_confirmation_events'].append(event);result['human_reply']=reply
    result['annotation_complete_images']=len(accepted)
    result['detector_train_confirmed_ids']=[r['image_id'] for r in result['images'] if r['human_confirmed'] and r['detector_partition']=='train']
    result['detector_development_confirmed_ids']=[r['image_id'] for r in result['images'] if r['human_confirmed'] and r['detector_partition']=='development']
    result['training_ready_images']=len(result['detector_train_confirmed_ids'])
    result['pending_ids']=[corrected_image_id]
    result['new_human_labels_created']=True
    return result


def record_others_confirmation(review_dir,reply):
    review_dir=Path(review_dir).resolve();check_pending(review_dir)
    progress_path=review_dir/'human_review_progress.json'
    require(not progress_path.exists(),'Human progress exists; do not overwrite prior replies')
    draft=read(review_dir/'review_draft.json');correction=read(review_dir/'human_location_correction.json')
    corrected=correction['image_id']
    require(correction['ball_box_boundary_confirmed'] is False and correction['whole_image_confirmed'] is False,
            'Wrong pending correction context')
    result=accept_other_ball_images(draft,corrected,reply)
    event=result['human_confirmation_events'][-1]
    event.update(recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                 source_review_draft_sha256=sha(review_dir/'review_draft.json'),
                 source_location_correction_sha256=sha(review_dir/'human_location_correction.json'),
                 accepted_boxes_and_display_evidence=[{'image_id':r['image_id'],'ball_boxes':deepcopy(r['ball_boxes_proposed']),
                      'image_sha256':r['image_sha256'],'shown_panel':r['shown_panel'],'shown_panel_sha256':r['shown_panel_sha256']}
                      for r in result['images'] if r['human_confirmed']])
    row=next(r for r in result['images'] if r['image_id']==corrected)
    result['next_human_question']={'image_id':corrected,'question_id':corrected+'_corrected_ball_and_completeness_v2',
        'question':'B006修正版球框與整張沒有其他漏球，是否可接受？',
        'shown_panel':row['shown_panel'],'shown_panel_sha256':row['shown_panel_sha256'],
        'scope':'B006 corrected box boundaries and full-image visible-ball completeness only','response':None}
    save(review_dir/'human_confirmation_others_v1.json',event);save(progress_path,result)
    confirmed='、'.join(event['accepted_image_ids'])
    page=f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>球框人工進度：只剩{corrected}</title><style>body{{font:18px system-ui;line-height:1.8;max-width:1100px;margin:30px auto;padding:0 20px}}img{{max-width:100%}}</style><h1>已確認：{confirmed}</h1><p>人工原文：{html.escape(reply)}。僅記錄其他三張球框與全圖球完整性；不是player/rim/角色確認，也不是整批通過。2張train、1張development分開保留。</p><h2>{corrected}修正版仍待確認</h2><p>{result['next_human_question']['question']} 可回覆「可接受」，或指出需要修正處。其餘三張不必重複確認。</p><img src="{row['shown_panel']}"><p>目前未匯出labels、未訓練。<a href="human_review_progress.json">人工進度與精確範圍</a> · <a href="human_confirmation_others_v1.json">原文／框／顯示圖稽核</a> · <a href="index.html">原4張顯示版本（AI草稿原檔保留）</a></p></html>'''
    (review_dir/'human_review_status.html').write_text(page,encoding='utf-8')
    protect_check(read(Path(draft['pool'])/'manifest.json'));return progress_path


def accept_current_ball_image(progress,reply):
    require(reply in ('可接受','球框可接受','框可接受'),'Explicit current-image acceptance required')
    require(progress['protocol']==PROTOCOL,'Wrong ball-only protocol')
    question=progress.get('next_human_question')
    require(question and question['response'] is None,'No unanswered current image question')
    iid=question['image_id'];rows=[r for r in progress['images'] if r['image_id']==iid]
    require(len(rows)==1 and progress['pending_ids']==[iid],'Wrong pending image scope')
    require(question['scope']==iid+' corrected box boundaries and full-image visible-ball completeness only',
            'Current question does not confirm box plus completeness')
    validate_pending(rows[0]);result=deepcopy(progress)
    row=next(r for r in result['images'] if r['image_id']==iid)
    require(question['shown_panel']==row['shown_panel'] and question['shown_panel_sha256']==row['shown_panel_sha256'],
            'Question display version differs from proposed box')
    row.update(status='human_confirmed_ball_only_full_image',human_confirmed=True,
               visible_ball_completeness_confirmed=True,annotation_complete=True,
               training_ready=row['detector_partition']=='train',
               annotations=[{'id':f'ball{i}','class_id':0,'xyxy':deepcopy(box),
                             'origin':'AI_box_accepted_in_direct_human_ball_only_review'}
                            for i,box in enumerate(row['ball_boxes_proposed'],1)])
    event={'source':'direct_human_chat_reply','user_reply':reply,'accepted_image_ids':[iid],
           'question':deepcopy(question),'protocol':PROTOCOL,
           'confirmation_scope':question['scope'],'other_images_reconfirmed_by_this_reply':False,
           'accepted_boxes_and_display_evidence':[{'image_id':iid,'ball_boxes':deepcopy(row['ball_boxes_proposed']),
               'image_sha256':row['image_sha256'],'shown_panel':row['shown_panel'],'shown_panel_sha256':row['shown_panel_sha256']} ]}
    result['human_confirmation_events'].append(event)
    result.setdefault('answered_questions',[]).append({**deepcopy(question),'response':reply})
    result['next_human_question']=None;result['pending_ids']=[]
    result['annotation_complete_images']=sum(r['human_confirmed'] and r['visible_ball_completeness_confirmed'] for r in result['images'])
    result['detector_train_confirmed_ids']=[r['image_id'] for r in result['images'] if r['human_confirmed'] and r['detector_partition']=='train']
    result['detector_development_confirmed_ids']=[r['image_id'] for r in result['images'] if r['human_confirmed'] and r['detector_partition']=='development']
    result['training_ready_images']=len(result['detector_train_confirmed_ids'])
    result['batch_annotation_complete']=result['annotation_complete_images']==len(result['images'])
    result['latest_human_reply']=reply
    return result


def record_current_confirmation(review_dir,reply):
    review_dir=Path(review_dir).resolve();check_pending(review_dir)
    progress_path=within(review_dir,'human_review_progress.json');progress=read(progress_path)
    draft=read(review_dir/'review_draft.json');by_id={r['image_id']:r for r in draft['images']}
    for r in progress['images']:
        original=by_id[r['image_id']]
        require(all(r[k]==original[k] for k in ('ball_boxes_proposed','image_sha256','shown_panel','shown_panel_sha256')),
                'Human progress geometry/display changed')
        require(sha(within(review_dir,r['shown_panel']))==r['shown_panel_sha256'],'Shown panel changed')
    result=accept_current_ball_image(progress,reply)
    history=Path(tempfile.mkdtemp(prefix='human_confirmation_',dir=review_dir))
    shutil.copy2(progress_path,history/'progress_before.json')
    if (review_dir/'human_review_status.html').exists():shutil.copy2(review_dir/'human_review_status.html',history/'status_before.html')
    event=result['human_confirmation_events'][-1]
    event.update(recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                 source_progress_before_sha256=sha(history/'progress_before.json'),
                 source_review_draft_sha256=sha(review_dir/'review_draft.json'),
                 history_dir=str(history))
    save(history/'event.json',event);save(history/'progress_after.json',result);save(progress_path,result)
    confirmed='、'.join(r['image_id'] for r in result['images'] if r['human_confirmed'])
    page=f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>球框人工確認4/4完成</title><style>body{{font:18px system-ui;line-height:1.8;max-width:1100px;margin:30px auto;padding:0 20px}}</style><h1>這批球框與全圖球完整性：4/4已確認</h1><p>{confirmed}。</p><p>B006位置先依上傳圖更正，再以本次「{html.escape(reply)}」確認修正版球框與全圖無漏球。先前「其餘都正常」只確認另外3張；兩次範圍分開保存，沒有推定你一次同意整批。</p><p>3張train：B001、B006、B012；1張development：B029。development不移入train。只確認可見實體球，不包含player/rim/holding_ball角色。</p><p>尚未匯出YOLO標籤、尚未開始新訓練，原模型與MixNet特徵未改。下一步可另建來源分離的新資料版本，再安排訓練。</p><p><a href="human_review_progress.json">完整人工進度與原文</a> · <a href="{history.name}/event.json">B006本次確認稽核</a> · <a href="index.html">原4張草稿顯示版本（保留）</a></p></html>'''
    (review_dir/'human_review_status.html').write_text(page,encoding='utf-8')
    protect_check(read(Path(draft['pool'])/'manifest.json'));return progress_path


def prepare(batch,source_ball_dataset):
    batch,source_ball_dataset=Path(batch).resolve(),Path(source_ball_dataset).resolve()
    require(batch.is_relative_to(ARTIFACTS.resolve()),'Foreign annotation batch')
    old=check_ball(source_ball_dataset)
    manifest=read(batch/'manifest.json');plan=read(batch/'split_plan.json');review=read(batch/'guided/guided_review.json')
    audit=Path(manifest['source_audit']);require(audit.is_relative_to(ARTIFACTS.resolve()),'Foreign native audit')
    audit_manifest=read(audit/'manifest.json');state=read(audit/'run_state.json')
    require(state['status']=='completed' and audit_manifest['split']=='train','Unsafe/incomplete cache')
    require(plan['source_assignments']==old['source_assignments'],'Source reservations changed')
    used={r['source_group'] for r in old['records']}
    selected=select_sources(audit_manifest['samples'],plan['source_assignments'],used)
    guard=SplitGuard('train',AccessLedger())
    for s in selected:
        require(s['video_id'] in guard.allowed,'Non-canonical original train source')
        official=guard.allowed[s['video_id']]
        require(all(s[k]==official[k] for k in ('source_group','original_class','split')),'Canonical source mismatch')
    uncertain_ids={r['image_id'] for r in review['images'] if r.get('class_visibility',{}).get('ball')=='uncertain'}
    uncertain={(r['audit_sample_id'],r['frame_index']) for r in plan['records'] if r['image_id'] in uncertain_ids}
    protected={str(p):sha(p) for p in (batch/'manifest.json',batch/'split_plan.json',batch/'guided/guided_review.json',
                source_ball_dataset/'dataset_manifest.json',audit/'manifest.json',audit/'run_state.json')}
    output=Path(tempfile.mkdtemp(prefix='ball_source_review_v2_',dir=ARTIFACTS))
    for sub in ('images','large','contacts','inputs'):(output/sub).mkdir()
    shutil.copy2(__file__,output/'source_snapshot.py')
    save(output/'inputs/source_split_plan.json',plan)
    records=[];contact_names=[];skipped=[]
    for s in selected:
        tiles=[]
        for q in (1,3,5,7,9):
            frame=s['frame_count']*q//10
            if (s['sample_id'],frame) in uncertain:
                skipped.append({'sample_id':s['sample_id'],'frame_index':frame,'reason':'existing_human_ball_uncertain_do_not_reask'});continue
            stem=f'{s["sample_id"]}_f{frame:06d}';native=within(audit,'original_frames/'+stem+'.png');pair=within(audit,'frames/'+stem+'.png')
            raw=cv2.imread(str(native));validate_native_pair(raw,cv2.imread(str(pair)))
            protected.update({str(native):sha(native),str(pair):sha(pair)})
            iid=f'B{len(records)+1:03d}';dest=output/'images'/f'{iid}.png';shutil.copy2(native,dest)
            large=f'large/{iid}.png';require(cv2.imwrite(str(output/large),cv2.resize(raw,(960,720),interpolation=cv2.INTER_NEAREST)),'Large preview failed')
            part='train' if plan['source_assignments'][s['source_group']]=='detector_train' else 'development'
            records.append({'image_id':iid,'image':str(dest.relative_to(output)),'large_preview':large,'image_sha256':sha(dest),
                'source_group':s['source_group'],'video_id':s['video_id'],'source_split':'train','detector_partition':part,
                'original_class':s['original_class'],'audit_sample_id':s['sample_id'],'frame_index':frame,
                'fixed_position_tenths':q,'source_native_png':str(native)})
            tile=np.zeros((504,640,3),np.uint8);tile[24:]=cv2.resize(raw,(640,480),interpolation=cv2.INTER_NEAREST)
            cv2.putText(tile,f'{iid} / {part} / frame {frame}',(5,17),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);tiles.append(tile)
        while len(tiles)<6:tiles.append(np.zeros((504,640,3),np.uint8))
        page=f'contacts/{s["sample_id"]}.png';require(cv2.imwrite(str(output/page),np.vstack([np.hstack(tiles[i:i+2]) for i in (0,2,4)])),'Contact failed');contact_names.append(page)
    result={'schema':SCHEMA,'protocol':PROTOCOL,'source_annotation_batch':str(batch),'source_ball_dataset':str(source_ball_dataset),
        'records':records,'contacts':contact_names,'used_sources_excluded':sorted(used),
        'selected_source_groups':[s['source_group'] for s in selected],'human_uncertain_frames_skipped':skipped,
        'source_assignments':plan['source_assignments'],'protected_input_sha256':protected,
        'source_sampling':'4 new training sources, one per2p0/2p1/3p0/3p1, lexical source choice;2 unused pre-reserved development sources; five fixed positions before new ball inspection',
        'AI_visual_selection_will_bias_final_labeled_subset':True,'not_blind_or_formal_test':True,
        'training_started':False,'yolo_labels_exported':False,'human_labels_created':False,'training_ready_images':0,
        'original_video_opens':0,'classification_validation_test_used':False,'source_annotations_modified':False,'feature_cache_modified':False}
    save(output/'manifest.json',result);protect_check(result);return output


def stage(pool,proposals):
    pool=Path(pool).resolve();require(pool.is_relative_to(ARTIFACTS.resolve()),'Foreign review pool')
    m=read(pool/'manifest.json');require(m['schema']==SCHEMA,'Wrong review protocol');protect_check(m)
    ai=read(proposals);require(ai['origin']=='AI_proposals_not_ground_truth','Wrong proposal origin')
    selected=ai['selected'];require(1<=len(selected)<=6 and len({r['image_id'] for r in selected})==len(selected),'Small unique review batch required')
    by_id={r['image_id']:r for r in m['records']};groups=set();draft=[]
    for item in selected:
        require(item['image_id'] in by_id,'Foreign selected image');r=by_id[item['image_id']]
        require(r['source_group'] not in groups,'One proposed frame per source only');groups.add(r['source_group'])
        require(r['source_group'] not in m['used_sources_excluded'],'Already-used source')
        check_proposal(item['ball_boxes']);p=within(pool,r['image']);require(sha(p)==r['image_sha256'],'PNG changed')
        draft.append({**deepcopy(r),'ball_boxes_proposed':deepcopy(item['ball_boxes']),'AI_reason':item['reason'],
                      'status':'pending_human_ball_only_full_image_review','human_confirmed':False,
                      'visible_ball_completeness_confirmed':False,'training_ready':False})
    require({r['detector_partition'] for r in draft}=={'train','development'},'Include train and development, do not silently abandon held sources')
    output=Path(tempfile.mkdtemp(prefix='review_',dir=pool));(output/'panels').mkdir()
    save(output/'AI_proposals_snapshot.json',ai)
    record={'schema':SCHEMA,'protocol':PROTOCOL,'pool':str(pool),'pool_manifest_sha256':sha(pool/'manifest.json'),
            'images':draft,'human_reply':None,'human_confirmation_events':[],'training_ready_images':0,
            'confirmation_scope':'All proposed ball boxes and completeness of visible physical balls in every selected full image only; not player/rim/roles, training permission or unseen frames',
            'training_started':False,'labels_exported':False,'new_human_labels_created':False}
    cards=[];panels=[]
    for r in draft:
        raw=cv2.imread(str(pool/r['image']));overlay=raw.copy()
        for j,b in enumerate(r['ball_boxes_proposed'],1):
            x1,y1,x2,y2=b;cv2.rectangle(overlay,(x1,y1),(x2-1,y2-1),(0,255,0),1)
            cv2.putText(overlay,f'ball{j}',(x1,max(9,y1-3)),cv2.FONT_HERSHEY_SIMPLEX,.3,(0,255,0),1)
        b=r['ball_boxes_proposed'][0];cx=(b[0]+b[2])//2;cy=(b[1]+b[3])//2
        left,right=max(0,cx-24),min(320,cx+25);top,bottom=max(0,cy-24),min(240,cy+25)
        panel=np.zeros((534,1000,3),np.uint8);panel[34:514,:640]=cv2.resize(overlay,(640,480),interpolation=cv2.INTER_NEAREST)
        panel[94:414,660:980]=cv2.resize(raw[top:bottom,left:right],(320,320),interpolation=cv2.INTER_NEAREST)
        cv2.putText(panel,f'{r["image_id"]} / {r["detector_partition"]} / AI DRAFT ONLY',(5,23),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
        cv2.putText(panel,'RAW CONTEXT / NO ENHANCEMENT',(655,65),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1)
        cv2.putText(panel,'box '+str(b),(655,450),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1)
        filename=f'panels/{r["image_id"]}.png';require(cv2.imwrite(str(output/filename),panel),'Panel failed')
        r['shown_panel']=filename;r['shown_panel_sha256']=sha(output/filename)
        r['raw_context_crop_xyxy']=[left,top,right,bottom]
        panels.append(panel);cards.append(f'<section id="{r["image_id"]}"><h2>{r["image_id"]}</h2><p>{html.escape(r["AI_reason"])}</p><img src="{filename}"></section>')
    save(output/'review_draft.json',record)
    pages=[]
    for start in range(0,len(panels),3):
        name=f'batch_{start//3+1}.png';require(cv2.imwrite(str(output/name),np.vstack(panels[start:start+3])),'Batch page failed');pages.append(name)
    save(output/'review_status.json',{'selected_images':len(draft),'train_images':sum(r['detector_partition']=='train' for r in draft),
            'development_images':sum(r['detector_partition']=='development' for r in draft),'review_pages':pages,
            'training_ready_images':0,'training_started':False,'requires_direct_human_reply':True})
    page='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>只確認球：少量新來源</title><style>body{font:18px system-ui;line-height:1.8;max-width:1100px;margin:30px auto;padding:0 20px}img{max-width:100%}section{padding:15px;margin:20px 0;background:#f4f6f9}</style><h1>只確認球，不重做球員與籃圈</h1><p>每張左側是完整原生圖的2倍顯示，綠框為AI提出的球；右側為原始像素最近鄰放大的局部，沒有銳化或生成。請確認球框位置／範圍可接受，且全圖沒有其他漏掉的可見實體球。看不清就指出編號，不能把不確定改成無球。</p><p>這些來源尚未用於現有模型的train或development，但曾參與診斷，不是正式盲測。沿用原來源預留，development不能移入train。只有球，類別0；本批確認不等於三類物件／角色確認，也不會自動修改原標註。</p><p>若這頁列出的所有圖都可接受，可在對話回覆「整批球框與完整性可接受」；否則列出需要修正／看不清的B編號，其餘也可明確接受。收到你的明確回覆前全數pending，尚未匯出訓練標籤、未訓練。</p><p><a href="review_draft.json">不可變AI提案與待人工確認狀態</a> · <a href="AI_proposals_snapshot.json">AI篩選紀錄</a></p>'''+''.join(cards)+'</html>'
    (output/'index.html').write_text(page,encoding='utf-8');protect_check(m);check_pending(output);return output


def main():
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--stage',action='store_true')
    p.add_argument('--batch');p.add_argument('--source-ball-dataset');p.add_argument('--pool');p.add_argument('--proposals');a=p.parse_args()
    print(prepare(a.batch,a.source_ball_dataset) if a.prepare else stage(a.pool,a.proposals))


if __name__=='__main__':main()
