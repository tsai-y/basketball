"""Record a direct whole-image human reply without editing immutable batch drafts.

Produces annotation progress/snapshots, never YOLO labels, training or split changes.
The caller must have an actual human reply to the currently displayed question.
"""
from copy import deepcopy
from collections import Counter
from pathlib import Path
import argparse
import json

from .probe import ARTIFACTS, read, require, save, sha, within
from development.yolo_audit_v1.target_label_pilot import CLASS_NAMES, validate_review


def accept_image(draft, review, ai, meta, question, answer, event):
    require(answer in ('可接受', '框可接受', '匡可接受', '整張可接受', '整張與角色可接受'),
            'Explicit whole-image acceptance required')
    iid = ai['image_id']
    require(question['image_id'] == iid and question['response'] is None
            and question['question_id'].startswith(iid+'_whole_image')
            and question['confirmation_scope'].startswith('Whole image'), 'Not a pending whole-image question')
    require(not ai.get('uncertain_regions') and ai.get('scene_hint','').startswith('court_play'),
            'Unresolved region or non-court scene')
    require(meta['image_id'] == iid and meta['image_sha256'] == event['image_sha256'], 'Wrong image identity')
    result=deepcopy(review)
    row=next(r for r in result['images'] if r['image_id']==iid)
    require(row['status']=='pending', 'Image already resolved; do not silently overwrite')
    require(row['image_sha256']==meta['image_sha256'], 'Review image SHA changed')
    annotations=[]
    for a in ai['annotations']:
        annotations.append({'id':a['id'],'class_id':a['class_id'],'xyxy':list(a['xyxy']),
            'source_proposal_id':None,'origin':'AI_box_explicitly_accepted_by_human',
            'human_confirmed':True})
    require(annotations,'Empty negative labels require a separate protocol')
    row.update(status='reviewed',scene='court_play',annotations=annotations,
        annotation_complete=True,proposal_states={p['id']:'rejected' for p in meta['proposals']},
        class_visibility={n:('present' if any(a['class_id']==i for a in annotations) else 'absent')
                          for i,n in enumerate(CLASS_NAMES)},
        notes='整張人工確認AI修正框與可見物件完整性；原弱模型提案由修正框取代。角色未判定，不匯出。')
    if 'ball_localization_status' in row:
        row['ball_localization_status']='confirmed_in_whole_image_review'
    counts=Counter(a['class_id'] for a in annotations)
    confirmation={**deepcopy(event),'image_id':iid,'question_id':question['question_id'],
        'question':question['question'],'user_reply':answer,'source':'direct_human_chat_reply',
        'whole_image_confirmed':True,'annotation_complete':True,
        'accepted_counts':{n:counts[i] for i,n in enumerate(CLASS_NAMES)},
        'class_visibility':deepcopy(row['class_visibility']),
        'confirmation_scope':'All shown boxes, no missing/extra visible objects, and per-class visibility as recorded; not optional holding_ball roles'}
    result.setdefault('human_full_image_confirmation_events',[]).append(confirmation)
    validate_review(draft,result)
    return result


def accept_image_excluding(draft, review, ai, meta, question, answer, event, rejected_ids):
    require(rejected_ids and len(set(rejected_ids))==len(rejected_ids),'Explicit unique rejected IDs required')
    named='、'.join(rejected_ids)
    require(answer in (named+'不是，其餘匡可接受',named+'不是，其餘框可接受',named+'不是，其餘可接受'),
            'Need explicit rejection plus acceptance of all remaining boxes')
    proposed={a['id']:a for a in ai['annotations']}
    require(all(i in proposed and proposed[i]['class_id']==0 for i in rejected_ids),
            'This pathway handles named non-player rejection only; ball corrections need location review')
    filtered=deepcopy(ai)
    filtered['annotations']=[a for a in filtered['annotations'] if a['id'] not in rejected_ids]
    result=accept_image(draft,review,filtered,meta,question,'可接受',event)
    confirmation=result['human_full_image_confirmation_events'][-1]
    confirmation.update(user_reply=answer,rejected_annotation_ids=list(rejected_ids),
        rejected_candidates=[deepcopy(proposed[i]) for i in rejected_ids],
        accepted_annotation_ids=[a['id'] for a in filtered['annotations']],
        snapshot_scope='Full shown AI draft, including named rejected candidates; export canonical accepted annotations only',
        confirmation_scope='Explicitly reject named non-player candidates; accept unchanged remaining boxes and whole-image completeness')
    row=next(r for r in result['images'] if r['image_id']==ai['image_id'])
    row['rejected_AI_candidate_ids']=list(rejected_ids)
    row['notes']='人工明確否決'+named+'，接受其餘原位置框與整張完整性；不匯出被否決提案，不猜其身份。'
    validate_review(draft,result)
    return result


def mark_region_uncertain(draft, review, question, answer, event):
    require('不確定' in answer, 'An explicit uncertain human reply is required')
    require(question['response'] is None and question['question_id'].endswith('_ball_visibility'),
            'Not a pending ball visibility question')
    result=deepcopy(review);iid=question['image_id']
    row=next(r for r in result['images'] if r['image_id']==iid)
    require(row['status']=='pending' and row['image_sha256']==event['image_sha256'],
            'Already completed or wrong image')
    row['class_visibility']['ball']='uncertain'
    row['annotation_complete']=False
    row['notes']=answer+'；僅確認疑問區域看不清，不推定整張無球，不猜遮擋球框。'
    result.setdefault('human_region_observation_events',[]).append({**deepcopy(event),
        'image_id':iid,'question_id':question['question_id'],'region_id':question['region_id'],
        'question':question['question'],'user_reply':answer,'source':'direct_human_chat_reply',
        'assessment':'uncertain','occlusion_cause_confirmed':False,'whole_image_confirmed':False,
        'confirmation_scope':question['confirmation_scope']})
    validate_review(draft,result)
    return result


def refresh_editor(batch, review):
    editor=batch/'editor.html';page=editor.read_text()
    start='<script id="pilot-data" type="application/json">';stop='</script>'
    require(page.count(start)==1,'Editor data block missing')
    left,rest=page.split(start,1);old,right=rest.split(stop,1)
    data=json.loads(old);data['draft']=review
    editor.write_text(left+start+json.dumps(data,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')+stop+right,encoding='utf-8')


def partition_readiness(status, split_plan):
    by_id={r['image_id']:r['detector_partition'] for r in split_plan['records']}
    ready=status['training_ready_ids']
    require(all(i in by_id for i in ready),'Ready image missing partition')
    return {'annotation_complete_images':len(ready),
        'detector_train_ready_ids':[i for i in ready if by_id[i]=='detector_train'],
        'detector_dev_validation_ready_ids':[i for i in ready if by_id[i]=='detector_dev_validation'],
        'legacy_training_ready_counter_means_annotation_complete_not_train_membership':True}


def question_overlay(batch, iid, question):
    require(question is not None and question['image_id']==iid,'Wrong pending image question')
    relative=question.get('overlay_path',f'ai/{iid}_overlay_v2.png')
    path=within(batch,relative)
    prefix=f'{iid}_overlay_v'
    require(path.parent==(Path(batch)/'ai').resolve() and path.suffix=='.png'
            and path.stem.startswith(prefix) and path.stem[len(prefix):].isdigit(),
            'Overlay must be this image and an explicit version')
    return path


def record_uncertainty(batch, iid, answer, timestamp):
    batch=Path(batch).resolve();require(batch.is_relative_to(ARTIFACTS.resolve()),'Not an expansion artifact')
    manifest=read(within(batch,'manifest.json'));require(manifest['schema']=='yolo_annotation_expansion_v1','Wrong batch schema')
    draft_path=within(batch,'draft_review.json');draft=read(draft_path)
    canonical=within(batch,'guided/guided_review.json');review=read(canonical)
    triage_path=within(batch,'ai/triage.json');triage=read(triage_path);q=triage['next_human_question']
    require(q['image_id']==iid,'Wrong current question image')
    meta=next(r for r in manifest['images'] if r['image_id']==iid)
    require(sha(within(batch,meta['image_path']))==meta['image_sha256'],'Image changed')
    ai_path=within(batch,f'ai/{iid}_draft.json');ai=read(ai_path)
    snapshot=batch/'guided/observations'/f'{q["question_id"]}.json'
    require(not snapshot.exists(),'Observation already recorded')
    event={'recorded_at_utc':timestamp,'image_sha256':meta['image_sha256'],
           'question_draft_path':str(snapshot.relative_to(batch/'guided')),'question_draft_sha256':sha(ai_path)}
    updated=mark_region_uncertain(draft,review,q,answer,event)
    snapshot.parent.mkdir(exist_ok=True);snapshot.write_bytes(ai_path.read_bytes())
    save(canonical,updated)
    triage.setdefault('answered_questions',[]).append({**q,'response':answer,'recorded_at_utc':timestamp,
        'assessment':'uncertain','whole_image_confirmed':False,'occlusion_cause_confirmed':False})
    triage['next_human_question']=None
    triage.setdefault('deferred',[]).append({'image_id':iid,'reason':'human_uncertain_ball_visibility',
        'finding':answer,'automatic_empty_labels_allowed':False,'guess_occluded_box_allowed':False})
    status=validate_review(draft,updated)
    status.update(partition_readiness(status,read(within(batch,'split_plan.json'))))
    triage.update(new_human_completed_images=status['training_ready_images'],
                  new_training_ready_images=len(status['detector_train_ready_ids']),
                  new_development_validation_ready_images=len(status['detector_dev_validation_ready_ids']))
    save(triage_path,triage)
    ai['human_region_assessment']={'region_id':q['region_id'],'answer':answer,'assessment':'uncertain',
        'recorded_at_utc':timestamp,'occlusion_cause_confirmed':False}
    ai['status']='deferred_uncertain_ball';ai['uncertain_regions'][0]['human_response']=answer
    save(ai_path,ai)
    save(batch/'guided/status.json',{**status,'original_draft_sha256':sha(draft_path),
        'review_sha256':sha(canonical),'source_split_unchanged':True,'training_started':False})
    refresh_editor(batch,updated)
    return {'image_id':iid,'assessment':'uncertain','box_guessed':False,**status}


def record(batch, iid, answer, timestamp, rejected_ids=()):
    batch=Path(batch).resolve()
    require(batch.is_relative_to(ARTIFACTS.resolve()),'Not an expansion artifact')
    manifest=read(within(batch,'manifest.json'));require(manifest['schema']=='yolo_annotation_expansion_v1','Wrong batch schema')
    draft_path=within(batch,'draft_review.json');draft=read(draft_path)
    triage_path=within(batch,'ai/triage.json');triage=read(triage_path)
    ai_path=within(batch,f'ai/{iid}_draft.json');ai=read(ai_path)
    overlay=question_overlay(batch,iid,triage['next_human_question'])
    meta=next(r for r in manifest['images'] if r['image_id']==iid)
    require(sha(within(batch,meta['image_path']))==meta['image_sha256'],'Original PNG changed')
    partition=next(r for r in read(within(batch,'split_plan.json'))['records'] if r['image_id']==iid)
    guided=batch/'guided';canonical=guided/'guided_review.json'
    review=read(canonical) if canonical.exists() else deepcopy(draft)
    snapshot_rel=f'confirmed/{iid}_accepted_AI_boxes.json'
    event={'image_sha256':meta['image_sha256'],'recorded_at_utc':timestamp,
        'draft_path':snapshot_rel,'draft_sha256':sha(ai_path),'overlay_path':str(overlay.relative_to(batch)),
        'overlay_sha256':sha(overlay),'detector_partition':partition['detector_partition']}
    if rejected_ids:
        updated=accept_image_excluding(draft,review,ai,meta,triage['next_human_question'],answer,event,rejected_ids)
    else:
        updated=accept_image(draft,review,ai,meta,triage['next_human_question'],answer,event)
    snapshot=guided/snapshot_rel;require(not snapshot.exists(),'Confirmation snapshot already exists')
    guided.mkdir(exist_ok=True);snapshot.parent.mkdir(exist_ok=True)
    snapshot.write_bytes(ai_path.read_bytes())
    save(canonical,updated)
    answered={**triage['next_human_question'],'response':answer,'recorded_at_utc':timestamp,
              'whole_image_confirmed':True,'accepted_counts':updated['human_full_image_confirmation_events'][-1]['accepted_counts']}
    if rejected_ids:answered['rejected_annotation_ids']=list(rejected_ids)
    triage.setdefault('answered_questions',[]).append(answered)
    triage['next_human_question']=None
    status=validate_review(draft,updated)
    status.update(partition_readiness(status,read(within(batch,'split_plan.json'))))
    triage.update(new_human_completed_images=status['training_ready_images'],
        new_training_ready_images=len(status['detector_train_ready_ids']),
        new_development_validation_ready_images=len(status['detector_dev_validation_ready_ids']),human_review_modified=True)
    save(triage_path,triage)
    save(guided/'status.json',{**status,'original_draft_sha256':sha(draft_path),
        'review_sha256':sha(canonical),'source_split_unchanged':True,'training_started':False})
    # Refresh only the generated editor's embedded initial progress. Preserve browser
    # localStorage without resetting it; JSON import remains the explicit recovery path.
    refresh_editor(batch,updated)
    return {'image_id':iid,'accepted_counts':answered['accepted_counts'],**status,'canonical_review':str(canonical)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch',required=True);p.add_argument('--image-id',required=True)
    p.add_argument('--reply',required=True);p.add_argument('--recorded-at-utc',required=True)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--uncertain',action='store_true')
    mode.add_argument('--reject',action='append',default=[])
    args=p.parse_args()
    if args.uncertain:
        result=record_uncertainty(args.batch,args.image_id,args.reply,args.recorded_at_utc)
    else:
        result=record(args.batch,args.image_id,args.reply,args.recorded_at_utc,args.reject)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
