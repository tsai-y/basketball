"""Prepare annotation-derived negative crop candidates, never train or export labels.

Only complete human-reviewed detector-training images are eligible. A crop must
avoid every accepted ball box plus a fixed native-pixel margin. This is an
augmentation proposal, not new independent data or a new human annotation.
"""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import argparse
import html
import math
import shutil
import tempfile

import cv2
import numpy as np

from .probe import ARTIFACTS, read, save, sha, require, within
from .ball_only_probe_v1 import check as check_ball
from .train_expansion_v1 import check_dataset as check_human

SCHEMA = 'annotation_derived_ball_negative_crop_candidates_v1'
MARGIN = 12


def intersection_area(a, b):
    return max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))


def source_balls(record):
    require(record.get('source_split') == 'train' and record.get('partition') == 'train',
            'Only original-train / detector-train sources may yield negative crops')
    event = record.get('human_full_image_confirmation', {})
    legacy = event.get('source') == 'human_chat_reply'
    require(event.get('source') in ('human_chat_reply','direct_human_chat_reply'), 'Direct human event required')
    reply = event.get('user_quote') if legacy else event.get('user_reply')
    complete = event.get('annotation_complete_confirmed') if legacy else event.get('annotation_complete')
    require(event.get('whole_image_confirmed') is True and complete is True and bool(reply),
            'Complete human source annotation required')
    annotations = record['original_human_accepted_annotations']
    balls = [a for a in annotations if a['class_id'] == 1]
    require(balls and all((a.get('origin') == 'AI_proposed_box_explicitly_confirmed_by_human'
                          and event.get('confirmation_scope') == 'full_image_three_classes_boxes_and_completeness')
                         if legacy else a.get('human_confirmed') is True for a in balls),
            'Accepted human ball geometry required')
    require(event.get('accepted_counts', {}).get('ball') == len(balls), 'Human ball completeness/count mismatch')
    if not legacy:
        require(event.get('class_visibility', {}).get('ball') == 'present', 'Uncertain/absent source ball forbidden')
    for a in balls:
        b = a['xyxy']
        require(len(b) == 4 and all(type(v) in (int, float) and math.isfinite(v) for v in b)
                and 0 <= b[0] < b[2] <= 320 and 0 <= b[1] < b[3] <= 240, 'Invalid native ball geometry')
    return deepcopy(balls)


def choose_crop(record):
    """Fixed corner candidates, top-first; no predictions, class outcome or tuning."""
    balls = source_balls(record)
    expanded = [[max(0,a['xyxy'][0]-MARGIN), max(0,a['xyxy'][1]-MARGIN),
                 min(320,a['xyxy'][2]+MARGIN), min(240,a['xyxy'][3]+MARGIN)] for a in balls]
    for width, height in ((160,120), (128,96)):
        for y in (0,240-height):
            candidates = [[x,y,x+width,y+height] for x in (0,320-width)]
            safe = [b for b in candidates if all(intersection_area(b,g) == 0 for g in expanded)]
            if safe:
                # Farthest from all balls, then fixed left-to-right tie-break.
                def distance(b):
                    cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2
                    return min((cx-(a['xyxy'][0]+a['xyxy'][2])/2)**2 +
                               (cy-(a['xyxy'][1]+a['xyxy'][3])/2)**2 for a in balls)
                selected = max(safe,key=distance)
                return {'crop_xyxy':selected,'native_margin_px':MARGIN,'source_ball_boxes':balls,
                        'expanded_ball_exclusion_boxes':expanded,
                        'intersection_with_each_expanded_ball':[intersection_area(selected,g) for g in expanded]}
    raise ValueError('No safe fixed crop; defer instead of weakening margin or guessing absence')


def crop_pixels(raw, box):
    require(raw is not None and raw.dtype == np.uint8 and raw.shape == (240,320,3), 'Wrong native source')
    require(len(box) == 4 and all(type(v) is int for v in box), 'Integer crop coordinates required')
    left,top,right,bottom = box
    require(0 <= left < right <= 320 and 0 <= top < bottom <= 240, 'Crop out of bounds')
    return raw[top:bottom,left:right].copy()


def check(output):
    output = Path(output).resolve()
    require(output.is_relative_to(ARTIFACTS.resolve()), 'Foreign artifact folder')
    m = read(output/'manifest.json');require(m['schema'] == SCHEMA, 'Wrong crop protocol')
    base = Path(m['source_ball_dataset']);source = check_ball(base)
    check_human(source['source_dataset'])
    require(sha(base/'dataset_manifest.json') == m['source_dataset_manifest_sha256'], 'Source manifest changed')
    for p,h in m['protected_input_sha256'].items():require(sha(p) == h, 'Protected source changed: '+p)
    expected = {r['image_id']:r for r in source['records'] if r['partition'] == 'train'}
    require(len(m['candidates']) == len(expected) == 11, 'Expected one candidate per fixed11 train images')
    require({c['source_image_id'] for c in m['candidates']} == set(expected), 'Wrong negative source allowlist')
    for c in m['candidates']:
        r = expected[c['source_image_id']];require(c['geometry_proof'] == choose_crop(r), 'Crop proof changed')
        require(c['source_group'] == r['source_group'] and c['video_id'] == r['video_id'], 'Crop source identity changed')
        require(c['separately_human_reviewed'] is False and c['new_independent_image'] is False, 'False provenance')
        image = within(output,c['image']);require(sha(image) == c['image_sha256'], 'Crop bytes changed')
        raw = cv2.imread(str(within(base,r['image'])))
        require(np.array_equal(cv2.imread(str(image)),crop_pixels(raw,c['geometry_proof']['crop_xyxy'])), 'Crop pixels differ')
    for split in ('train','val'):
        original = {r['image_id']:r['image_sha256'] for r in source['records'] if r['partition'] == split}
        require(m['unchanged_positive_image_ids_and_sha256'][split] == original, 'Positive/development list changed')
    require(not (output/'labels').exists() and not (output/'data.yaml').exists(), 'Candidates are not a training dataset')
    require(m['training_started'] is False and m['yolo_labels_exported'] is False, 'Wrong stage claim')
    return m


def prepare(source_dir):
    source_dir = Path(source_dir).resolve();m = check_ball(source_dir);check_human(m['source_dataset'])
    require(m['image_counts'] == {'train':11,'val':3}, 'Frozen source counts changed')
    proofs = {r['image_id']:choose_crop(r) for r in m['records'] if r['partition']=='train'}
    output = Path(tempfile.mkdtemp(prefix='ball_negative_crop_candidates_',dir=ARTIFACTS))
    (output/'crops').mkdir();(output/'panels').mkdir();(output/'overview').mkdir()
    shutil.copy2(__file__,output/'source_snapshot.py')
    protected = {str(source_dir/'dataset_manifest.json'):sha(source_dir/'dataset_manifest.json')}
    candidates=[];panels=[];cards=[]
    for r in m['records']:
        p = within(source_dir,r['image']);require(sha(p) == r['image_sha256'], 'Source PNG changed')
        protected[str(p)]=sha(p)
        if r['partition'] != 'train':continue
        proof = proofs[r['image_id']];raw=cv2.imread(str(p));crop=crop_pixels(raw,proof['crop_xyxy'])
        iid = 'N_'+r['image_id'];destination=output/'crops'/f'{iid}.png'
        require(cv2.imwrite(str(destination),crop), 'Cannot write lossless crop')
        entry = {'candidate_id':iid,'source_image_id':r['image_id'],'source_group':r['source_group'],
            'video_id':r['video_id'],'source_split':'train','detector_partition':'train',
            'source_image_sha256':r['image_sha256'],'image':str(destination.relative_to(output)),
            'image_sha256':sha(destination),'shape_hwc':list(crop.shape),'geometry_proof':proof,
            'label_origin':'derived_by_crop_from_complete_human_ball_annotation',
            'separately_human_reviewed':False,'new_independent_image':False,
            'AI_visual_review_status':'pending','training_export_ready':False}
        candidates.append(entry)
        left=raw.copy()
        for a in proof['source_ball_boxes']:
            x1,y1,x2,y2=map(int,a['xyxy']);cv2.rectangle(left,(x1,y1),(x2-1,y2-1),(0,255,0),1)
        x1,y1,x2,y2=proof['crop_xyxy'];cv2.rectangle(left,(x1,y1),(x2-1,y2-1),(255,255,0),1)
        panel=np.zeros((264,640,3),np.uint8)
        panel[24:,:320]=left
        panel[24:,320:]=cv2.resize(crop,(320,240),interpolation=cv2.INTER_NEAREST)
        cv2.putText(panel,iid+' SOURCE / CROP CANDIDATE',(5,17),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
        filename=f'panels/{iid}.png';require(cv2.imwrite(str(output/filename),panel),'Panel write failed')
        entry['panel']=filename;panels.append(panel)
        cards.append(f'<section><h2>{html.escape(iid)}</h2><img src="{filename}"></section>')
    pages=[]
    for start in range(0,len(panels),4):
        name=f'overview/page_{start//4+1}.png'
        require(cv2.imwrite(str(output/name),np.vstack(panels[start:start+4])), 'Overview write failed');pages.append(name)
    report = {'schema':SCHEMA,'source_ball_dataset':str(source_dir),'source_dataset_manifest_sha256':sha(source_dir/'dataset_manifest.json'),
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':sha(__file__),
        'candidates':candidates,'overview':pages,'protected_input_sha256':protected,
        'unchanged_positive_image_ids_and_sha256':{s:{r['image_id']:r['image_sha256'] for r in m['records'] if r['partition']==s} for s in ('train','val')},
        'training_started':False,'yolo_labels_exported':False,'new_human_labels_created':False,
        'original_classification_validation_test_used':False,'feature_cache_changed':False,'original_video_opens':0,
        'purpose':'Train-only negative crop augmentation candidates; no deployment/inference cropping',
        'limitation':'Derived from same11 train images/eight sources, not new independent data. Depends on completeness of existing human ball annotations. Crop scale/texture distribution differs from full frames; experiment must retain full positive images and unchanged full-frame development validation.',
        'next_experiment_requirement':'Do not train from this folder directly. A separate versioned exporter/trainer and paired control must address changed image sampling and optimizer-update budget; not merely compare equal epoch counts.'}
    save(output/'manifest.json',report)
    page='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>球偵測：既有真值衍生負例候選</title><style>body{font:18px system-ui;line-height:1.8;max-width:1000px;margin:30px auto;padding:0 20px}img{max-width:100%}section{margin:25px 0}</style><h1>11張衍生負例候選：尚未訓練</h1><p>左：完整人工確認的訓練圖，綠框為既有球真值、青框為裁圖範圍。右：候選區域，最近鄰放大僅供顯示。每張裁圖避開所有球框及原生12像素邊界。</p><p>這不是新增人工確認，也不是新增獨立來源。所有資料仍來自原始分類train、且只從偵測器train裁圖；3張開發圖維持原樣，不從它們產生訓練負例。不改原三類標註、原影片、模型或MixNet特徵。</p><p>目前僅候選，未匯出labels/data.yaml、未訓練；球員、裁判或籃圈出現在球專用負例裡不代表它們是其他任務的背景。既有完整人工球標註若有漏球，此推導也會失效。下一輪需保留全圖正例、控制抽樣及optimizer更新數；不能把增加資料後相同epoch當完全公平對照。</p><p><a href="manifest.json">完整幾何證明與來源</a> · <a href="AI_review.html">AI檢查（若有）</a></p>'''+''.join(cards)+'</html>'
    (output/'index.html').write_text(page,encoding='utf-8');check(output);return output


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--check',action='store_true')
    parser.add_argument('--source-ball-dataset');parser.add_argument('--candidate-dir');a=parser.parse_args()
    if a.prepare:print(prepare(a.source_ball_dataset))
    else:print({'candidates':len(check(a.candidate_dir)['candidates']),'training_started':False})


if __name__=='__main__':main()
