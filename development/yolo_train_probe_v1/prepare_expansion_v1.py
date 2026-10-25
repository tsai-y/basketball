"""Prepare source-partitioned lossless annotation candidates, never train/export labels.

All pixels come from the completed original-train video audit's native PNGs.
Detector development-validation is drawn within classification train, not from
classification validation/test. Existing development exposure is disclosed.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np

from development.data_pipeline.pipeline import AccessLedger, SplitGuard
from development.yolo_audit_v1.target_label_pilot import (
    CLASS_NAMES, FOLDER as EDITOR, PROTOCOL, validate_box, validate_review,
)
from development.yolo_audit_v1.role_metadata import ROLE_PROTOCOL
from .probe import ARTIFACTS, check_dataset, read, require, save, sha, within

SEED = 20261025
# Reviewed failure stills are selected only if their group was assigned training.
# Otherwise the held-out selection remains fixed-position and unchanged.
TRAIN_FOCUS = {'U001': (5, 9), 'U007': (1, 9), 'U009': (1, 7),
               'U013': (5, 9), 'U015': (7, 9), 'U016': (1, 9)}


def assign_sources(samples, previous_groups, seed=SEED):
    require(samples and all(s.get('split') == 'train' and str(s.get('excluded')) == '0'
                            for s in samples), 'Only original train sources allowed')
    require(len({s['video_id'] for s in samples}) == len(samples), 'Duplicate video')
    require(len({s['source_group'] for s in samples}) == len(samples), 'Expected one video per source group')
    groups = {s['source_group'] for s in samples}
    require(not groups.intersection(previous_groups), 'Previous training source in expansion audit')
    held = set()
    for shot in ('2p', '3p', 'ft', 'mp'):
        pool = [s for s in samples if s['original_class'][:-1] == shot]
        require(len(pool) >= 2, 'Need train and development-validation sources for '+shot)
        chosen = min(pool, key=lambda s: hashlib.sha256(
            f'yolo-expansion-source-v1:{seed}:{s["source_group"]}'.encode()).hexdigest())
        held.add(chosen['source_group'])
    partition = {g: 'detector_train' for g in previous_groups}
    partition.update({g: ('detector_dev_validation' if g in held else 'detector_train') for g in sorted(groups)})
    require(not set(previous_groups).intersection(held), 'Previous training source reserved as validation')
    return partition


def choose_frames(sample, partition):
    held = partition[sample['source_group']] == 'detector_dev_validation'
    quantiles = (3, 7) if held else TRAIN_FOCUS.get(sample['sample_id'], (3, 7))
    n = sample['frame_count']
    require(type(n) is int and n > 0, 'Invalid frame count')
    indices = tuple(n*q//10 for q in quantiles)
    require(len(set(indices)) == 2 and all(0 <= i < n for i in indices), 'Invalid selected frame')
    return indices


def validate_native_pair(raw, pair):
    require(raw is not None and raw.dtype == np.uint8 and raw.shape == (240,320,3), 'Invalid native PNG')
    require(pair is not None and pair.dtype == np.uint8 and pair.shape == (264,640,3), 'Invalid audit pair')
    require(np.array_equal(raw, pair[24:,:320]), 'Native image differs from lossless original audit panel')


def prepare(audit_dir, dataset_dir, *, dry_run=False):
    audit = Path(audit_dir).resolve()
    require(audit.is_relative_to(ARTIFACTS.resolve()), 'Not a local inference audit')
    files = {n: within(audit,n) for n in ('manifest.json','config.json','summary.json','run_state.json')}
    manifest, config, summary, state = (read(files[n]) for n in files)
    require(state.get('status') == 'completed' and summary.get('test_used') is False
            and summary.get('source_groups_disjoint_from_probe_training') is True, 'Incomplete/unsafe audit')
    training = check_dataset(dataset_dir)
    require(sorted(manifest['trained_source_groups']) == sorted(training['source_groups']), 'Training-source mismatch')
    require(config['weights_sha256'] == sha(Path(config['weights'])), 'Audit checkpoint changed')
    require(all(sha(Path(p)) == digest for p,digest in config['protected_source_sha256'].items()),
            'Protected inference audit input changed')
    samples = manifest['samples']
    require(len(samples) == 16, 'This fixed expansion protocol requires the 16-video audit')
    guard = SplitGuard('train', AccessLedger())
    for s in samples:
        require(s['video_id'] in guard.allowed, 'Not in canonical original train')
        official = guard.allowed[s['video_id']]
        require(all(s[k] == official[k] for k in ('source_group','original_class','binary_label','split')),
                'Audit source differs from fixed original manifest')
    partition = assign_sources(samples, training['source_groups'])
    plan = [(s, index) for s in samples for index in choose_frames(s,partition)]
    counts = Counter(partition[s['source_group']] for s, _ in plan)
    result = {'candidate_images':len(plan), 'new_sources':len(samples), 'image_partition_counts':dict(counts),
              'original_video_opens':0, 'classification_validation_test_used':False,
              'training_started':False,'yolo_labels_exported':False}
    if dry_run:
        return {**result, 'dry_run':True,'image_decodes':0,'files_written':False,'source_assignments':partition}
    protected = {str(p):sha(p) for p in files.values()}
    data = {}
    for s in samples:
        path = within(audit,f'detections/{s["sample_id"]}.json')
        obj = read(path)
        require(obj['video_id'] == s['video_id'] and len(obj['frames']) == s['frame_count'], 'Detection identity mismatch')
        require(all(f['frame_index']==i for i,f in enumerate(obj['frames'])), 'Frame sequence mismatch')
        data[s['sample_id']] = obj
        protected[str(path)] = sha(path)
    output = Path(tempfile.mkdtemp(prefix='annotation_expansion_v1_',dir=ARTIFACTS))
    (output/'images').mkdir()
    (output/'contacts').mkdir()
    metas=[]; reviews=[]; private=[]; tiles=[]
    for number,(sample,index) in enumerate(plan,1):
        sid=sample['sample_id'];iid=f'E{number:03d}';filename=f'{sid}_f{index:06d}.png'
        native = within(audit,'original_frames/'+filename)
        pair = within(audit,'frames/'+filename)
        raw=cv2.imread(str(native));viz=cv2.imread(str(pair));validate_native_pair(raw,viz)
        protected.update({str(p):sha(p) for p in (native,pair)})
        destination=output/'images'/f'{iid}.png';shutil.copy2(native,destination)
        require(sha(destination)==sha(native),'Native PNG copy changed bytes')
        detections=data[sid]['frames'][index]['detections']
        proposals=[]
        for d in detections:
            validate_box(d['xyxy'])
            if d['class_id']==0:
                proposals.append({'id':f'D{len(proposals)+1}','xyxy':d['xyxy'],'confidence':d['confidence'],
                    'origin':'five_image_overfit_prediction_not_ground_truth'})
        meta={'image_id':iid,'image_path':f'images/{iid}.png','image_sha256':sha(destination),
            'width':320,'height':240,'audit_sample_id':sid,'frame_index':index,
            'time_seconds':data[sid]['frames'][index]['time_seconds'],'proposals':proposals}
        metas.append(meta)
        reviews.append({'image_id':iid,'image_sha256':meta['image_sha256'],'status':'pending',
            'scene':'unreviewed','annotations':[],'proposals':proposals,
            'proposal_states':{p['id']:'pending' for p in proposals},
            'class_visibility':{name:'unreviewed' for name in CLASS_NAMES},'annotation_complete':False,'notes':''})
        private.append({'image_id':iid,'source_video_id':sample['video_id'],
            'source_group':sample['source_group'],'detector_partition':partition[sample['source_group']],
            'original_split':'train','original_class':sample['original_class'],'audit_sample_id':sid,
            'frame_index':index,'source_png':str(native),'image_sha256':sha(destination),
            'detector_predictions_not_ground_truth':detections})
        tile=np.zeros((264,320,3),np.uint8);tile[24:]=raw
        cv2.putText(tile,f'{iid} {sid} f{index}',(4,16),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
        tiles.append(tile)
    pilot_id=hashlib.sha256(json.dumps({'protocol':PROTOCOL,'images':metas,'partition':partition},
                       sort_keys=True,separators=(',',':')).encode()).hexdigest()
    draft={'schema':'yolo_target_label_review_v1','pilot_id':pilot_id,'protocol':PROTOCOL,
        'annotation_origin':'pending_human_review_with_weak_detector_proposals','classes':CLASS_NAMES,
        'images':reviews,'role_metadata_protocol':ROLE_PROTOCOL}
    save(output/'draft_review.json',draft)
    split_plan={'schema':'yolo_expansion_source_partition_v1','seed':SEED,'source_assignments':partition,
        'previous_training_sources':training['source_groups'],'records':private,
        'validation_scope':'development holdout within original classification train; previously inspected diagnostic sources, not untouched test',
        'train_val_source_disjoint':True,'exclude_reserved_sources_from_all_future_detector_training':True,
        'original_classification_split_changed':False,'training_export_implemented':False}
    save(output/'split_plan.json',split_plan)
    save(output/'manifest.json',{'schema':'yolo_annotation_expansion_v1','pilot_id':pilot_id,
        'protocol':PROTOCOL,'classes':CLASS_NAMES,'source_audit':str(audit),'images':metas,
        'source_sha256':protected,'source_pixels':'bitwise copy of native lossless original-frame PNG, no H264 re-decoding',
        'sampling':'two frames per source; hashed source split before annotations; heldout fixed30/70%; selected training failure cases',
        'selection_bias':'diagnostic development pool, not representative population',
        'script_sha256':sha(Path(__file__)),**result})
    for name in ('label_core.js','label_editor.js'):shutil.copy2(EDITOR/name,output/name)
    template=(EDITOR/'label_editor.html').read_text()
    require(template.count('__PILOT_DATA__')==1,'UI placeholder mismatch')
    template=template.replace('24張開發影像，尚未訓練。橘色虛線是官方person提案，不是球員真值。',
        '32張新增候選，尚未新增訓練。橘色虛線是五張過擬合模型的弱提案，可能漏框或誤框，不是球員真值。')
    template=template.replace('先做L001、L002確認規則即可，不必立刻標24張。',
        '先由AI另存初審，再逐張人工確認；不需一次填完32張。')
    inline=json.dumps({'pilot_id':pilot_id,'classes':CLASS_NAMES,'images':metas,'draft':draft},
                      ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    (output/'editor.html').write_text(template.replace('__PILOT_DATA__',inline),encoding='utf-8')
    cards=[]
    for start in range(0,len(tiles),8):
        contact=np.vstack([np.hstack(tiles[j:j+2]) for j in range(start,start+8,2)])
        name=f'contacts/page_{start//8+1}.png'
        require(cv2.imwrite(str(output/name),contact),'Contact write failed')
        cards.append(f'<h2>E{start+1:03d}–E{start+8:03d}</h2><img src="{name}">')
    title='下一批32張：來源分組已固定，標註尚未確認'
    page='<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>'+title+'</title><style>body{font:17px system-ui;max-width:1000px;margin:30px auto;padding:0 20px;line-height:1.8}img{max-width:100%}</style><h1>'+title+'</h1><p>新增12個訓練來源、4個開發驗證來源，每來源2張；沿用舊5張的4個來源只可作訓練。全部來自原始分類train，未碰分類validation/test。</p><p>沒有輸出YOLO標籤或啟動訓練；所有人工欄位仍pending。轉場／特寫先記錄，不直接當負樣本；球不確定不可填成沒有球。AI初審另存，不冒充人工確認。</p><p><a href="editor.html">開啟32張標註編輯器</a> · <a href="split_plan.json">固定來源規劃（開發用，非盲測）</a></p>'+''.join(cards)+'</html>'
    (output/'index.html').write_text(page,encoding='utf-8')
    status=validate_review(draft,draft);require(status['training_ready_images']==0,'Pending draft cannot become training truth')
    require(all(sha(Path(p))==v for p,v in protected.items()),'Protected audit changed')
    check_dataset(dataset_dir)
    save(output/'initial_status.json',{**status,**result,'inputs_unchanged':True})
    return {**result,'output':str(output),'training_ready_images':0}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run',action='store_true');mode.add_argument('--prepare',action='store_true')
    parser.add_argument('--audit-run',required=True);parser.add_argument('--dataset-dir',required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.audit_run,args.dataset_dir,dry_run=args.dry_run),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
