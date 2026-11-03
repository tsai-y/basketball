"""Export14 train /4 development ball-only images; no training implementation."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import argparse
import shutil
import tempfile

import cv2
import yaml

from .probe import ARTIFACTS,read,save,sha,require,within,xyxy_to_yolo
from .ball_only_probe_v1 import check as check_base,runtime_at
from .ball_source_review_v2 import (
    PROTOCOL,check_pending,protect_check,accept_other_ball_images,accept_current_ball_image,check_proposal,
)

SCHEMA='human_confirmed_ball_only_source_expansion_dataset_v2'
NEW_IDS={'B001','B006','B012','B029'}


def human_rows(progress,draft):
    """Validate immutable geometry separately from mutable acceptance state."""
    require(progress['protocol']==draft['protocol']==PROTOCOL,'Wrong annotation protocol')
    require(progress.get('batch_annotation_complete') is True and progress['pending_ids']==[]
            and progress['next_human_question'] is None,'Human batch incomplete')
    require(progress['annotation_complete_images']==4 and progress['training_ready_images']==3,'Wrong counts')
    by_id={r['image_id']:r for r in draft['images']}
    require(set(by_id)=={r['image_id'] for r in progress['images']}==NEW_IDS and len(progress['images'])==4,'Wrong IDs')
    mutable={'status','human_confirmed','visible_ball_completeness_confirmed','training_ready'}
    for row in progress['images']:
        original=by_id[row['image_id']]
        require(all(row[k]==v for k,v in original.items() if k not in mutable),'Source metadata or proposed box changed')
        require(row['status']=='human_confirmed_ball_only_full_image' and row['human_confirmed'] is True
                and row['visible_ball_completeness_confirmed'] is True and row['annotation_complete'] is True,
                'Complete human ball confirmation required')
        require(row['source_split']=='train' and row['detector_partition'] in ('train','development'),'Original classification train only')
        require(row['training_ready']==(row['detector_partition']=='train'),'Development cannot become training')
        check_proposal(row['ball_boxes_proposed'])
        expected=[{'id':f'ball{i}','class_id':0,'xyxy':box,'origin':'AI_box_accepted_in_direct_human_ball_only_review'}
                  for i,box in enumerate(row['ball_boxes_proposed'],1)]
        require(row['annotations']==expected,'Accepted box class/geometry differs from shown proposal')
    require(set(progress['detector_train_confirmed_ids'])=={'B001','B006','B012'} and
            progress['detector_development_confirmed_ids']==['B029'],'Wrong reserved partitions')
    return deepcopy(progress['images'])


def collect_review(review_dir):
    review_dir=Path(review_dir).resolve();check_pending(review_dir)
    draft_path=within(review_dir,'review_draft.json');progress_path=within(review_dir,'human_review_progress.json')
    draft,progress=read(draft_path),read(progress_path);rows=human_rows(progress,draft)
    pool=Path(draft['pool']);pool_manifest=read(pool/'manifest.json');protect_check(pool_manifest)
    correction_path=within(review_dir,'human_location_correction.json');correction=read(correction_path)
    require(correction['image_id']=='B006','Unexpected corrected image')
    events=progress['human_confirmation_events'];require(len(events)==2,'Expected two explicitly scoped human replies')
    first,second=events
    first_path=within(review_dir,'human_confirmation_others_v1.json')
    require(first==read(first_path) and first['user_reply']=='其餘都正常','Other-images event changed')
    require(first['source']=='direct_human_chat_reply' and set(first['accepted_image_ids'])==NEW_IDS-{'B006'},'Wrong first event scope')
    require(first['source_review_draft_sha256']==sha(draft_path) and first['source_location_correction_sha256']==sha(correction_path),'First event input changed')
    history=Path(second['history_dir']).resolve();require(history.is_relative_to(review_dir),'Foreign human event history')
    second_path=within(history,'event.json');before_path=within(history,'progress_before.json')
    require(second==read(second_path) and second['source']=='direct_human_chat_reply'
            and second['accepted_image_ids']==['B006'],'Corrected-image event changed')
    require(second['source_progress_before_sha256']==sha(before_path) and second['source_review_draft_sha256']==sha(draft_path),'Second event input changed')
    before=read(before_path);replayed_first=accept_other_ball_images(draft,'B006',first['user_reply'])
    require(replayed_first['images']==before['images'],'First acceptance semantic replay failed')
    require(second['question']==before['next_human_question'],'Question snapshot differs')
    replayed_final=accept_current_ball_image(before,second['user_reply'])
    require(replayed_final['images']==progress['images'],'Final acceptance semantic replay failed')
    by_id={r['image_id']:r for r in rows};coverage=[]
    protected={str(p):sha(p) for p in (draft_path,progress_path,first_path,second_path,before_path,correction_path,pool/'manifest.json')}
    for event in events:
        require(event['protocol']==PROTOCOL and bool(event['user_reply']),'No direct scoped human reply')
        evidence=event['accepted_boxes_and_display_evidence']
        require({e['image_id'] for e in evidence}==set(event['accepted_image_ids']),'Missing acceptance evidence')
        for e in evidence:
            r=by_id[e['image_id']];coverage.append(r['image_id'])
            require(e['ball_boxes']==r['ball_boxes_proposed'] and e['image_sha256']==r['image_sha256']
                    and e['shown_panel_sha256']==r['shown_panel_sha256'] and e['shown_panel']==r['shown_panel'],'Accepted version mismatch')
            panel=within(review_dir,e['shown_panel']);image=within(pool,r['image'])
            require(sha(panel)==r['shown_panel_sha256'] and sha(image)==r['image_sha256'],'Human accepted pixels changed')
            protected.update({str(panel):sha(panel),str(image):sha(image)})
    require(len(coverage)==4 and set(coverage)==NEW_IDS,'Human event coverage incomplete/duplicate')
    evidence=Path(correction['evidence_path']);require(sha(evidence)==correction['evidence_sha256'],'Location evidence changed')
    protected[str(evidence)]=sha(evidence)
    return rows,protected,pool_manifest


def validate_partitions(records,assignments,counts=None):
    require(records and len({r['image_id'] for r in records})==len(records),'Duplicate IDs')
    for r in records:
        require(r['source_split']=='train' and r['partition'] in ('train','val'),'Original train only')
        expected='detector_train' if r['partition']=='train' else 'detector_dev_validation'
        require(assignments.get(r['source_group'])==expected,'Source reservation violated')
        require(r['annotations'] and all(a['class_id']==0 for a in r['annotations']),'Only positive ball class0 labels')
        check_proposal([a['xyxy'] for a in r['annotations']])
    for field in ('source_group','video_id','image_sha256'):
        a={r[field] for r in records if r['partition']=='train'};b={r[field] for r in records if r['partition']=='val'}
        require(a and b and a.isdisjoint(b),'Train/development overlap: '+field)
    if counts is not None:require(dict(Counter(r['partition'] for r in records))==counts,'Unexpected split counts')


def remaining_unlabeled(inputs,used_sources):
    require(len({r['item_id'] for r in inputs})==len(inputs),'Duplicate diagnostic IDs')
    return [deepcopy(r) for r in inputs if r['ground_truth'] is None and r['source_group'] not in used_sources]


def expected_records(base,review_rows,pool_manifest):
    require(base['source_assignments']==pool_manifest['source_assignments'],'Different source reservations')
    rows=[]
    for r in base['records']:
        rows.append({**deepcopy(r),'annotation_origin':'inherited_human_ball_v1',
                     'source_png':str(Path(pool_manifest['source_ball_dataset'])/r['image'])})
    used={r['source_group'] for r in rows}
    for r in review_rows:
        require(r['source_group'] not in used,'New source already used in base dataset')
        pool=Path(pool_manifest['_pool_path'])
        rows.append({'image_id':r['image_id'],'source_split':'train','source_group':r['source_group'],
            'video_id':r['video_id'],'original_class':r['original_class'],'frame_index':r['frame_index'],
            'partition':'train' if r['detector_partition']=='train' else 'val',
            'image_sha256':r['image_sha256'],'annotations':deepcopy(r['annotations']),
            'annotation_origin':'human_ball_only_v2','annotation_protocol':PROTOCOL,
            'source_png':str(pool/r['image'])})
    for r in rows:
        r['image']=f'images/{r["partition"]}/{r["image_id"]}.png';r['label']=f'labels/{r["partition"]}/{r["image_id"]}.txt'
    validate_partitions(rows,base['source_assignments'],{'train':14,'val':4});return rows


def prepare(base_dir,review_dir,reference):
    base_dir,review_dir,reference=map(lambda p:Path(p).resolve(),(base_dir,review_dir,reference))
    require(reference.is_relative_to(ARTIFACTS.resolve()),'Foreign diagnostic reference')
    base=check_base(base_dir);new,protected,pool=collect_review(review_dir)
    require(Path(pool['source_ball_dataset'])==base_dir,'Review came from different base')
    pool['_pool_path']=read(review_dir/'review_draft.json')['pool']
    records=expected_records(base,new,pool)
    for r in records:require(sha(r['source_png'])==r['image_sha256'],'Source PNG changed')
    cfg=read(reference/'config.json');require(len(cfg['inputs'])==53,'Fixed53 diagnostic reference required')
    require(cfg['parameters']=={'imgsz':640,'conf':.25,'iou':.7,'max_det':100,'half':False},'Reference parameters changed')
    selected=remaining_unlabeled(cfg['inputs'],{r['source_group'] for r in records})
    require(len(selected)==30 and len({r['source_group'] for r in selected})==6,'Expected30 remaining unlabeled diagnostics')
    for r in selected:require(sha(r['path'])==r['image_sha256'],'Diagnostic PNG changed')
    protected.update({str(base_dir/'dataset_manifest.json'):sha(base_dir/'dataset_manifest.json'),
                      str(reference/'config.json'):sha(reference/'config.json')})
    protected.update({r['source_png']:r['image_sha256'] for r in records})
    protected.update({r['path']:r['image_sha256'] for r in selected})
    protected[base['initializer']]=base['initializer_sha256']
    output=Path(tempfile.mkdtemp(prefix='ball_dataset_v2_',dir=ARTIFACTS))
    for name in ('images/train','images/val','labels/train','labels/val','inputs'):(output/name).mkdir(parents=True)
    shutil.copy2(__file__,output/'source_snapshot.py')
    save(output/'inputs/base_manifest.json',base);save(output/'inputs/review_progress.json',read(review_dir/'human_review_progress.json'))
    initializer=output/'inputs/initializer.pt';shutil.copy2(base['initializer'],initializer)
    require(sha(initializer)==base['initializer_sha256'],'Initializer copy changed')
    files={}
    for r in records:
        image=output/r['image'];label=output/r['label'];shutil.copy2(r['source_png'],image)
        label.write_text('\n'.join(xyxy_to_yolo(0,a['xyxy']) for a in r['annotations'])+'\n',encoding='utf-8')
        for p in (image,label):files[str(p.relative_to(output))]=sha(p)
    data={'path':str(output),'train':'images/train','val':'images/val','names':{0:'ball'}}
    (output/'data.yaml').write_text(yaml.safe_dump(data,sort_keys=False),encoding='utf-8')
    diagnostics=[]
    for r in records:
        if r['partition']!='val':continue
        diagnostics.append({'item_id':r['image_id'],'path':str(output/r['image']),'image_sha256':r['image_sha256'],
            'video_id':r['video_id'],'source_group':r['source_group'],'frame_index':r['frame_index'],
            'ground_truth':[{'class_id':1,'xyxy':a['xyxy']} for a in r['annotations']],
            'scope':'human_ball_only_development_NOT_formal_test'})
    diagnostics+=selected
    diag={'inputs':diagnostics,'parameters':cfg['parameters'],'images':34,'human_development_images':4,
          'unlabeled_images':30,'all_actual_train_and_development_sources_excluded_from_unlabeled_diagnostics':True,
          'old53_inputs_not_reused_unchanged':True,'ground_truth_class_id1_is_comparison_only_training_class_id0':True,
          'generalization_metrics_valid':False,'old_and_new_models_must_be_evaluated_on_same_updated_inputs':True}
    save(output/'diagnostic_plan.json',diag)
    plan={'status':'prepared_not_started','initializer':str(initializer),'initializer_sha256':sha(initializer),
          'initialization':'same preserved COCO sports-ball semantic initializer; not previous fitted best or negative-crop final',
          'protocol_candidate':'Original ball_only head-training policy, not all-BN-frozen fixed330 negative experiment',
          'feature_extractor':'Freeze all modules before Detect, including their BN buffers',
          'maximum_epochs':100,'patience':20,'imgsz':640,'batch_and_nbs':2,'optimizer':'AdamW','lr0':.0001,
          'AMP':False,'augmentation':False,'negative_crops_used':False,
          'next_gate':'Implement/verify14-4-aware trainer, then independent1-epoch smoke before full training',
          'v1_hardcoded_11_3_trainer_not_directly_compatible':True,'training_started':False}
    save(output/'training_plan.json',plan)
    for p in [output/'data.yaml',output/'diagnostic_plan.json',output/'training_plan.json',initializer,
              output/'source_snapshot.py',output/'inputs/base_manifest.json',output/'inputs/review_progress.json']:
        files[str(p.relative_to(output))]=sha(p)
    report={'schema':SCHEMA,'base_dataset':str(base_dir),'review_dir':str(review_dir),'records':records,
        'source_assignments':base['source_assignments'],'image_counts':{'train':14,'val':4},
        'ball_counts':{s:sum(len(r['annotations']) for r in records if r['partition']==s) for s in ('train','val')},
        'source_group_counts':{s:len({r['source_group'] for r in records if r['partition']==s}) for s in ('train','val')},
        'initializer':str(initializer),'initializer_sha256':sha(initializer),'dataset_file_sha256':files,
        'protected_source_sha256':protected,'newly_added_ids':sorted(NEW_IDS),
        'training_started':False,'labels_exported':True,'negative_crops_used':False,
        'source_annotations_modified':False,'previous_models_overwritten':False,'feature_cache_changed':False,
        'original_classification_validation_test_used':False,'generalization_metrics_valid':False,
        'limitations':'18 positive images, easy-visible-ball AI selection plus scoped human confirmation, previously inspected sources, only4 development images; no formal performance claim'}
    save(output/'dataset_manifest.json',report);check(output)
    page='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>球專用資料v2：14train／4development</title><style>body{font:18px system-ui;line-height:1.8;max-width:1000px;margin:30px auto;padding:0 20px}</style><h1>獨立資料v2已匯出，尚未訓練</h1><p>14張訓練、4張開發圖，全部320×240原生PNG，單類ball=0，每張1個人工確認球框；train11個來源群組、development3個來源群組，來源／影片／影像SHA分離。舊11/3資料、原標註與模型不變。</p><p>新增B001/B006/B012為train，B029保持development；B006用人工更正後的右側球框。新標註只確認球，未冒稱player/rim/角色完成。不加入衍生負例、未使用原始分類validation/test、不改MixNet特徵。</p><p>舊53圖診斷有部分來源現在進入新train或development，不能原封重用：更新計畫為4張有真值開發圖＋30張未標註的其他來源全圖，共34張。未標註檢出數仍不是recall，所有來源曾作開發檢查，不是正式盲測。</p><p>下一步須驗證適用14/4的新trainer並跑獨立1epoch試跑；目前沒有訓練、沒有新模型成績。4張開發圖仍太少，不能宣稱可靠泛化。</p><p><a href="data.yaml">YOLO資料設定</a> · <a href="dataset_manifest.json">來源、框、SHA與限制</a> · <a href="diagnostic_plan.json">更新後34圖診斷計畫</a> · <a href="training_plan.json">訓練前計畫</a> · <a href="preflight.json">CPU預檢（若已完成）</a></p></html>'''
    (output/'index.html').write_text(page,encoding='utf-8');return output


def check(output):
    output=Path(output).resolve();require(output.is_relative_to(ARTIFACTS.resolve()),'Foreign data version')
    m=read(output/'dataset_manifest.json');require(m['schema']==SCHEMA,'Wrong expanded dataset schema')
    base=check_base(m['base_dataset']);new,_,pool=collect_review(m['review_dir'])
    pool['_pool_path']=read(Path(m['review_dir'])/'review_draft.json')['pool']
    require(m['records']==expected_records(base,new,pool),'Export rows differ from confirmed source rows')
    for p,h in m['protected_source_sha256'].items():require(sha(p)==h,'Protected source changed')
    for p,h in m['dataset_file_sha256'].items():require(sha(within(output,p))==h,'Exported file changed')
    require(yaml.safe_load((output/'data.yaml').read_text())=={'path':str(output),'train':'images/train','val':'images/val','names':{0:'ball'}},'Data config changed')
    validate_partitions(m['records'],m['source_assignments'],{'train':14,'val':4})
    for split in ('train','val'):
        rows=[r for r in m['records'] if r['partition']==split]
        require({p.name for p in (output/f'images/{split}').iterdir()}=={r['image_id']+'.png' for r in rows},'Unexpected image')
        require({p.name for p in (output/f'labels/{split}').iterdir()}=={r['image_id']+'.txt' for r in rows},'Unexpected label')
        for r in rows:
            require(sha(output/r['image'])==r['image_sha256'],'Image bytes differ')
            image=cv2.imread(str(output/r['image']));require(image is not None and image.shape==(240,320,3),'Wrong native pixels')
            require((output/r['label']).read_text().splitlines()==[xyxy_to_yolo(0,a['xyxy']) for a in r['annotations']],'YOLO geometry differs from acceptance')
    diag=read(output/'diagnostic_plan.json');used={r['source_group'] for r in m['records']}
    unlabeled=[r for r in diag['inputs'] if r['ground_truth'] is None]
    require(len(diag['inputs'])==34 and len(unlabeled)==30 and all(r['source_group'] not in used for r in unlabeled),'Updated diagnostic source leakage')
    require(not m['training_started'] and not m['negative_crops_used'],'Wrong preparation stage')
    return m


def preflight(output):
    output=Path(output).resolve();m=check(output);runtime_at(output)
    import torch
    from ultralytics import YOLO
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.cfg import get_cfg
    torch.set_num_threads(4);data={'names':{0:'ball'},'nc':1,'channels':3}
    report={'status':'running','device':'cpu','training_started':False,'optimizer_created':False,'backward_called':False,'splits':{}}
    for split in ('train','val'):
        dataset=YOLODataset(img_path=str(output/f'images/{split}'),imgsz=640,batch_size=2,
                            augment=False,rect=False,cache=False,hyp=get_cfg(),data=data,task='detect')
        expected={str((output/r['image']).resolve()) for r in m['records'] if r['partition']==split}
        require({str(Path(p).resolve()) for p in dataset.im_files}==expected,'Loader escaped allowlist')
        count=0
        for i in range(len(dataset)):
            item=dataset[i];require(tuple(item['img'].shape)==(3,640,640) and item['img'].dtype==torch.uint8,'Wrong loader tensor')
            require(item['cls'].numel()==1 and torch.equal(item['cls'],torch.zeros_like(item['cls'])),'Wrong single-ball label')
            require(torch.isfinite(item['bboxes']).all().item() and (item['bboxes']>=0).all().item() and (item['bboxes']<=1).all().item(),'Invalid YOLO targets')
            count+=1
        require(count==m['image_counts'][split],'Loader count differs')
        report['splits'][split]={'images_loaded':count,'all_tensors_3_640_640_uint8':True,'all_single_ball_targets_valid':True,'allowlist_checked':True}
    wrapper=YOLO(m['initializer']);model=wrapper.model.float().eval()
    require(wrapper.names=={0:'ball'} and model.model[-1].nc==1,'Wrong initializer semantic class')
    activation=type(model.model[9].cv1.act).__name__;require(activation=='SiLU','Checkpoint structure unexpectedly reconstructed')
    with torch.inference_mode():result=model(torch.zeros(1,3,64,64))[0]
    require(result.shape[:2]==(1,5) and torch.isfinite(result).all().item(),'Initializer forward invalid')
    report.update(status='completed',initializer_semantic_class_checked=True,initializer_sha256=sha(m['initializer']),
                  SPPF_cv1_activation=activation,synthetic_forward_shape=list(result.shape),synthetic_forward_finite=True,
                  source_data_unchanged=True,model_weights_file_unchanged=sha(m['initializer'])==m['initializer_sha256'])
    check(output);save(output/'preflight.json',report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group(required=True)
    for name in ('prepare','check','preflight'):mode.add_argument('--'+name,action='store_true')
    p.add_argument('--base-dataset');p.add_argument('--review-dir');p.add_argument('--reference');p.add_argument('--dataset-dir');a=p.parse_args()
    if a.prepare:print(prepare(a.base_dataset,a.review_dir,a.reference))
    elif a.check:print({'image_counts':check(a.dataset_dir)['image_counts'],'training_started':False})
    else:print(preflight(a.dataset_dir))


if __name__=='__main__':main()
