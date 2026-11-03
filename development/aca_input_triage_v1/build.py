"""Publish already-performed assistant image observations, not automated vision inference.

Reads existing validation metadata/PNG hashes only. No video decoder or ML model.
Never updates the user's human_review, labels, predictions or prior reports.
"""
import hashlib
from collections import Counter
from html import escape
import json
import os
from pathlib import Path
import tempfile

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[1]
REVIEW = ROOT / 'development/aca_review_v1/artifacts/review24_ckaqp3hd'
FRAMES = ROOT / 'development/aca_frame_review_v1/artifacts/frame_compare_v0szqhxt'
CATEGORIES = {'clear':'關鍵關係較清楚', 'limited':'事件脈絡保留，細節有限', 'uncertain':'需要使用者複查輸入'}


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024), b''):h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def validate(observations, cases):
    expected={f'R{i:02d}' for i in range(1,25)}
    rows=observations['cases']
    assert len(rows)==24 and {r['review_id'] for r in rows}==expected
    assert len(cases)==24 and {r['review_id'] for r in cases}==expected
    assert all(c['split']=='validation' for c in cases)
    for row in rows:
        assert row['category'] in CATEGORIES and row['observation']
        assert (row['category']=='uncertain')==bool(row.get('question'))
        assert row.get('assistant_outcome_hypothesis') in (None,0,1)
        for name in [row.get('mainline_viewed_file','window2_02.png'),*row.get('extra_viewed_files',[])]:
            assert Path(name).name==name and name.endswith('.png')
    by_id={c['review_id']:c for c in cases}
    assert by_id['R02']['label']==0 and by_id['R02']['human_review']['human_outcome']==1
    assert by_id['R24']['label']==1 and by_id['R24']['human_review']['target_attempt_index']==2


def evidence_names(row):
    return list(dict.fromkeys(['aca50_01.png','key_comparison.png',row.get('mainline_viewed_file','window2_02.png'),
                              *row.get('extra_viewed_files',[])]))


def build():
    observations=read(FOLDER/'observations.json');human=read(REVIEW/'review_template.json')
    validate(observations,human['cases'])
    original=read(REVIEW/'sampling_summary.json')
    protected=dict(original['protected_files_sha256'])
    # Snapshot current annotations and all earlier evidence; no updates to these files.
    for directory in (REVIEW,FRAMES):
        protected.update({str(p):digest(p) for p in directory.rglob('*') if p.is_file()})
    protected.update({str(p):digest(p) for p in FOLDER.iterdir() if p.is_file()})
    for name,expected in protected.items():assert digest(Path(name))==expected,name
    cases={c['review_id']:c for c in human['cases']};results=[]
    for observation in observations['cases']:
        rid=observation['review_id'];case=cases[rid];info=read(FRAMES/rid/'comparison.json')
        assert info['review_id']==rid and info['video_id']==case['video_id'] and info['split']=='validation'
        assert info['aca_pixel_parity'] and info['mainline_pixel_parity']
        evidence=[]
        for name in evidence_names(observation):
            path=FRAMES/rid/name;assert path.is_file()
            entry={'file':str(path),'sha256':digest(path)}
            if name=='aca50_01.png':entry['source_indices']=info['timing']['aca_indices']
            else:
                page=next((p for pages in info['pages'].values() for p in pages if p['file']==name),None)
                if page:entry['source_indices']=page['source_indices']
                else:assert name=='key_comparison.png'
            evidence.append(entry)
        results.append({**observation,'video_id':case['video_id'],'split':'validation','original_label':case['label'],
            'human_review_reference':case['human_review'],'requires_user_recheck':observation['category']=='uncertain',
            'assistant_visual_assessment_only':True,'label_independently_verified':False,
            'entire_aca_50_frame_sheet_viewed':True,'all_192_mainline_frames_viewed':False,
            'raw_video_playback_performed':False,'evidence_viewed':evidence})
    artifacts=FOLDER/'artifacts';artifacts.mkdir(exist_ok=True)
    out=Path(tempfile.mkdtemp(prefix='screen24_',dir=artifacts))
    report={'schema':'aca_assistant_input_triage_v1','status':'completed','review_date':observations['review_date'],
        'case_count':24,'category_counts':dict(Counter(c['category'] for c in results)),
        'user_recheck_ids':[c['review_id'] for c in results if c['requires_user_recheck']],
        'known_label_discrepancy_ids':['R02'],'known_target_definition_resolved_by_user_ids':['R24'],
        'review_scope':'Assistant manually inspected the listed existing image sheets in this chat; this code only validates/publishes observations.',
        'limitations':['Not blind: original labels, model predictions and user answers were visible.',
                      'Not a frame-by-frame video playback, not a second-human label audit, not a calibrated visibility score.',
                      'Every complete ACA50 sheet was viewed; only listed source/mainline pages were viewed.',
                      'Viewer may rescale sheets slightly; stored PNG pixels remain unchanged.',
                      'No urgent recheck is not a quality pass; human-readable does not prove network usability.',
                      'A targeted 24-case diagnostic sample cannot estimate population accuracy or label-error rate.',
                      'Apparent detail loss and temporal complexity are hypotheses, not proven causes of model errors.'],
        'dataset_labels_changed':False,'user_human_review_changed':False,'metrics_recomputed':False,
        'model_instantiated':False,'training_started':False,'test_videos_opened':0,'cases':results}
    report_path=out/'assistant_screening.json'
    with report_path.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2);stream.write('\n')
    def href(path):return escape(os.path.relpath(path,out),quote=True)
    cards=[]
    for c in sorted(results,key=lambda c:(not c['requires_user_recheck'],c['review_id'])):
        images=''.join(f"<p>{escape(Path(e['file']).name)}</p><a href='{href(e['file'])}'><img loading='lazy' src='{href(e['file'])}' alt='{c['review_id']} {escape(Path(e['file']).name)}'></a>" for e in c['evidence_viewed'])
        cards.append(f"<article id='{c['review_id']}'><h2>{c['review_id']}：{CATEGORIES[c['category']]}</h2><p>{escape(c['observation'])}</p>"
            + (f"<p class='ask'>請你複查：{escape(c['question'])}</p>" if c['requires_user_recheck'] else "<p>暫不要求你重看；不是品質合格或標籤已獨立確認。</p>")
            +f"<p><a href='{href(FRAMES/c['review_id']/'index.html')}'>完整三路對照</a> · <a href='{escape(cases[c['review_id']]['source_url'],quote=True)}'>原片（需要時才開啟）</a></p><details><summary>我實際查看的證據影格頁</summary>{images}</details></article>")
    html="""<!doctype html><html lang='zh-Hant'><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>
<title>24部模型輸入：助理初判與3部複查</title><style>body{font:16px/1.7 system-ui;max-width:1100px;margin:28px auto;padding:0 18px;color:#202631}article{border-top:1px solid #ccd3dd;padding:20px 0}.notice,.ask{background:#fff3d6;padding:12px}img{display:block;max-width:100%;height:auto}a{color:#1857a4}summary{cursor:pointer}</style>
<h1>24部模型輸入：助理初判</h1><p class='notice'>這是看過標籤、模型預測及你先前答案後的非盲式初判，不是獨立真值審查，也不是自動視覺模型的分數。未改你的紀錄、資料標籤或成績，未讀test、未啟動訓練。</p>
<p>3部關鍵關係較清楚（R07、R12、R17）；18部事件脈絡保留但細節有限；3部需要你補查模型輸入可辨識性。沒有要求複查不等於品質合格。</p>
<p>優先只看 <a href='#R11'>R11 遮擋</a>、<a href='#R14'>R14 遠景小球</a>、<a href='#R21'>R21 模糊背景</a>。原片結果你已填過，不必重述；只需回答ACA／主線各自可判讀、不可判讀或不確定。</p>
<p>R02：保留你確認進球與原標籤不同的疑點，不再重問；R24：沿用你確認第二次投籃為目標，不刪除，也不宣稱多次進攻已證實致錯。</p>
<p>24部完整ACA50張均已看過；原片與主線僅查看下列列出的影格頁，沒有播放全部原片或看完每部192張主線影格。較大的主線與ACA同時改變解析度及時間密度，不能由此斷言提高解析度就會改善準確率。</p>
<p>下一個候選實驗：先評估維持50張／全畫面、只提高ACA解析度的成本，凍結其餘設定；本次沒有建立或啟動新訓練。R12的反例顯示不能把所有錯誤歸因為看不到球。</p>
<p><a href='assistant_screening.json'>逐部初判與證據清單</a> · <a href='provenance.json'>唯讀保護與來源指紋</a></p>
"""+''.join(cards)+'</html>\n'
    with (out/'index.html').open('x',encoding='utf-8') as stream:stream.write(html)
    for name,expected in protected.items():assert digest(Path(name))==expected,name
    with (out/'provenance.json').open('x',encoding='utf-8') as stream:
        json.dump({'protected_files_sha256':protected,'protected_files_unchanged':True,'test_used':False,
                   'new_source_video_reads':0,'image_pixels_modified':False,'reviewer':'assistant_not_blind'},stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps({'output':str(out),'case_count':24,'counts':report['category_counts'],
                      'user_recheck_ids':report['user_recheck_ids'],'protected_files_unchanged':len(protected)},ensure_ascii=False))
    return out


if __name__=='__main__':build()
