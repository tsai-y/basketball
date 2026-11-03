"""Render evidence-backed AI review separately; never change frozen proposals/labels."""
from pathlib import Path
import argparse
import html
import json

import cv2
import numpy as np

from development.yolo_train_probe_v1.probe import read, save, sha, require
from .probe import ARTIFACTS


def make_focus(run, ids):
    """Reuse the existing native16 PNGs, not another original-video read.

    Each untouched264x320 tile includes a24-pixel header and a native320x240
    frame. Six of the16 tile positions are copied; no interpolation is used.
    """
    run=Path(run).resolve();require(run.is_relative_to(ARTIFACTS),'Foreign diagnostic run')
    manifest=read(run/'manifest.json');by={r['sample_id']:r for r in manifest['videos']}
    require(len(set(ids))==len(ids) and set(ids)<=by.keys(),'Unknown/duplicate focus IDs')
    folder=run/'focus';require(not folder.exists(),'Do not overwrite focus pages');folder.mkdir()
    strips=[];pages=[];sources={}
    for sid in ids:
        p=run/by[sid]['contact_pages'][0];img=cv2.imread(str(p))
        require(img is not None and img.shape==(1056,1280,3),'Unexpected native16 contact layout')
        sources[str(p)]=sha(p)
        tiles=[img[(i//4)*264:(i//4+1)*264,(i%4)*320:(i%4+1)*320] for i in (0,2,5,8,12,15)]
        strips.append(np.hstack(tiles))
        if len(strips)==4:
            rel=f'focus/focus_{len(pages)+1:02d}.png'
            require(cv2.imwrite(str(run/rel),np.vstack(strips)),'Focus write failed');pages.append(rel);strips=[]
    if strips:
        rel=f'focus/focus_{len(pages)+1:02d}.png'
        require(cv2.imwrite(str(run/rel),np.vstack(strips)),'Focus write failed');pages.append(rel)
    save(run/'focus_plan.json',{'sample_ids':ids,'tile_positions_from_native16':[0,2,5,8,12,15],
         'source_contacts_sha256':sources,'focus_pages':pages,'new_original_video_opens':0,
         'purpose':'AI triage only, not complete event truth or interval approval'})
    return pages


def render_review(run, findings):
    run=Path(run).resolve();require(run.is_relative_to(ARTIFACTS),'Foreign run')
    report=read(run/'manifest.json');evidence=read(findings)
    require(evidence['schema']=='AI_fixed_v4_train100_review_v1','Foreign AI evidence')
    require(evidence['manifest_sha256']==sha(run/'manifest.json'),'Evidence uses another manifest')
    require(len(evidence['reviewed_sample_ids'])==100 and
            set(evidence['reviewed_sample_ids'])=={r['sample_id'] for r in report['videos']},'Review scope must be explicit100')
    for rel in evidence['viewed_pages']:
        require((run/rel).resolve().is_relative_to(run) and (run/rel).is_file(),'Missing/foreign reviewed image: '+rel)
    evidence['viewed_page_sha256']={rel:sha(run/rel) for rel in evidence['viewed_pages']}
    for finding in evidence['findings']:
        require(finding['sample_id'] in evidence['reviewed_sample_ids'] and
                set(finding['evidence_pages']) <= set(evidence['viewed_pages']),
                'Finding references an unreviewed sample/page')
    require(evidence['training_use_approved'] is False and evidence['human_ground_truth'] is False,
            'AI review cannot promote to human training truth')
    for name in ('AI_review.json','AI_review.html'):
        require(not (run/name).exists(),'Keep previous review history; do not overwrite')
    save(run/'AI_review.json',evidence)
    issues=''.join(f'<section><h2>{html.escape(r["sample_id"])}：{html.escape(r["finding"])}</h2><p>{html.escape(r["observation"])}<br>{html.escape(r["implication"])}</p><p><a href="index.html#{r["sample_id"]}">原片／提案對照</a></p>'+''.join(f'<a href="{p}"><img loading="lazy" src="{p}"></a>' for p in r['evidence_pages'])+'</section>' for r in evidence['findings'])
    page='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>100支固定v4：AI初審</title><style>body{font:18px system-ui;line-height:1.8;max-width:1400px;margin:30px auto;padding:0 18px}img{max-width:100%}section{border-top:1px solid #ccc;padding:15px 0}.warn{background:#fff0ca;padding:15px}pre{white-space:pre-wrap}</style><h1>100支固定v4：AI初審，未批准訓練</h1><p class="warn">這不是100支逐幀人工真值。看過每支固定8幀的小總覽，再針對列明案例看原生位置與邊界；沒有逐支連續播放或逐球確認結果。抽幀未見異常不能當成沒有異常。</p><p>固定100支／35來源，76支不改、24支多場景回退；新裁段採用0。76支不改不是76支乾淨：已看到特寫／圖卡漏切。24支多場景也不等於24個多投籃事件：包括同次罰球換鏡及疑似曝光誤切。此輪不調參、不重建特徵、不訓練、不改分類標籤或TANet。</p><p><a href="index.html">100支完整診斷頁</a> · <a href="AI_review.json">AI範圍、反例與限制</a> · <a href="summary.json">程式統計</a></p><h2>結論</h2><p>'''+html.escape(evidence['conclusion'])+'</p><h2>已查看的範圍</h2><p>'+html.escape(evidence['review_scope'])+'</p>'+issues+'</html>'
    (run/'AI_review.html').write_text(page,encoding='utf-8')
    return run/'AI_review.html'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    p.add_argument('--focus',nargs='+');p.add_argument('--findings',type=Path)
    args=p.parse_args();require(bool(args.focus)!=bool(args.findings),'Choose focus creation or reviewed findings rendering')
    print(make_focus(args.run,args.focus) if args.focus else render_review(args.run,args.findings))
