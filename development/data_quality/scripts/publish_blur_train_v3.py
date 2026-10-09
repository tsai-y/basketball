#!/usr/bin/env python3
"""Publish the fixed smoke report from verified derived outputs; never reads videos."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import audit_blur_train_v3 as s
import verify_blur_train_v3 as verify

REPORT = s.ROOT/'docs/data_quality/blur_screening_smoke_v3.json'
MARKDOWN = REPORT.with_suffix('.md')
EVIDENCE = s.ARTIFACTS/'blur_train_v3_validation'


def table(headers, rows):
    return ['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |'] + [
        '| '+' | '.join(str(value) for value in row)+' |' for row in rows]


def markdown(r):
    a=r['aggregate']; perf=r['performance']; est=r['full_train_extrapolation']; checks=r['checks']
    lines=['# Basketball-51 Blur Train Scanner v3 Smoke Report','',
        '**SCANNER IMPLEMENTATION COMPLETE — 僅完成固定 20 支 train smoke；完整 7,193 支尚未執行。**','',
        f"開始 HEAD `{r['initial_commit']}`，Pilot v2 已提交且工作樹乾淨。沒有 git add、commit 或 push；既有 tracked 檔僅預期更新 README 與追加 progress log。",'',
        'v3 是離線資料品質候選掃描器；沒有正式清理、排除、刪格、拼接、重編碼、增強、特徵擷取、checkpoint 或模型訓練／評估。Gate 1C／原 Experiment D 維持 paused。','',
        '[方法與防護](blur_screening_method_v3.md)；[完整 Ubuntu/tmux 手冊](../../development/data_quality/runbooks/blur_train_v3_tmux.md)；[固定 smoke manifest](../../development/data_quality/smoke_manifest_v3.csv)。','',
        '## 凍結規則與版本','',
        'Primary 僅 Laplacian variance；log-score 中心 3 秒 rolling median/MAD，scale=max(1.4826×MAD,0.05)，log floor=1e-12，z≤−2.0，連續 duration≥0.16 秒。Tenengrad 分數／local z 繼續保存，但不使用 AND/OR 改變候選。Prompt 與已提交 v2 證據沒有差異，也未重調參數。','',
        f"v3 config SHA-256：`{r['config_hash']}`；scanner/dependency fingerprint：`{r['scanner']['sha256']}`。",'',
        '| v2 frozen evidence | SHA-256 |','| --- | --- |']
    for name,sha in sorted(r['frozen_evidence'].items()):lines.append(f'| {name} | `{sha}` |')
    lines+=['','## 樣本、候選與每支執行時間','',
        f"seed=20261009，固定 {a['selected']} 個 train IDs、20 來源、Make/Miss 各 10。先保留既有 train FPS metadata 所示的兩群，再交替 label、優先未使用來源、SHA-256 破同分；不讀品質分數選樣，失敗不替換。",'',
        f"成功解碼 {a['processed_frames']:,} 幀；候選 {a['candidate_frames']} 幀（{a['candidate_ratio']:.4%}）、{a['candidate_segments']} 區段、涉及 {a['affected_videos']} 支；實際 decode failures=0。所有影片均有 16/32/64 非 candidate window（僅資訊，非模型長度選擇）。",'']
    lines+=table(['video_id','label','source','FPS','frames','候選幀／段','秒','輸出 bytes'],[
        [v['video_id'],v['label'],v['source_group'],v['fps'],v['processed_frame_count'],
         f"{v['candidate_frames']}/{len(v['candidate_segments'])}",f"{v['processing_seconds']:.6f}",v['output_bytes']]
        for v in r['videos']])
    lines+=['','每支秒數包含 probe、decode、評分、local z 與 frame CSV 寫入；不包含最後 summary/completion 寫入。總 scanner wall time 另含初始化、所有寫入與 aggregate，初次是在 strace 下量測。','',
        '## 描述性分布','']
    for dim in ('label','fps_rational','source_group'):
        lines += [f'### {dim}','']+table(['群組','影片','影格','候選幀','候選比例'],[
            [name,g['videos'],g['frames'],g['candidate_frames'],f"{g['candidate_frames']/g['frames']:.4%}"]
            for name,g in sorted(a['groups'][dim].items())])+['']
    lines+=['FPS 實測為 23.976×16、29.97×4，包含 145/146/180 幀影片。來源代碼未核實為 game ID；本樣本刻意分散來源且每來源一支，不能估計 full train 的模糊盛行率，不能把 label/FPS/source 差異當因果。','',
        '## 時間、容量與記憶體','']
    lines+=table(['量測','值'],[
        ['初次 scanner wall seconds',f"{perf['initial_wall_seconds']:.6f}"],
        ['每支平均／P50／P90 seconds',' / '.join(f'{perf[k]:.6f}' for k in ('mean_video_seconds','p50_video_seconds','p90_video_seconds'))],
        ['每支輸出平均／P50／P90 bytes',' / '.join(f'{perf[k]:.1f}' for k in ('mean_video_bytes','p50_video_bytes','p90_video_bytes'))],
        ['20 支 per-video files bytes',perf['per_video_bytes']],
        ['目前主 smoke run bytes（含 resume/rebuild snapshots）',perf['run_bytes']],
        ['scanner peak RSS MiB',f"{perf['scanner_peak_rss_kib']/1024:.3f}"],
        ['子程序 peak RSS KiB',perf['child_peak_rss_kib']],
        ['父＋子 peak RSS 保守上界 MiB',f"{perf['process_tree_rss_upper_bound_kib']/1024:.3f}"],
        ['20/20 已完成 resume seconds',f"{perf['resume_seconds']:.6f}"],
        ['aggregate rebuild seconds',f"{perf['rebuild_seconds']:.6f}"]])
    lines+=['','RSS 由 Linux getrusage 取得；父＋子各自峰值相加是總記憶體的保守上界，**不是同步採樣的 process-tree 峰值**，可能包含 fork 時繼承的 RSS。單 worker、單畫面、單影片最多 10,000 筆分數，沒有保存整個資料集的分數或影像；詳細邊界見方法文件。','',
        '## Resume／recovery 與一致性','',
        '- 初次 20/20 完成；普通 resume 跳過 20 支、attempted=0，access ledger 未增加。',
        '- 同設定另開獨立目錄重跑同一 20 支，逐影片 frames.csv 位元組完全一致；summary 排除 processing_seconds 後完全一致。',
        '- Derived-only fixture：複製 18 支完成結果、建立 1 支無 completion 的部分輸出、以 mock 空 decoder 產生 1 支 failure。建置 fixture 沒有開資料集影片。',
        '- 普通 resume 保留 incomplete/failed、exit 2；--retry-failures 跳過 18 支，只重試原 smoke 中的 2 支，最後 20/20，與初次科學結果一致。舊 partial、failed completion、decode failure ledger 都保留。',
        '- Unit tests 另覆蓋真正 KeyboardInterrupt、輸出 hash 毀損、atomic rename 失敗、輸入 identity 改變及 concurrent writer lock。',
        '- 相同參數但不同 config bytes hash 的實際 CLI resume 被拒絕（exit 2）；error ledger 留存，video access ledger 不變。',
        '- Aggregate 從 per-video outputs 重建一致；獨立 median/MAD、segment/window、quantile、JSON/Markdown/CSV 檢查通過。','',
        f"單元測試：v3 {checks['v3_unit_tests']}、v2 {checks['v2_unit_tests']}、v1 {checks['v1_unit_tests']}，全部通過。Syntax/import、CLI help、smoke dry-run、全 train 正向抽樣 guard、全部 held-out ID/path 字串拒絕、seed 重現與 v2 hashes 通過。Full-train 缺确认拒絕；本輪未執行 full-train dry-run，因其也要求禁止由 Codex 使用的確認 flag。",'',
        '## 存取及原始檔保護證據','']
    lines+=table(['執行','唯一 IDs','guarded opens','probes','decodes','唯讀 open syscalls'],[
        [name,ev['access']['unique_video_ids'],ev['access']['guarded_opens'],ev['access']['train_probes'],ev['access']['train_decodes'],ev['syscall_trace']['video_open_syscalls']]
        for name in ('initial','repeat','recovery') for ev in [r['verification_runs'][name]]])
    lines+=['',
        '三次實際存取涵蓋同一固定 20 個 IDs：初次 20、重跑 20、恢復重試 2；合計 42 probe/decode sessions，各次都有 durable access ledger 與獨立 strace。含 descriptor reopen 共 168 次唯讀 open syscall。**validation/test opens、probes、decodes 均為 0**，沒有 encoder 或 dataset write syscall。',
        '最初 strace capability test 被 sandbox ptrace 限制，未執行 scanner；取得允許後成功追蹤。第一次實際 smoke 嘗試遇 ledger 參數命名錯誤，在影片 open 前停止；修正並新增回歸測試，原目錄保留為 blur_train_v3_smoke_preflight_failed，該 trace 影片開檔為 0。','',
        f"初始 {r['protected_files']['initial_tracked_files']} 個 tracked files 的 SHA-256 比對：{r['protected_files']['unchanged']} 個不變；預期修改 {', '.join(r['protected_files']['expected_modified'])}。Progress log 舊內容以 HEAD 前綴比對保留。20 支原影片 stat identity 前後一致，未計算全資料集影片 hash。原始 manifest/split、v1/v2、baseline、runs 與 frozen evidence 未改；沒有資料刪除。",'',
        '## 完整 7,193 支的 smoke extrapolation','',
        f"以初次 wall time /20×7,193，預估 {est['wall_minutes']:.2f} 分鐘；以每支 P50/P90×7,193 分別為 {est['p50_scaled_minutes']:.2f}/{est['p90_scaled_minutes']:.2f} 分鐘。這些是 smoke extrapolation，**不是信賴區間或實際時間保證**；冷 cache、磁碟、主機負載、失敗及 retry 都會改變時間。",'',
        f"每支檔案容量線性外推約 {est['per_video_gib']:.3f} GiB；將本次 run 的 logs/多次 snapshots 一起按影片數保守外推約 {est['run_gib']:.3f} GiB。建議預留至少 2 GiB，若保存額外 strace、重跑／重試目錄需再增加。Full-run 預設只寫 access ledger，不產生 contact sheets 或逐影片影像。",'',
        '正式執行請先 review、commit/push v3，記錄 commit/config hash，依 [tmux 手冊](../../development/data_quality/runbooks/blur_train_v3_tmux.md) 執行 dry-run、scan、detach/attach、監看、resume、retry、rebuild、verify。推薦 output：`development/data_quality/artifacts/blur_train_v3_full_train/`。本次不自動執行。','',
        '## 未決政策','',
        'Local z 可能漏掉整片持續低品質；低絕對分數只列診斷。沒有人工模糊真值、球／籃框 ROI、真實模糊嚴重度或下游分類效益證據。來源/FPS/紋理/字幕與拍攝情境混雜，品質定義、人工複核、錯誤成本及任何保留／排除／清理策略仍待決定。Scanner implementation complete 不代表資料已清理，也不代表全量結果已完成。','']
    return '\n'.join(lines)


def publish(out, evidence):
    s.need(not REPORT.exists() and not MARKDOWN.exists(),'Refusing existing tracked smoke report')
    ctx=s.read_json(out/'run.json')['context']
    s.need(ctx['scope']=='smoke' and len(ctx['selection'])==20,'Publisher only accepts the full fixed 20-video smoke')
    s.need(ctx['selection']==s.choose_smoke(s.Guard(),ctx['seed']),'Unexpected smoke selection')
    runs={}
    for name,directory,trace,compare in [
        ('initial',out,'smoke_verified_syscalls.log',s.ARTIFACTS/'blur_train_v3_smoke_repeat'),
        ('repeat',s.ARTIFACTS/'blur_train_v3_smoke_repeat','repeat_syscalls.log',out),
        ('recovery',s.ARTIFACTS/'blur_train_v3_smoke_recovery','recovery_syscalls.log',out)]:
        runs[name]=verify.verify(directory,evidence/trace,compare,evidence/'initial_tracked_hashes.json')
    s.need(verify.trace_audit(evidence/'smoke_syscalls.log',set())['video_open_syscalls']==0,'Preflight opened a video')
    tests={}
    for name,filename,count in [('v3','unit_tests.log',15),('v2','v2_tests.log',10),('v1','v1_tests.log',6)]:
        log=(evidence/filename).read_text()
        s.need(f'Ran {count} tests' in log and '\nOK\n' in log,'Unit test evidence missing')
        tests[name+'_unit_tests']=count
    s.need(s.read_json(evidence/'config_mismatch_test.json')['video_access_ledger_unchanged'],'Config mismatch test failed')
    s.need(s.read_json(evidence/'source_identity_check.json')['stat_identity_unchanged'],'Input identities changed')
    videos=[]
    for row in ctx['selection']:
        attempt=s.latest_attempt(out,row['video_id']); summary=s.read_json(attempt/'summary.json')
        summary['output_bytes']=sum(f.stat().st_size for f in attempt.iterdir())
        videos.append(summary)
    invocations=[s.read_json(f) for f in sorted((out/'invocations').glob('*.json'))]
    first=next(x for x in invocations if x['attempted']==20)
    resume=next(x for x in invocations if x['skipped']==20 and x['attempted']==0)
    rebuild=next(x for x in invocations if '--rebuild-summary' in x['command'])
    times=[v['processing_seconds'] for v in videos];sizes=[v['output_bytes'] for v in videos]
    perf=dict(initial_wall_seconds=first['elapsed_seconds'],mean_video_seconds=float(np.mean(times)),
        p50_video_seconds=float(np.quantile(times,.5)),p90_video_seconds=float(np.quantile(times,.9)),
        mean_video_bytes=float(np.mean(sizes)),p50_video_bytes=float(np.quantile(sizes,.5)),
        p90_video_bytes=float(np.quantile(sizes,.9)),per_video_bytes=sum(sizes),
        run_bytes=sum(f.stat().st_size for f in out.rglob('*') if f.is_file()),
        scanner_peak_rss_kib=first['peak_rss_kib'],child_peak_rss_kib=first['child_peak_rss_kib'],
        process_tree_rss_upper_bound_kib=first['peak_rss_kib']+first['child_peak_rss_kib'],
        resume_seconds=resume['elapsed_seconds'],rebuild_seconds=rebuild['elapsed_seconds'])
    extrap=dict(basis='20-video smoke extrapolation; not a time guarantee or confidence interval',train_videos=7193,
        wall_minutes=perf['initial_wall_seconds']/20*7193/60,
        p50_scaled_minutes=perf['p50_video_seconds']*7193/60,p90_scaled_minutes=perf['p90_video_seconds']*7193/60,
        per_video_gib=perf['mean_video_bytes']*7193/1024**3,run_gib=perf['run_bytes']/20*7193/1024**3,
        recommended_free_gib=2)
    protected=runs['initial']['protected_files']
    original=subprocess.check_output(['git','show','HEAD:docs/progress_log.md'],cwd=s.ROOT)
    s.need((s.ROOT/'docs/progress_log.md').read_bytes().startswith(original),'Progress history modified')
    report=dict(status='SCANNER_IMPLEMENTATION_COMPLETE',full_train_executed=False,
        initial_commit=s.read_json(out/'run.json')['commit'],config_hash=ctx['config_hash'],scanner=ctx['scanner'],
        frozen_evidence=ctx['config']['frozen_evidence'],config=ctx['config'],smoke_manifest_hash=s.digest(s.ROOT/ctx['config']['smoke_manifest']),
        aggregate=s.aggregate(out,ctx),videos=videos,performance=perf,full_train_extrapolation=extrap,
        checks=tests,verification_runs=runs,protected_files=protected,
        evidence_dir=str(evidence.relative_to(s.ROOT)),output_dir=str(out.relative_to(s.ROOT)),
        published_utc=s.utc())
    columns=['video_id','label','source_group','fps','processed_frame_count','candidate_frames','candidate_ratio','processing_seconds','output_bytes']
    s.atomic_text(evidence/'per_video_summary.csv',s.csv_text([{k:v[k] for k in columns} for v in videos]))
    s.atomic_json(REPORT,report)
    s.atomic_text(MARKDOWN,markdown(report))
    return report


def check(evidence):
    report=s.read_json(REPORT)
    s.need(MARKDOWN.read_text()==markdown(report),'JSON/Markdown mismatch')
    rows=s.p.read_csv(evidence/'per_video_summary.csv')
    s.need(len(rows)==len(report['videos'])==20,'CSV count mismatch')
    for row,video in zip(rows,report['videos']):
        s.need(all(row[key]==str(video[key]) for key in row),'CSV/JSON mismatch')
    s.need(s.digest(s.ROOT/report['config']['smoke_manifest'])==report['smoke_manifest_hash'],'Smoke manifest changed')
    s.need(s.code_identity()==report['scanner'] and s.digest(s.DEFAULT_CONFIG)==report['config_hash'],'Publication context stale')
    for name,verification in report['verification_runs'].items():
        directory=s.ARTIFACTS/('blur_train_v3_smoke'+{'initial':'','repeat':'_repeat','recovery':'_recovery'}[name])
        for rel,sha in verification['artifact_hashes'].items():
            s.need(s.digest(directory/rel)==sha,'Published artifact hash changed')
    print('PASS: JSON / Markdown / CSV / smoke selection / output hashes')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,default=s.ARTIFACTS/'blur_train_v3_smoke')
    ap.add_argument('--evidence-dir',type=Path,default=EVIDENCE)
    ap.add_argument('--check-only',action='store_true')
    args=ap.parse_args()
    if not args.check_only:publish(args.output_dir,args.evidence_dir)
    check(args.evidence_dir)


if __name__=='__main__':
    main()
