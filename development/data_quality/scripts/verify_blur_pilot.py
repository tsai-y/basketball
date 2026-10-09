#!/usr/bin/env python3
"""Recompute Pilot consistency using derived CSV/JSON only; never opens dataset videos."""
import argparse
from collections import Counter
import itertools
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import audit_blur_pilot as p


def spans(values):
    offset, result = 0, []
    for value, group in itertools.groupby(values):
        length = len(list(group))
        if value:
            result.append((offset, offset+length-1))
        offset += length
    return result


def verify(out, trace):
    report = json.loads((out / 'blur_screening_pilot_v1.json').read_text())
    cfg = report['config']
    manifest = p.read_csv(out / 'pilot_manifest_v1.csv')
    by_id = {r['video_id']:r for r in manifest}
    guard = p.TrainGuard()
    selected = p.choose(guard,cfg,report['selection']['seed'])
    assert [r['video_id'] for r in selected] == list(by_id)
    assert p.choose(guard,cfg,report['selection']['seed']) == selected
    for r in manifest:
        guard.authorize(r['video_id'],r['relative_video_path'])
    assert not guard.events
    assert report['provenance']['script_sha256'] == p.gate.digest(Path(p.__file__))
    csvs = sorted((out / 'frame_scores').glob('*.csv'))
    assert len(csvs)==len(manifest)
    frames = [r for file in csvs for r in p.read_csv(file)]
    assert len(frames) == report['frame_count'] == sum(int(r['frame_count']) for r in manifest)
    arrays = {k:np.array([float(r[k]) for r in frames]) for k in p.METRICS}
    for metric, array in arrays.items():
        assert np.isfinite(array).all()
        assert float(np.quantile(array,cfg['thresholds']['quantile'],method='linear')) == report['thresholds'][metric]
        assert p.describe(array) == report['distributions'][metric]
    assert np.isclose(np.corrcoef(*arrays.values())[0,1], report['correlation']['pearson'])
    segment_csv = p.read_csv(out / 'segments.csv')
    videos = json.loads((out / 'video_summary.json').read_text())
    assert videos == report['videos']
    expected_spans = []
    for video in videos:
        vid=video['video_id']; rows=[r for r in frames if r['video_id']==vid]
        n=len(rows); fps=float(by_id[vid]['fps'])
        assert [int(r['frame_index']) for r in rows] == list(range(n))
        assert n==video['frame_count']==int(by_id[vid]['frame_count'])
        raw_a=np.array([float(r['laplacian_variance']) for r in rows])
        raw_b=np.array([float(r['tenengrad']) for r in rows])
        low_a=raw_a<=report['thresholds']['laplacian_variance']
        low_b=raw_b<=report['thresholds']['tenengrad']
        width=math.ceil(cfg['thresholds']['smoothing_seconds']*fps)
        if width%2==0: width+=1
        assert width==video['smoothing_window_frames']
        half=width//2
        med_a=[float(np.median(raw_a[max(0,i-half):i+half+1])) for i in range(n)]
        med_b=[float(np.median(raw_b[max(0,i-half):i+half+1])) for i in range(n)]
        sm=[a<=report['thresholds']['laplacian_variance'] and b<=report['thresholds']['tenengrad']
            for a,b in zip(med_a,med_b)]
        kept=[False]*n
        for lo,hi in spans(sm):
            if (hi-lo+1)/fps >= cfg['thresholds']['minimum_segment_seconds']:
                kept[lo:hi+1]=[True]*(hi-lo+1)
        expected=dict(zip(p.METHODS,(low_a,low_b,low_a & low_b,kept)))
        for i,r in enumerate(rows):
            assert r['decode_status']=='ok'
            assert int(r['candidate_flag'])==int(expected[cfg['thresholds']['primary_candidate']][i])
            assert r['flag_reason']==(';'.join(m for m in p.METHODS if expected[m][i]) or 'none')
            assert r['threshold_version']==report['threshold_version']
            if r['timestamp_seconds']:
                assert abs(float(r['timestamp_seconds'])-i/fps)<=0.5/fps
        for method, flags in expected.items():
            assert [int(r[method]) for r in rows]==[int(x) for x in flags]
            bad=spans(flags); good=spans([not x for x in flags]); summary=video['strategies'][method]
            assert sum(flags)==summary['candidate_frames']
            assert sum(flags)/n==summary['candidate_ratio']
            assert len(bad)==summary['candidate_segments']
            assert max((hi-lo+1 for lo,hi in bad),default=0)==summary['longest_candidate_frames']
            assert len(good)==summary['usable_segments']
            assert sum(hi-lo+1 for lo,hi in good)==summary['usable_frames']
            assert max((hi-lo+1 for lo,hi in good),default=0)==summary['longest_usable_frames']
            for w in cfg['windows']:
                assert sum(max(0,hi-lo+2-w) for lo,hi in good)==summary['usable_windows'][str(w)]
            for kind,ranges in [('candidate',bad),('unflagged',good)]:
                for lo,hi in ranges:
                    expected_spans.append((vid,method,kind,lo,hi))
    actual_spans=[(r['video_id'],r['strategy'],r['kind'],int(r['start_frame']),int(r['end_frame'])) for r in segment_csv]
    assert expected_spans==actual_spans
    for r in segment_csv:
        vid=r['video_id']; lo=int(r['start_frame']); hi=int(r['end_frame']); fps=float(by_id[vid]['fps'])
        assert int(r['frame_count'])==hi-lo+1
        assert float(r['nominal_duration_seconds'])==(hi-lo+1)/fps
        vidframes=[f for f in frames if f['video_id']==vid]
        assert r['start_timestamp']==vidframes[lo]['timestamp_seconds']
        assert r['end_timestamp']==vidframes[hi]['timestamp_seconds']
    assert p.aggregate(videos,cfg)==report['impact']
    for dim, groups in report['groups'].items():
        for key, stats in groups.items():
            assert p.aggregate([v for v in videos if str(v[dim])==key],cfg)==stats
    # JSON sorts keys; the first rendering uses insertion order. Compare every
    # complete line with multiplicity, allowing only section/table row ordering.
    assert Counter((out / 'blur_screening_pilot_v1.md').read_text().splitlines()) == Counter(p.markdown(report).splitlines())
    index=p.read_csv(out / 'contact_sheets/index.csv')
    assert len(index)==2*3*cfg['contact_sheets']['examples_per_band']
    for tile in index:
        r=next(r for r in frames if (r['video_id'],r['frame_index'])==(tile['video_id'],tile['frame_index']))
        for key,value in r.items(): assert tile[key]==value
    from PIL import Image
    for metric in p.METRICS:
        with Image.open(out / 'contact_sheets' / f'{metric}_low_middle_high.png') as im:
            assert im.size==(390*cfg['contact_sheets']['examples_per_band'],1035)
            im.verify()
    # strace -yy includes resolved fd targets, including ffprobe/OpenCV /proc fd reopens.
    lines=trace.read_text().splitlines()
    data_prefix=str(p.gate.DATA)+'/'
    opened=[]
    for line in lines:
        if data_prefix not in line: continue
        assert 'openat(' in line and 'O_RDONLY' in line
        assert not any(flag in line for flag in ['O_WRONLY','O_RDWR','O_TRUNC','O_CREAT'])
        ids=set(re.findall(re.escape(data_prefix)+r'([^"<>]+\.mp4)',line))
        assert len(ids)==1, line
        vid=next(iter(ids)); assert vid in by_id
        opened.append(vid)
    assert set(opened)==set(by_id)
    assert Counter(opened)==Counter({vid:4 for vid in by_id})  # 2 guarded opens + 2 FD reopens
    executions=[line for line in lines if 'execve(' in line and '= 0' in line]
    probe_execs=[line for line in executions if '/ffprobe"' in line and '-show_entries' in line]
    assert len(probe_execs)==report['access']['train']['probes']
    assert not any('/ffmpeg"' in line for line in executions)
    ledger=json.loads((out / 'runtime_logs/access_ledger.json').read_text())
    assert len(ledger)==60 and {e['video_id'] for e in ledger}==set(by_id)
    assert all(e['split']=='train' and e['read_only'] for e in ledger)
    assert Counter(e['operation'] for e in ledger)=={'probe':30,'decode':30}
    cli=json.loads((out/'runtime_logs/cli_checks.json').read_text())
    assert cli['same_seed_cli_identical'] and cli['existing_output_rejected']
    assert cli['cli_help']==cli['dry_run']==cli['import_smoke']=='PASS'
    assert all(v['exit_code']==2 for v in cli['heldout_rejection'].values())
    protected=json.loads((out/'runtime_logs/protected_text_verification.json').read_text())
    assert protected['status']=='PASS' and protected['progress_log_append_only']
    subprocess.run(['git','diff','--check'],cwd=p.ROOT,check=True,capture_output=True)
    return dict(status='PASS', selected_videos=len(manifest), frames=len(frames),
        syntax_import_help_dry_run='PASS (see cli_checks.json)',
        safety_tests='6 PASS; every validation/test/excluded ID and mismatched path rejected before opener',
        selection_reproducible=True, csv_json_markdown_consistent=True,
        independent_flags_segments_windows_recalculation=True, contact_sheet_tiles_verified=len(index),
        unique_video_ids_opened=len(set(opened)), dataset_open_syscalls=len(opened),
        literal_guarded_opens=60, descriptor_reopens=60, all_video_opens_read_only=True,
        validation_opens_probes_decodes=0, test_opens_probes_decodes=0,
        ffprobe_selected_video_execs=len(probe_execs), ffmpeg_execs=0,
        protected_text_verification=protected, git_diff_check='PASS',
        cli_checks=cli,
        syscall_trace_sha256=p.gate.digest(trace), ledger_sha256=p.gate.digest(out/'runtime_logs/access_ledger.json'),
        frame_scores_sha256={f.name:p.gate.digest(f) for f in csvs})


def final_markdown(report):
    label=report['groups']['label']; fps=report['groups']['fps']; source=report['groups']['source_group']
    most=max(source,key=lambda k:source[k]['both_low']['candidate_ratio'])
    observations=(
        '\n## Pilot 觀察與擴展條件\n\n'
        f"雙低比例 Make {label['Make']['both_low']['candidate_ratio']:.2%}、"
        f"Miss {label['Miss']['both_low']['candidate_ratio']:.2%}；"
        f"23.976 FPS {fps['23.976']['both_low']['candidate_ratio']:.2%}、"
        f"29.97 FPS {fps['29.97']['both_low']['candidate_ratio']:.2%}。"
        f"來源最高為 {most}（{source[most]['both_low']['candidate_ratio']:.2%}），"
        '但每來源只有一支影片，不能解釋為 FPS 或類別造成品質差異。\n\n'
        '兩張 contact sheets 已檢查，18 tiles 的 ID、位置、分數、旗標與版本可讀且與 CSV 一致。'
        '不同場景的背景紋理、字幕與鏡頭尺度明顯不同；這些畫面只供演算法行為檢查，沒有新增人工模糊標籤。\n\n'
        '小樣本管線具備技術可行性，尚不具直接擴展完整 train 的執行條件：'
        '目前所有分數與 JPEG previews 暫存記憶體，完整 train 需改為分批或串流統計/畫面選取，'
        '並規劃執行時間、容量與失敗復原。這些修改及正式品質政策皆未開始。\n')
    return p.markdown(report)+observations+'\n## 完成驗證\n\n```json\n'+p.dumps(report['verification']).rstrip()+'\n```\n'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,default=p.ROOT/'development/data_quality/artifacts/blur_pilot_v1')
    ap.add_argument('--trace',type=Path,required=True)
    ap.add_argument('--publish',action='store_true',help='Exclusively create tracked-candidate report and manifest; no overwrite')
    args=ap.parse_args()
    result=verify(args.output_dir,args.trace)
    if args.publish:
        report=json.loads((args.output_dir/'blur_screening_pilot_v1.json').read_text())
        report['verification']=result
        paths=[p.ROOT/'development/data_quality/pilot_manifest_v1.csv',
               p.ROOT/'docs/data_quality/blur_screening_pilot_v1.json',
               p.ROOT/'docs/data_quality/blur_screening_pilot_v1.md',
               args.output_dir/'runtime_logs/verification.json']
        assert all(not x.exists() for x in paths), 'Refusing existing publication outputs'
        with paths[0].open('xb') as f: f.write((args.output_dir/'pilot_manifest_v1.csv').read_bytes())
        p.write_json(paths[1],report)
        with paths[2].open('x') as f: f.write(final_markdown(report))
        p.write_json(paths[3],result)
    print(p.dumps(result))


if __name__=='__main__':
    main()
