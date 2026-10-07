# Experiment D Gate 1A — Temporal Feasibility Audit

Status: **COMPLETE**. UTC: 2026-10-06T19:04:15.588069+00:00.

本報告只描述 train／validation 的時間 metadata；不選擇 Experiment D protocol，也不是 pre-release 研究。

## 範圍、方法與可重現性

- 既有 dataset_summary.json 僅含全資料集彙總，沒有可重用的逐影片、分 split metadata；未用其全體分布制定本報告。
- 僅由 frozen manifest allowlist 開啟 train／validation；不遍歷資料目錄、不 probe test。拒絕 test row/path、symlink 與 hardlink。
- ffprobe 讀取 metadata，未使用 -count_frames 或完整解碼；raw frame count 為 nb_frames，不是本輪逐格解碼計數。
- duration 優先使用 video stream duration，缺失時才用 format duration；兩者原值均保留。FPS 主統計用 avg_frame_rate，同時保存 r_frame_rate。
- 百分位採線性插值：h=(n−1)p，介於兩個排序值時線性插值。比例分母為該組有效 metadata 數；失敗另列，不當作零。
- 25 FPS 等效值 E=duration×25，用 Fraction 精確計算並保存未取整值。主 short 判定用 E<T；另列 floor、round-half-up=floor(E+0.5)、ceil 敏感度。
- Looping 僅為反事實分析：E<64 時 A=floor(E)，repeat_factor=ceil(64/A)（包含首輪），重複位置=64−A，duplicate exposure ratio=(64−A)/64；A=0 則未定義，不強制補成1。
- 16/32/64/96/128 與 0.64/1.28/2.56/3.84/5.12 秒只是描述閾值，不代表選定長度或25 FPS。
- Metadata 明顯不一致定義：|nb_frames/avg_fps − duration| > max(0.1 秒, 1/fps)。小差值仍完整統計；沒有修改或排除資料。
- 逐影片 probe、未取整值、取整值與存取 ledger 存在 /tmp 的 cache（不進 Git）；JSON 報告只保留彙總及異常。

```json
{
  "started_utc": "2026-10-06T19:04:15.588069+00:00",
  "elapsed_seconds": 60.512,
  "python": "3.13.13",
  "ffprobe": "ffprobe version 8.0.1-3ubuntu2 Copyright (c) 2007-2025 the FFmpeg developers",
  "command": [
    "python3",
    "-B",
    "scripts/audit_i3d_temporal_feasibility.py",
    "--mode",
    "audit"
  ],
  "effective_arguments": {
    "mode": "audit",
    "splits": [
      "train",
      "validation"
    ],
    "workers": 8
  },
  "script_sha256": "7cdbdac59633b7f69c21aa772a2dc2e5c05c0f80c0f07f6eed89fa26335fe998",
  "ffprobe_template": [
    "ffprobe",
    "-v",
    "error",
    "-protocol_whitelist",
    "file,pipe",
    "-show_entries",
    "stream=codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,duration,time_base,duration_ts:format=format_name,duration",
    "-of",
    "json",
    "<authorized_fd>"
  ],
  "cache_basename": "basketball51_gate1a_metadata.json",
  "metadata_source": "fresh_ffprobe_no_full_decode",
  "cache_sha256": "669915d0e41c33562f29860016b3e5068ca002d2a36c4246829eb50ad8f814ee",
  "completion_verification": {
    "system_call_trace_sha256": "7d7e33a670ea4262577191a96f1c7f8086bf7b2b518041a074b9b841cf472cd4",
    "trace_file": "basketball_gate1a_syscalls.log (local /tmp only)",
    "method": "strace -f -qq -e trace=openat,execve; compare all literal MP4 opens with frozen manifest allowlist",
    "dataset_open_syscalls": 8740,
    "unique_allowed_videos": 8740,
    "split_open_counts": {
      "train": 7193,
      "validation": 1547
    },
    "test_video_opens": 0,
    "test_video_probes": 0,
    "test_video_decodes": 0,
    "ffmpeg_exec_count": 0,
    "replayed_statistics_identical": true,
    "independent_raw_probe_recalculation": "PASS",
    "numerators_denominators_and_group_totals": "PASS",
    "help_check_and_test_split_rejection": "PASS",
    "injected_test_row_path_rejected_before_open": "PASS",
    "synthetic_looping_and_exact_boundary_checks": "PASS",
    "protected_files_sha256_unchanged": 70,
    "protected_files_scope": "all files under runs/ and configs/, main manifest/split, research_baseline_v1.3.md",
    "output_absolute_data_paths_and_forbidden_ids": "NONE"
  }
}
```

## 完整性與零 test 存取

| Split | 預期 | 列數 | 找到 | Probe | Probe失敗 | 無效metadata |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 7193 | 7193 | 7193 | 7193 | 0 | 0 |
| validation | 1547 | 1547 | 1547 | 1547 | 0 | 0 |

```json
{
  "scope": {
    "input_hashes": {
      "manifest": "e7bba1acd6028cf67cdf28251e69852e82cff201e13519ed05970ac483cce6b6",
      "split_summary": "953da31313976763c8f5d597da5f0fe5a6a45e10f5d5edcd42fb0c120af2f7c5"
    },
    "manifest_counts": {
      "excluded": 32,
      "test": 1539,
      "train": 7193,
      "validation": 1547
    },
    "duplicate_rows": 0,
    "duplicate_ids": 0,
    "duplicate_paths": 0
  },
  "access": {
    "actual_splits": [
      "train",
      "validation"
    ],
    "video_open_count": 8740,
    "probe_count": 8740,
    "decode_count": 0,
    "test_video_open_count": 0,
    "test_probe_count": 0,
    "test_decode_count": 0,
    "selected_id_sha256": "f39fbb7a9e048dbfa2ca3b3dba99ca11d73fb985f014c012ffddc0ef62b2b0b9",
    "ledger_sha256": "9df6abc74d258a5f832c8543dcfe72045f818e1b9cb1c25506e17ce30113e71f"
  },
  "issue_counts": {},
  "missing_paths": [],
  "probe_errors": []
}
```

## train（n=7193，有效=7193）

### FPS / 格式分布

```json
{
  "fps_rational": {
    "2997/100": 1623,
    "2997/125": 5570
  },
  "r_frame_rate": {
    "2997/100": 1623,
    "2997/125": 5570
  },
  "fps_float": {
    "23.976": 5570,
    "29.97": 1623
  },
  "fps_groups": {
    "23.976": 5570,
    "29.97": 1623
  },
  "fps_other_video_ids": [],
  "codec": {
    "h264": 7193
  },
  "resolution": {
    "320x240": 7193
  },
  "container": {
    "mov,mp4,m4a,3gp,3g2,mj2": 7193
  },
  "duration_source": {
    "video_stream": 7193
  }
}
```

### 長度統計

| 量 | Min | Mean | Median | P5 | P10 | P25 | P75 | P90 | P95 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| frame_count | 145 | 153.641874044 | 146 | 146 | 146 | 146 | 146 | 180 | 180 | 180 |
| duration_seconds | 6.006006 | 6.069360228 | 6.089423 | 6.006006 | 6.006006 | 6.047714 | 6.089423 | 6.089423 | 6.089423 | 6.089423 |
| equivalent_frames_25fps | 150.15015 | 151.734005707 | 152.235575 | 150.15015 | 150.15015 | 151.19285 | 152.235575 | 152.235575 | 152.235575 | 152.235575 |
| metadata_signed_delta_seconds | -0.000000244 | -0.000000169 | -0.000000244 | -0.000000244 | -0.000000244 | -0.000000244 | 0.000000006 | 0.000000006 | 0.000000006 | 0.000000381 |
| metadata_absolute_delta_seconds | 0.000000006 | 0.000000194 | 0.000000244 | 0.000000006 | 0.000000006 | 0.000000244 | 0.000000244 | 0.000000244 | 0.000000244 | 0.000000381 |
| container_minus_stream_seconds | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### 閾值（嚴格少於；count/denominator, %）

| Frames閾值 | Raw | 秒閾值 | Duration | E未取整 | Floor | Half-up | Ceil |
| ---: | --- | ---: | --- | --- | --- | --- | --- |
| 16 | 0/7193 (0.000000%) | .64 | 0/7193 (0.000000%) | 0/7193 (0.000000%) | 0/7193 (0.000000%) | 0/7193 (0.000000%) | 0/7193 (0.000000%) |
| 32 | 0/7193 (0.000000%) | 1.28 | 0/7193 (0.000000%) | 0/7193 (0.000000%) | 0/7193 (0.000000%) | 0/7193 (0.000000%) | 0/7193 (0.000000%) |
| 64 | 0/7193 (0.000000%) | 2.56 | 0/7193 (0.000000%) | 0/7193 (0.000000%) | 0/7193 (0.000000%) | 0/7193 (0.000000%) | 0/7193 (0.000000%) |
| 96 | 0/7193 (0.000000%) | 3.84 | 0/7193 (0.000000%) | 0/7193 (0.000000%) | 0/7193 (0.000000%) | 0/7193 (0.000000%) | 0/7193 (0.000000%) |
| 128 | 0/7193 (0.000000%) | 5.12 | 0/7193 (0.000000%) | 0/7193 (0.000000%) | 0/7193 (0.000000%) | 0/7193 (0.000000%) | 0/7193 (0.000000%) |

### Looping exposure（只對 E<64 的子集）

```json
{
  "short_n": 0,
  "defined_n": 0,
  "undefined_n": 0,
  "available_frames": {
    "n": 0,
    "min": null,
    "mean": null,
    "median": null,
    "p05": null,
    "p10": null,
    "p25": null,
    "p75": null,
    "p90": null,
    "p95": null,
    "max": null
  },
  "repeat_factor": {
    "n": 0,
    "min": null,
    "mean": null,
    "median": null,
    "p05": null,
    "p10": null,
    "p25": null,
    "p75": null,
    "p90": null,
    "p95": null,
    "max": null
  },
  "repeat_factor_counts": {},
  "duplicate_positions": {
    "n": 0,
    "min": null,
    "mean": null,
    "median": null,
    "p05": null,
    "p10": null,
    "p25": null,
    "p75": null,
    "p90": null,
    "p95": null,
    "max": null
  },
  "duplicate_exposure_ratio": {
    "n": 0,
    "min": null,
    "mean": null,
    "median": null,
    "p05": null,
    "p10": null,
    "p25": null,
    "p75": null,
    "p90": null,
    "p95": null,
    "max": null
  },
  "cycles_2": 0,
  "cycles_3": 0,
  "cycles_4_or_more": 0,
  "worst_cases": []
}
```

## validation（n=1547，有效=1547）

### FPS / 格式分布

```json
{
  "fps_rational": {
    "2997/100": 199,
    "2997/125": 1348
  },
  "r_frame_rate": {
    "2997/100": 199,
    "2997/125": 1348
  },
  "fps_float": {
    "23.976": 1348,
    "29.97": 199
  },
  "fps_groups": {
    "23.976": 1348,
    "29.97": 199
  },
  "fps_other_video_ids": [],
  "codec": {
    "h264": 1547
  },
  "resolution": {
    "320x240": 1547
  },
  "container": {
    "mov,mp4,m4a,3gp,3g2,mj2": 1547
  },
  "duration_source": {
    "video_stream": 1547
  }
}
```

### 長度統計

| 量 | Min | Mean | Median | P5 | P10 | P25 | P75 | P90 | P95 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| frame_count | 146 | 150.373626374 | 146 | 146 | 146 | 146 | 146 | 180 | 180 | 180 |
| duration_seconds | 6.006006 | 6.078692565 | 6.089423 | 6.006006 | 6.006006 | 6.089423 | 6.089423 | 6.089423 | 6.089423 | 6.089423 |
| equivalent_frames_25fps | 150.15015 | 151.967314124 | 152.235575 | 150.15015 | 150.15015 | 152.235575 | 152.235575 | 152.235575 | 152.235575 | 152.235575 |
| metadata_signed_delta_seconds | -0.000000244 | -0.000000212 | -0.000000244 | -0.000000244 | -0.000000244 | -0.000000244 | -0.000000244 | 0.000000006 | 0.000000006 | 0.000000006 |
| metadata_absolute_delta_seconds | 0.000000006 | 0.000000213 | 0.000000244 | 0.000000006 | 0.000000006 | 0.000000244 | 0.000000244 | 0.000000244 | 0.000000244 | 0.000000244 |
| container_minus_stream_seconds | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### 閾值（嚴格少於；count/denominator, %）

| Frames閾值 | Raw | 秒閾值 | Duration | E未取整 | Floor | Half-up | Ceil |
| ---: | --- | ---: | --- | --- | --- | --- | --- |
| 16 | 0/1547 (0.000000%) | .64 | 0/1547 (0.000000%) | 0/1547 (0.000000%) | 0/1547 (0.000000%) | 0/1547 (0.000000%) | 0/1547 (0.000000%) |
| 32 | 0/1547 (0.000000%) | 1.28 | 0/1547 (0.000000%) | 0/1547 (0.000000%) | 0/1547 (0.000000%) | 0/1547 (0.000000%) | 0/1547 (0.000000%) |
| 64 | 0/1547 (0.000000%) | 2.56 | 0/1547 (0.000000%) | 0/1547 (0.000000%) | 0/1547 (0.000000%) | 0/1547 (0.000000%) | 0/1547 (0.000000%) |
| 96 | 0/1547 (0.000000%) | 3.84 | 0/1547 (0.000000%) | 0/1547 (0.000000%) | 0/1547 (0.000000%) | 0/1547 (0.000000%) | 0/1547 (0.000000%) |
| 128 | 0/1547 (0.000000%) | 5.12 | 0/1547 (0.000000%) | 0/1547 (0.000000%) | 0/1547 (0.000000%) | 0/1547 (0.000000%) | 0/1547 (0.000000%) |

### Looping exposure（只對 E<64 的子集）

```json
{
  "short_n": 0,
  "defined_n": 0,
  "undefined_n": 0,
  "available_frames": {
    "n": 0,
    "min": null,
    "mean": null,
    "median": null,
    "p05": null,
    "p10": null,
    "p25": null,
    "p75": null,
    "p90": null,
    "p95": null,
    "max": null
  },
  "repeat_factor": {
    "n": 0,
    "min": null,
    "mean": null,
    "median": null,
    "p05": null,
    "p10": null,
    "p25": null,
    "p75": null,
    "p90": null,
    "p95": null,
    "max": null
  },
  "repeat_factor_counts": {},
  "duplicate_positions": {
    "n": 0,
    "min": null,
    "mean": null,
    "median": null,
    "p05": null,
    "p10": null,
    "p25": null,
    "p75": null,
    "p90": null,
    "p95": null,
    "max": null
  },
  "duplicate_exposure_ratio": {
    "n": 0,
    "min": null,
    "mean": null,
    "median": null,
    "p05": null,
    "p10": null,
    "p25": null,
    "p75": null,
    "p90": null,
    "p95": null,
    "max": null
  },
  "cycles_2": 0,
  "cycles_3": 0,
  "cycles_4_or_more": 0,
  "worst_cases": []
}
```

## combined（n=8740，有效=8740）

### FPS / 格式分布

```json
{
  "fps_rational": {
    "2997/100": 1822,
    "2997/125": 6918
  },
  "r_frame_rate": {
    "2997/100": 1822,
    "2997/125": 6918
  },
  "fps_float": {
    "23.976": 6918,
    "29.97": 1822
  },
  "fps_groups": {
    "23.976": 6918,
    "29.97": 1822
  },
  "fps_other_video_ids": [],
  "codec": {
    "h264": 8740
  },
  "resolution": {
    "320x240": 8740
  },
  "container": {
    "mov,mp4,m4a,3gp,3g2,mj2": 8740
  },
  "duration_source": {
    "video_stream": 8740
  }
}
```

### 長度統計

| 量 | Min | Mean | Median | P5 | P10 | P25 | P75 | P90 | P95 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| frame_count | 145 | 153.063386728 | 146 | 146 | 146 | 146 | 146 | 180 | 180 | 180 |
| duration_seconds | 6.006006 | 6.071012073 | 6.089423 | 6.006006 | 6.006006 | 6.089423 | 6.089423 | 6.089423 | 6.089423 | 6.089423 |
| equivalent_frames_25fps | 150.15015 | 151.775301831 | 152.235575 | 150.15015 | 150.15015 | 152.235575 | 152.235575 | 152.235575 | 152.235575 | 152.235575 |
| metadata_signed_delta_seconds | -0.000000244 | -0.000000177 | -0.000000244 | -0.000000244 | -0.000000244 | -0.000000244 | -0.000000244 | 0.000000006 | 0.000000006 | 0.000000381 |
| metadata_absolute_delta_seconds | 0.000000006 | 0.000000198 | 0.000000244 | 0.000000006 | 0.000000006 | 0.000000244 | 0.000000244 | 0.000000244 | 0.000000244 | 0.000000381 |
| container_minus_stream_seconds | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### 閾值（嚴格少於；count/denominator, %）

| Frames閾值 | Raw | 秒閾值 | Duration | E未取整 | Floor | Half-up | Ceil |
| ---: | --- | ---: | --- | --- | --- | --- | --- |
| 16 | 0/8740 (0.000000%) | .64 | 0/8740 (0.000000%) | 0/8740 (0.000000%) | 0/8740 (0.000000%) | 0/8740 (0.000000%) | 0/8740 (0.000000%) |
| 32 | 0/8740 (0.000000%) | 1.28 | 0/8740 (0.000000%) | 0/8740 (0.000000%) | 0/8740 (0.000000%) | 0/8740 (0.000000%) | 0/8740 (0.000000%) |
| 64 | 0/8740 (0.000000%) | 2.56 | 0/8740 (0.000000%) | 0/8740 (0.000000%) | 0/8740 (0.000000%) | 0/8740 (0.000000%) | 0/8740 (0.000000%) |
| 96 | 0/8740 (0.000000%) | 3.84 | 0/8740 (0.000000%) | 0/8740 (0.000000%) | 0/8740 (0.000000%) | 0/8740 (0.000000%) | 0/8740 (0.000000%) |
| 128 | 0/8740 (0.000000%) | 5.12 | 0/8740 (0.000000%) | 0/8740 (0.000000%) | 0/8740 (0.000000%) | 0/8740 (0.000000%) | 0/8740 (0.000000%) |

### Looping exposure（只對 E<64 的子集）

```json
{
  "short_n": 0,
  "defined_n": 0,
  "undefined_n": 0,
  "available_frames": {
    "n": 0,
    "min": null,
    "mean": null,
    "median": null,
    "p05": null,
    "p10": null,
    "p25": null,
    "p75": null,
    "p90": null,
    "p95": null,
    "max": null
  },
  "repeat_factor": {
    "n": 0,
    "min": null,
    "mean": null,
    "median": null,
    "p05": null,
    "p10": null,
    "p25": null,
    "p75": null,
    "p90": null,
    "p95": null,
    "max": null
  },
  "repeat_factor_counts": {},
  "duplicate_positions": {
    "n": 0,
    "min": null,
    "mean": null,
    "median": null,
    "p05": null,
    "p10": null,
    "p25": null,
    "p75": null,
    "p90": null,
    "p95": null,
    "max": null
  },
  "duplicate_exposure_ratio": {
    "n": 0,
    "min": null,
    "mean": null,
    "median": null,
    "p05": null,
    "p10": null,
    "p25": null,
    "p75": null,
    "p90": null,
    "p95": null,
    "max": null
  },
  "cycles_2": 0,
  "cycles_3": 0,
  "cycles_4_or_more": 0,
  "worst_cases": []
}
```

## 分組比較

只描述分布；完整分位數、閾值與 FPS 分布見 JSON groups。

| Scope | 維度 | Group | n | Raw min/mean/median/max | Duration min/mean/median/max (s) | Raw<64 | E<64 |
| --- | --- | --- | ---: | --- | --- | --- | --- |
| train | outcome | Make | 3772 | 145/153.744432662/146/180 | 6.006006/6.069298525/6.089423/6.089423 | 0/3772 (0.000000%) | 0/3772 (0.000000%) |
| train | outcome | Miss | 3421 | 145/153.528792751/146/180 | 6.006006/6.069428262/6.089423/6.089423 | 0/3421 (0.000000%) | 0/3421 (0.000000%) |
| train | shot_type | 2P | 2419 | 145/153.457213725/146/180 | 6.006006/6.069611876/6.089423/6.089423 | 0/2419 (0.000000%) | 0/2419 (0.000000%) |
| train | shot_type | 3P | 2258 | 145/153.750664305/146/180 | 6.006006/6.068901219/6.089423/6.089423 | 0/2258 (0.000000%) | 0/2258 (0.000000%) |
| train | shot_type | FT | 1575 | 145/153.808888889/146/180 | 6.006006/6.069058646/6.089423/6.089423 | 0/1575 (0.000000%) | 0/1575 (0.000000%) |
| train | shot_type | MP | 941 | 145/153.575982997/146/180 | 6.006006/6.070319526/6.089423/6.089423 | 0/941 (0.000000%) | 0/941 (0.000000%) |
| train | original_class | 2p0 | 1033 | 145/153.094869313/146/180 | 6.006006/6.069921325/6.089423/6.089423 | 0/1033 (0.000000%) | 0/1033 (0.000000%) |
| train | original_class | 2p1 | 1386 | 145/153.727272727/146/180 | 6.006006/6.069381241/6.089423/6.089423 | 0/1386 (0.000000%) | 0/1386 (0.000000%) |
| train | original_class | 3p0 | 1440 | 145/153.944444444/146/180 | 6.006006/6.068337018/6.089423/6.089423 | 0/1440 (0.000000%) | 0/1440 (0.000000%) |
| train | original_class | 3p1 | 818 | 145/153.409535452/146/180 | 6.006006/6.069894433/6.089423/6.089423 | 0/818 (0.000000%) | 0/818 (0.000000%) |
| train | original_class | ft0 | 400 | 145/153.2/146/180 | 6.006006/6.070654163/6.089423/6.089423 | 0/400 (0.000000%) | 0/400 (0.000000%) |
| train | original_class | ft1 | 1175 | 145/154.016170213/146/180 | 6.006006/6.068515491/6.089423/6.089423 | 0/1175 (0.000000%) | 0/1175 (0.000000%) |
| train | original_class | mp0 | 548 | 145/153.494525547/146/180 | 6.006006/6.070471504/6.089423/6.089423 | 0/548 (0.000000%) | 0/548 (0.000000%) |
| train | original_class | mp1 | 393 | 145/153.68956743/146/180 | 6.006006/6.070107608/6.089423/6.089423 | 0/393 (0.000000%) | 0/393 (0.000000%) |
| validation | outcome | Make | 807 | 146/150.255266419/146/180 | 6.006006/6.078982954/6.089423/6.089423 | 0/807 (0.000000%) | 0/807 (0.000000%) |
| validation | outcome | Miss | 740 | 146/150.502702703/146/180 | 6.006006/6.078375884/6.089423/6.089423 | 0/740 (0.000000%) | 0/740 (0.000000%) |
| validation | shot_type | 2P | 437 | 146/150.823798627/146/180 | 6.006006/6.077588094/6.089423/6.089423 | 0/437 (0.000000%) | 0/437 (0.000000%) |
| validation | shot_type | 3P | 529 | 146/149.72778828/146/180 | 6.006006/6.080277091/6.089423/6.089423 | 0/529 (0.000000%) | 0/529 (0.000000%) |
| validation | shot_type | FT | 367 | 146/151.465940054/146/180 | 6.006006/6.076012638/6.089423/6.089423 | 0/367 (0.000000%) | 0/367 (0.000000%) |
| validation | shot_type | MP | 214 | 146/149.177570093/146/180 | 6.006006/6.081627019/6.089423/6.089423 | 0/214 (0.000000%) | 0/214 (0.000000%) |
| validation | original_class | 2p0 | 171 | 146/151.964912281/146/180 | 6.006006/6.074788439/6.089423/6.089423 | 0/171 (0.000000%) | 0/171 (0.000000%) |
| validation | original_class | 2p1 | 266 | 146/150.090225564/146/180 | 6.006006/6.079387872/6.089423/6.089423 | 0/266 (0.000000%) | 0/266 (0.000000%) |
| validation | original_class | 3p0 | 356 | 146/149.915730337/146/180 | 6.006006/6.079815986/6.089423/6.089423 | 0/356 (0.000000%) | 0/356 (0.000000%) |
| validation | original_class | 3p1 | 173 | 146/149.341040462/146/180 | 6.006006/6.081225954/6.089423/6.089423 | 0/173 (0.000000%) | 0/173 (0.000000%) |
| validation | original_class | ft0 | 90 | 146/152.044444444/146/180 | 6.006006/6.074593311/6.089423/6.089423 | 0/90 (0.000000%) | 0/90 (0.000000%) |
| validation | original_class | ft1 | 277 | 146/151.277978339/146/180 | 6.006006/6.076473791/6.089423/6.089423 | 0/277 (0.000000%) | 0/277 (0.000000%) |
| validation | original_class | mp0 | 123 | 146/149.040650407/146/180 | 6.006006/6.081962943/6.089423/6.089423 | 0/123 (0.000000%) | 0/123 (0.000000%) |
| validation | original_class | mp1 | 91 | 146/149.362637363/146/180 | 6.006006/6.081172967/6.089423/6.089423 | 0/91 (0.000000%) | 0/91 (0.000000%) |
| combined | outcome | Make | 4579 | 145/153.129504259/146/180 | 6.006006/6.071005303/6.089423/6.089423 | 0/4579 (0.000000%) | 0/4579 (0.000000%) |
| combined | outcome | Miss | 4161 | 145/152.990627253/146/180 | 6.006006/6.071019524/6.089423/6.089423 | 0/4161 (0.000000%) | 0/4161 (0.000000%) |
| combined | shot_type | 2P | 2856 | 145/153.054271709/146/180 | 6.006006/6.070832327/6.089423/6.089423 | 0/2856 (0.000000%) | 0/2856 (0.000000%) |
| combined | shot_type | 3P | 2787 | 145/152.987082885/146/180 | 6.006006/6.071060471/6.089423/6.089423 | 0/2787 (0.000000%) | 0/2787 (0.000000%) |
| combined | shot_type | FT | 1942 | 145/153.366117405/146/180 | 6.006006/6.070372814/6.089423/6.089423 | 0/1942 (0.000000%) | 0/1942 (0.000000%) |
| combined | shot_type | MP | 1155 | 145/152.761038961/146/180 | 6.006006/6.072414594/6.089423/6.089423 | 0/1155 (0.000000%) | 0/1155 (0.000000%) |
| combined | original_class | 2p0 | 1204 | 145/152.934385382/146/180 | 6.006006/6.070612585/6.089423/6.089423 | 0/1204 (0.000000%) | 0/1204 (0.000000%) |
| combined | original_class | 2p1 | 1652 | 145/153.141646489/146/180 | 6.006006/6.070992478/6.089423/6.089423 | 0/1652 (0.000000%) | 0/1652 (0.000000%) |
| combined | original_class | 3p0 | 1796 | 145/153.145879733/146/180 | 6.006006/6.070612359/6.089423/6.089423 | 0/1796 (0.000000%) | 0/1796 (0.000000%) |
| combined | original_class | 3p1 | 991 | 145/152.699293643/146/180 | 6.006006/6.071872589/6.089423/6.089423 | 0/991 (0.000000%) | 0/991 (0.000000%) |
| combined | original_class | ft0 | 490 | 145/152.987755102/146/180 | 6.006006/6.07137768/6.089423/6.089423 | 0/490 (0.000000%) | 0/490 (0.000000%) |
| combined | original_class | ft1 | 1452 | 145/153.493801653/146/180 | 6.006006/6.070033707/6.089423/6.089423 | 0/1452 (0.000000%) | 0/1452 (0.000000%) |
| combined | original_class | mp0 | 671 | 145/152.678092399/146/180 | 6.006006/6.072577982/6.089423/6.089423 | 0/671 (0.000000%) | 0/671 (0.000000%) |
| combined | original_class | mp1 | 484 | 145/152.876033058/146/180 | 6.006006/6.072188079/6.089423/6.089423 | 0/484 (0.000000%) | 0/484 (0.000000%) |

## 異常與待人工檢查項目

```json
[]
```

## 解讀與未決問題

- 原始 <64 frames：0/8740 (0.000000%)；25 FPS 等效 <64 frames／<2.56秒：0/8740 (0.000000%)。
- 若 short 子集為空，looping 分布與最嚴重案例為空／N/A，而非聲稱一個未觀察到的曝光比率。
- 長度充足僅表示 metadata 層面的64-frame／2.56秒窗口相容；未保證實際重採樣端點、內容覆蓋、逐格時間戳或模型可行性。
- 不同原生 FPS 下固定 frame count 對應不同秒數；後續 protocol 須明定時間基準、duration來源、取整／端點及短片處理，但本輪不選擇任何政策。
- 本輪無完整解碼，不能重新證明逐格可解碼性或排除 VFR／PTS 問題；小 metadata 差值不等於影像損壞。
- Gate 1A 只涵蓋時間資料稽核；權重 provenance、shape、VRAM、throughput 與其他 Gate 未在本輪執行。
- 未實作 I3D、未下載 checkpoint、未訓練／評估、未讀取 test 影片、未修改 A1/A2/B/C、未 commit 或 push。

## 重現命令

```bash
python3 -B scripts/audit_i3d_temporal_feasibility.py --help
python3 -B scripts/audit_i3d_temporal_feasibility.py --mode check
python3 -B scripts/audit_i3d_temporal_feasibility.py --mode audit
python3 -B scripts/audit_i3d_temporal_feasibility.py --mode replay
```

audit 重做 allowlist metadata probe；replay 僅讀 /tmp cache、核對 hash／身份並重算，不開影片。重跑統計 hash 應相同；執行時間與來源 provenance 可不同。

Statistics SHA-256: `c94edd0869bb3e4de93f000edb6340b391ab909be8c8091504466ba22f5c828b`
