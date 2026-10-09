# Basketball-51 自動模糊品質篩檢 Pilot v1

狀態：**COMPLETE**；train-only；候選標記不是人工模糊真值，沒有正式清理。

開始時 branch `main`，HEAD `cafe04b62bb6ac4719a6c05131ded13b3d467024`；工作樹乾淨。

固定 seed 20261008；balanced_label_unique_source_sha256_v1。交替 Miss/Make，優先尚未使用的來源群組，SHA-256(seed:video_id) 排序破同分，最後按 video_id 排序。

影片 30；成功評分影格 4617；來源群組 30。

Metadata：Gate 1A per-video cache absent; probe selected pilot train only。沒有因分數或 probe 結果更換樣本。

## 指標分布及相關性

| 指標 | 最小 | P10 | 中位數 | P90 | 最大 |
| --- | ---: | ---: | ---: | ---: | ---: |
| laplacian_variance | 503.357491 | 2052.339911 | 2926.023224 | 3972.331270 | 6844.092948 |
| tenengrad | 7417.597474 | 26513.657885 | 36756.742786 | 44938.177516 | 67681.385469 |

逐影格 Pearson = 0.836563；Spearman = 0.840420。影格有時間相關，不能視為獨立樣本推論。

## 探索性門檻及影響

版本 `pilot_train_q10_s025_d020_v1`，Pilot train 全影格等權 P10，linear quantile；比較使用 ≤。

Laplacian ≤ 2052.339911078；Tenengrad ≤ 26513.657885417。只適用本 Pilot 尺寸、解碼、數值定義，不能當正式清理規則。

| 策略 | 候選影格 | 比例 | 連續區段 | 涉及影片 | 有16/32/64未標記連續窗的影片 |
| --- | ---: | ---: | ---: | ---: | --- |
| both_low | 327 | 7.08% | 47 | 14/30 | 30/30/28 |
| laplacian_low | 462 | 10.01% | 52 | 15/30 | 30/29/28 |
| smoothed_both_low | 314 | 6.80% | 15 | 11/30 | 30/30/29 |
| tenengrad_low | 462 | 10.01% | 54 | 16/30 | 30/30/27 |

區段端點 frame index 為零起始、兩端包含；timestamp 為解碼器時間且需通過單調及 CFR 一致性檢查。區段名義長度是 n/FPS，首末 timestamp 差為 (n−1)/FPS。未標記不代表內容可用；窗口只提供可行性資訊，不選定模型輸入長度。

## 類別、來源與 FPS

### fps

| 群組 | 影片/影格 | Lap低 | Ten低 | 雙低 | 平滑雙低 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 23.976 | 23/3357 | 12.09% | 13.23% | 9.20% | 8.82% |
| 29.97 | 7/1260 | 4.44% | 1.43% | 1.43% | 1.43% |
### label

| 群組 | 影片/影格 | Lap低 | Ten低 | 雙低 | 平滑雙低 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Make | 15/2258 | 7.75% | 9.39% | 6.11% | 5.62% |
| Miss | 15/2359 | 12.17% | 10.60% | 8.01% | 7.93% |
### source_group

| 群組 | 影片/影格 | Lap低 | Ten低 | 雙低 | 平滑雙低 |
| --- | ---: | ---: | ---: | ---: | ---: |
| v111 | 1/180 | 0.00% | 0.00% | 0.00% | 0.00% |
| v115 | 1/146 | 30.82% | 37.67% | 20.55% | 16.44% |
| v116 | 1/180 | 6.11% | 0.00% | 0.00% | 0.00% |
| v117 | 1/146 | 0.00% | 0.00% | 0.00% | 0.00% |
| v118 | 1/146 | 19.86% | 19.18% | 19.18% | 19.18% |
| v119 | 1/146 | 0.00% | 0.00% | 0.00% | 0.00% |
| v120 | 1/146 | 0.00% | 4.11% | 0.00% | 0.00% |
| v122 | 1/146 | 10.96% | 2.74% | 2.74% | 0.00% |
| v124 | 1/146 | 0.00% | 23.97% | 0.00% | 0.00% |
| v127 | 1/180 | 0.00% | 0.00% | 0.00% | 0.00% |
| v128 | 1/146 | 0.00% | 0.00% | 0.00% | 0.00% |
| v130 | 1/146 | 4.11% | 2.05% | 2.05% | 0.00% |
| v131 | 1/180 | 0.00% | 0.00% | 0.00% | 0.00% |
| v133 | 1/146 | 20.55% | 20.55% | 20.55% | 20.55% |
| v145 | 1/146 | 0.00% | 0.00% | 0.00% | 0.00% |
| v147 | 1/146 | 0.00% | 0.00% | 0.00% | 0.00% |
| v150 | 1/146 | 0.00% | 0.00% | 0.00% | 0.00% |
| v155 | 1/146 | 38.36% | 15.75% | 15.75% | 15.07% |
| v156 | 1/146 | 12.33% | 12.33% | 12.33% | 12.33% |
| v158 | 1/180 | 0.00% | 0.00% | 0.00% | 0.00% |
| v159 | 1/180 | 25.00% | 10.00% | 10.00% | 10.00% |
| v160_v166 | 1/146 | 31.51% | 32.19% | 28.08% | 29.45% |
| v161 | 1/146 | 4.11% | 41.78% | 4.11% | 0.00% |
| v163 | 1/146 | 9.59% | 13.70% | 8.22% | 8.22% |
| v167 | 1/146 | 0.00% | 0.00% | 0.00% | 0.00% |
| v169 | 1/146 | 0.00% | 0.00% | 0.00% | 0.00% |
| v175 | 1/180 | 0.00% | 0.00% | 0.00% | 0.00% |
| v181 | 1/145 | 14.48% | 13.10% | 13.10% | 13.10% |
| v183 | 1/146 | 6.16% | 6.16% | 6.16% | 6.16% |
| v184 | 1/146 | 75.34% | 58.90% | 58.90% | 62.33% |

來源代碼尚未核實為 game ID。每組僅少量影片；刻意增加來源覆蓋與類別平衡，因此比例不是完整 train 盛行率估計，類別/FPS/來源差異互有混雜。

## 存取與驗證證據

```json
{
  "test": {
    "decodes": 0,
    "opens": 0,
    "probes": 0
  },
  "train": {
    "decodes": 30,
    "opens": 60,
    "probes": 30
  },
  "validation": {
    "decodes": 0,
    "opens": 0,
    "probes": 0
  }
}
```

所有影片入口先經固定 train ID/path allowlist；probe 與 decode ledger 保留於本機 runtime_logs。非 train ID/path 拒絕測試不開啟影片。外部 syscall trace 與独立重算結果見 verification.json。

## 本機產物

根目錄：`development/data_quality/artifacts/blur_pilot_v1`。

- `frame_scores/`：每影片逐影格 CSV、四策略 flags/reasons、decode status。
- `segments.csv`、`video_summary.json`：候選與未標記區段、最長區段、连续窗口摘要。
- `contact_sheets/laplacian_variance_low_middle_high.png`、`tenengrad_low_middle_high.png` 與 index.csv：各低/中/高三張不同影片畫面，共 18 tiles，僅檢查演算法反應。
- `runtime_logs/`：probe metadata、decode status、存取 ledger、版本、syscall trace 與驗證。

## 能回答、不能回答與下一步

可回答原始解析度全畫面局部強度變化的相對強弱、探索性門檻會標記多少影格及時間區段，以及各群組受到的影響。不能證明真實模糊、球或投籃者清晰度、模型效果或正式資料可用性。低紋理可能低分；運動模糊與失焦不能可靠區分；高紋理背景可能掩蓋主體模糊；全畫面清晰不等於球或投籃者清晰；兩指標分數不是人工模糊真值。

Pilot 提供後續完整 train 評分的技術基礎；現行 CLI 上限 40 支，仍需另行授權與容量規劃，沒有自動擴展。正式清理政策仍待決定：品質定義、主體區域、跨來源/解析度校準、門檻與錯誤成本、持續時間、短片處理、保留或排除的審核規則。不得直接將候選轉為排除清單。

原始影片、固定 manifest/split、A1/A2/B/C frozen artifacts 未修改；沒有刪格、拼接、生成清理後影片、增強、特徵擷取、融合、訓練、評估或 checkpoint loading。Gate 1C/原 D 維持暫停。

## Pilot 觀察與擴展條件

雙低比例 Make 6.11%、Miss 8.01%；23.976 FPS 9.20%、29.97 FPS 1.43%。來源最高為 v184（58.90%），但每來源只有一支影片，不能解釋為 FPS 或類別造成品質差異。

兩張 contact sheets 已檢查，18 tiles 的 ID、位置、分數、旗標與版本可讀且與 CSV 一致。不同場景的背景紋理、字幕與鏡頭尺度明顯不同；這些畫面只供演算法行為檢查，沒有新增人工模糊標籤。

小樣本管線具備技術可行性，尚不具直接擴展完整 train 的執行條件：目前所有分數與 JPEG previews 暫存記憶體，完整 train 需改為分批或串流統計/畫面選取，並規劃執行時間、容量與失敗復原。這些修改及正式品質政策皆未開始。

## 完成驗證

```json
{
  "all_video_opens_read_only": true,
  "cli_checks": {
    "cli_help": "PASS",
    "dry_run": "PASS",
    "executed_command": "conda run --no-capture-output -n yolo11 python -B development/data_quality/scripts/audit_blur_pilot.py",
    "existing_output_rejected": true,
    "heldout_rejection": {
      "test": {
        "exit_code": 2,
        "reason": "ERROR: ID is not in the fixed train allowlist"
      },
      "validation": {
        "exit_code": 2,
        "reason": "ERROR: ID is not in the fixed train allowlist"
      }
    },
    "import_smoke": "PASS",
    "initial_strace_sandbox_attempt": "ptrace denied before program launch; escalated traced execution succeeded",
    "same_seed_cli_identical": true,
    "syntax": "PASS 3 files",
    "trace_command": "strace -f -qq -yy -e trace=openat,execve,rename,renameat,renameat2,unlink,unlinkat -o <local_trace> python -B development/data_quality/scripts/audit_blur_pilot.py",
    "train_guard_positive": "PASS"
  },
  "contact_sheet_tiles_verified": 18,
  "csv_json_markdown_consistent": true,
  "dataset_open_syscalls": 120,
  "descriptor_reopens": 60,
  "ffmpeg_execs": 0,
  "ffprobe_selected_video_execs": 30,
  "frame_scores_sha256": {
    "2p0_v145_002952_x264.csv": "e6aaef6d4cd61256832ad105f99b34a8b13f3c3656f6cc78dc6532288413958f",
    "2p0_v159_001620_x264.csv": "af1430cbae8a51a87590f550b979a2bfc7ea1915ff22ff9c996381991a878299",
    "2p0_v167_003410_x264.csv": "e842ca2304fd4d413198c67cd53a9ce434e4dc7437f3d47d7076495ce7e049f6",
    "2p1_v117_003955_x264.csv": "add7e0bcca1521b7f9c35991bf59effb1a4496d4145e47c17a23b9729a41562a",
    "2p1_v124_002736_x264.csv": "545f0e6696f2464e971b85ed2e8aa17b4ba22efb00539f0345131e92241083d8",
    "2p1_v130_001715_x264.csv": "1176c4ecaf4579d306e5302df1a10dfb13db4406870a7e432309708d60019205",
    "2p1_v158_000914_x264.csv": "4b6530491ffe022e85494c32aa9b74020bbd430de0ca42b28e6a5862f54d26d4",
    "3p0_v111_013808_x264.csv": "a89af587741e23cd098d89f8f4ae97e1c07e41a1c13d3aa100a94ef63ae3b163",
    "3p0_v116_013521_x264.csv": "2aaad5661e6bb4afa72974d8557c60e8fb8008fb79938a4e418526cb6387f903",
    "3p0_v118_001220_x264.csv": "5217f7c1f160ddf641d7d0e9430d4d2c2f220968962bd2d0fcb7de9e0a555100",
    "3p0_v161_013537_x264.csv": "8112294b8a3945501311d83a1dcce2b3de8a6f3cf19c53acfd702aea246306a6",
    "3p0_v183_002210_x264.csv": "31d4012cf34aecc4aeeca63cc053c4353a31993d1d7064afd0daf6150a9916f3",
    "3p0_v184_001944_x264.csv": "766de3c9a76d826bb28b838cd74f9ec24d93a8462f7fd8f52e75e8b4a6977f27",
    "3p1_v115_004311_x264.csv": "2afb168b7a95611327083f7fc31212c9c4f74d4e1fb029344378cdd318ce089f",
    "3p1_v122_000929_x264.csv": "e3a7630d7d00246c55b9ac2c660f2db09aafcab179cf7f6409d11b284f0fdfad",
    "3p1_v147_000602_x264.csv": "df3cb6b97f1bf564ae056d4388343adba19865a62185180bbc4480ab2579dbd4",
    "3p1_v150_003223_x264.csv": "d0b18f5fe674ff79dfc42a42c8967befbe6c4401b6eebc3876301fcc7bb183a0",
    "ft0_v175_004100_x264.csv": "2029058f2f13ec4a2892079cc8e1076954a8a9579c95ee5924466a59710dc050",
    "ft1_v119_010830_x264.csv": "56ad18ae971f0c8e16dedbbc634eb84d4cc8b1136f90c91fc56e4eab643a4a2c",
    "ft1_v128_011921_x264.csv": "a64e7d37970a586eab1200722302abd17124da63cf77b91f3b5c32a6077c44d6",
    "ft1_v131_010017_x264.csv": "482853f45a0484180e4dd96695403f28dcbd15cce1137aa84a2518b26f27f394",
    "ft1_v133_010514_x264.csv": "1b04d9bd12aee35776e2a487f1076edf299b2480381f80ae94f25d9488f7470b",
    "ft1_v160_003503_x264.csv": "eee8a0de3446ea3968bfe5f83eccddd61e03e40de21aada55087cb0fc7d04b00",
    "mp0_v120_000607_x264.csv": "ce9b561a9870bfb0388972fda896dc5a103f92a8287a7d98f233f25086375e48",
    "mp0_v127_000439_x264.csv": "d2d7031fc201aff4694beeec744de66089380270c0367ca4d4ca460b400f87b0",
    "mp0_v155_012140_x264.csv": "10d6e1fa196526c73346f9ad3a01600a40533f1ca8630bc6aede8bfde3c81e99",
    "mp0_v169_011852_x264.csv": "46912fbe493c3b0e54e0f91448d40e1813d4437328c72f52caab4a9abf5aa161",
    "mp0_v181_010625_x264.csv": "ad709bce258443c7930757264ea571032d9ce3561c2acea4c0e167aa38a4653d",
    "mp1_v156_011552_x264.csv": "7cc0b2ac824a4476e2bb2b7cbbf4d59e75556ce9aefa16efb9c40a4ab8eeb2ac",
    "mp1_v163_010146_x264.csv": "a61b66db8592a74bb5f0640f6096bb2734acf5f0a9dd79294b07b56ae7d6724b"
  },
  "frames": 4617,
  "git_diff_check": "PASS",
  "independent_flags_segments_windows_recalculation": true,
  "ledger_sha256": "3855adce9c4daa883d5a9a63d7133be3fcf201429e295ffb52b87348035c71a4",
  "literal_guarded_opens": 60,
  "protected_text_verification": {
    "expected_modified": [
      ".gitignore",
      "development/data_quality/README.md",
      "docs/progress_log.md"
    ],
    "original_video_checkpoint_hashes_performed": false,
    "progress_log_append_only": true,
    "status": "PASS",
    "unchanged_tracked_text_files": 102
  },
  "safety_tests": "6 PASS; every validation/test/excluded ID and mismatched path rejected before opener",
  "selected_videos": 30,
  "selection_reproducible": true,
  "status": "PASS",
  "syntax_import_help_dry_run": "PASS (see cli_checks.json)",
  "syscall_trace_sha256": "6ea7c53672de65efe77ad0addb195db2ec878ed6bf10bcd9c218d8311560853c",
  "test_opens_probes_decodes": 0,
  "unique_video_ids_opened": 30,
  "validation_opens_probes_decodes": 0
}
```
