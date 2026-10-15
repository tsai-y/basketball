# Basketball-51 籃球影片研究

> - **給誰看：** 第一次接觸專案的組員與需要確認目前範圍的人。
> - **回答什麼：** 專案在研究什麼、目前到哪裡、接下來應讀哪份文件。
> - **不包含：** 完整實驗數字、技術命令、資料品質方法全文與歷史進度。
> - **上一層入口：** 無；本頁是 repository 的正式入口。
> - **建議下一份：** [目前研究方向](docs/current_research_direction.md)。

本專案研究 **full-video Make/Miss classification**（完整影片投籃命中／未進分類）。A1/A2/B/C 已完成訓練與一次性正式 test，均為 frozen baseline；完整影片結果不能解讀為 pre-release prediction 能力。

目前資料品質完整 train 掃描與 Quality-aware Clip Sampler v1 已完成，視覺前處理／增強完成 train-only Pilot，正式前處理政策與下一個受控實驗仍待凍結。跨階段狀態與優先順序只以[目前研究方向](docs/current_research_direction.md)為準。

## 組員最短閱讀路徑

### 一般組員

1. 本頁 `README.md`
2. [目前研究方向](docs/current_research_direction.md)
3. [資料品質研究索引](docs/data_quality/README.md)

### 實作者

1. 本頁 `README.md`
2. [資料品質技術入口](development/data_quality/README.md)
3. 該入口指向的 method 或 runbook

### Baseline 查詢

1. [Baseline 入口](baseline/README.md)
2. [Research Baseline v1.3](docs/research_baseline_v1.3.md)
3. [歷史進度](docs/progress_log.md)

不需要依序閱讀 repository 內全部文件。

## 文件與程式入口

| 入口 | 用途 |
| --- | --- |
| [baseline/README.md](baseline/README.md) | A1/A2/B/C 定義、frozen 邊界、程式與歷史產物入口 |
| [development/README.md](development/README.md) | 新研究工作區的放置與紀錄規則 |
| [development/data_quality/README.md](development/data_quality/README.md) | configs、scripts、samplers、transforms、tests、sidecars 與 runbooks |
| [docs/README.md](docs/README.md) | 研究文件的分類索引 |
| [docs/data_quality/README.md](docs/data_quality/README.md) | 給一般組員的資料品質研究導覽 |
| [docs/repository_map.md](docs/repository_map.md) | 目錄責任與保留原位的原因 |
| [discord_project_assistant/README.md](discord_project_assistant/README.md) | 獨立 Discord 專案助理模組的安全、部署與操作說明 |

## 目前邊界

- A1/A2/B/C 的模型、sampling／preprocessing、設定、split、checkpoint 選擇與正式產物維持 frozen；不要因 test 結果回頭修改或重跑。
- Gate 1A／1B completed；Gate 1C 與原 Experiment D paused。I3D、YOLO 特徵／ROI 與融合尚未開始。
- 品質候選不等於確認模糊、排除名單或刪除授權；原始影片、固定 manifest／split 與歷史 runs 保持原位。
- Pre-release 與 SPL Open Data 延伸不在目前範圍。

完整限制與決策見[目前研究方向](docs/current_research_direction.md)；A1/A2/B/C 結果見 [Research Baseline](docs/research_baseline_v1.3.md)；歷史變更見[進度紀錄](docs/progress_log.md)。
