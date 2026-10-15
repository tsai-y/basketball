# 新研究工作區

> - **給誰看：** 要建立非 frozen 研究元件的實作者。
> - **回答什麼：** 新工作放在哪裡、需記錄哪些資訊、哪些路徑不可改。
> - **不包含：** A1/A2/B/C 細節或資料品質 CLI；後者集中在專屬技術入口。
> - **上一層入口：** [專案 README](../README.md)。
> - **建議下一份：** [資料品質技術入口](data_quality/README.md)。

本目錄承接與 frozen baseline 分離的新研究。現有模組為 `data_quality/`；完整 train 品質掃描與品質感知取樣已完成，視覺前處理／增強完成 train-only Pilot，正式政策與下一個受控實驗仍待團隊決定。

## 放置規則

- 新研究放在 `development/<研究主題>/`，先記錄問題、範圍、輸入、輸出及狀態；不要預先建立尚未定案的主題目錄。
- 不修改 [baseline](../baseline/README.md)、`reference_c3d/`、`experiment_c/`、其設定、歷史入口或 `runs/`。
- 共用[固定 manifest](../data/processed/dataset_manifest.csv)與[固定 split](../data/processed/split_summary.json)，不複製、不重抽、不依 test 結果回寫。
- 衍生品質資料以 video ID 關聯，不覆寫原始影片、manifest 或 split；validation/test 保持封存。
- 新輸出使用事前指定的獨立位置，不寫入 baseline 或歷史 runs；大型產物的版本管理方式需先審查。

## 每個實驗的必要紀錄

| 類別 | 必須記錄 |
| --- | --- |
| 問題與狀態 | 研究問題、範圍、負責人、日期、planned／running／completed |
| 資料與 split | manifest／split 路徑、版本或 hash、實際使用範圍 |
| 前處理 | 版本、參數、程式來源、train／validation／test 行為 |
| 設定與環境 | config、seed、環境、程式版本、實際命令與 override |
| 輸出 | 唯一輸出目錄、產物清單、完成狀態與不可覆寫規則 |

目前跨階段狀態見[目前研究方向](../docs/current_research_direction.md)。資料品質的 configs、scripts、samplers、transforms、tests、sidecars、CLI 與 runbooks 只由[資料品質技術入口](data_quality/README.md)導覽。
