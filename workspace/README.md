# 研究工作台

更新：2026-11-19。八類baseline已完成12輪、5400次更新，依最低validation CE選第7輪；LSTA已完成，每小時排程現已改追蹤TSCI並啟用。先看[最新狀態與完整執行順序](../docs/research/ACA_CURRENT_STATUS.md)，再看[ACA架構缺口與待決事項](../docs/research/ACA_ARCHITECTURE_GAPS.md)。目前唯一候選的[CPU與有限ACA真實GPU驗收](../docs/research/ACA_ENGINEERING_STATUS.md)已通過；[單seed消融正式計畫與入口](../docs/research/ACA_SINGLE_SEED_LAUNCH.md)已備妥，baseline核定沿用；TSCI一組已啟動，詳見[啟動紀錄](../docs/research/TSCI_FORMAL_LAUNCH.md)。

已驗收的資料、八類介面與訓練工程見[資料輸入入口](../docs/research/input_processing/README.md)；正式baseline完成紀錄見[正式入口](../docs/research/input_processing/FORMAL_LAUNCH.md)。歷史文件的「下一步」保留當時語境，以最新狀態頁為準。

先看[影片輸入調整總表](../docs/research/input_processing/INPUT_CHANGES.md)，再看[固定16支事件參考](../docs/research/input_processing/EVENT_REFERENCE.md)。瀏覽圖片可開啟[index.html](index.html)。

| 資料夾 | 日常用途 |
| --- | --- |
| [01_active_input](01_active_input/README.md) | 目前唯一採用的50幀RGB基準、契約及驗收 |
| [02_event_review](02_event_review/README.md) | 本輪16支事件表與證據；不是新模型 |
| [03_candidates](03_candidates/README.md) | 明確分開固定5%否決方案與尚未啟用候選 |
| [04_history](04_history/README.md) | ACA、TANet、主線及其他歷史實驗；不自動續跑 |
| [05_shared_resources](05_shared_resources/README.md) | 共用資料清單、來源程式與研究報告 |

**後續已完成實體產物整理。** 兩個工作樹的37個產物目錄已搬至主專案outputs，依目前／否決／未啟用／歷史分類。這個workspace仍是程式與文件的導覽；development中的舊artifacts位置只保留相容連結。原程式套件與凍結run路徑仍保留。

[實際產物入口](/home/tsai/projects/basketball/outputs/index.html) · [完整配置](/home/tsai/projects/basketball/docs/storage_layout.md)

請從原ACA工作樹執行程式，不要透過程式檔捷徑啟動。訓練、資料與歷史輸出仍使用原路徑。[完整位置對照](inventory.json)與[整理方式](../docs/research/input_processing/WORKSPACE_LAYOUT.md)可查每個入口實際指向哪裡。

後續文字摘要集中於docs/research/input_processing；大型事件證據集中於主專案outputs/01_current；既有development/artifacts相容連結保留。不要再為一般檢查新增一個模型資料夾。
