# 資料處理基礎流程：七步完成，基準版本已凍結

**2026-11-16後續：已完成固定16支[時間／裁切採用判斷](INPUT_CROP_RETENTION_DECISION.md)，不採用本輪固定5%方案；正式輸入維持本頁的50幀RGB基準。未訓練。**

日期：2026-11-15。依使用者決定保留完整片段、記錄來源內容問題，完成固定基準資料處理；沒有啟動訓練。

| 步驟 | 工作 | 結果 |
| --- | --- | --- |
| 1 | 固定資料版本、標籤與split | 完成；原來源分組不變，套用既定八類歧義隔離 |
| 2 | 單支原片至實際輸入核對 | 完成；使用者決定維持160×120 |
| 3 | 固定16支、每類2支對照 | 完成；技術通過，50幀接觸表已檢視，內容問題已記錄 |
| 4 | 只處理已證實問題 | 完成；依使用者決定保留整片，不刪幀、裁切或改標籤 |
| 5 | 封裝唯一可重現的loader | 完成；重用既有像素處理，加入固定八類／二類與排除規則 |
| 6 | 全量train／validation解碼驗收 | 完成；7192＋1547＝8739支，0失敗 |
| 7 | 凍結與版本保存 | 完成；契約、程式指紋、逐片來源／張量SHA及小型發布摘要已保存 |

## 已凍結的資料版本

版本：`native_uniform50_whole_clip_eight_class_v1`。

| 範圍 | 數量 | 本輪處理 |
| --- | ---: | --- |
| train | 7192 | 全量完整解碼與輸入檢查 |
| validation | 1547 | 全量完整解碼與輸入檢查 |
| test | 1538 | 僅metadata，未開啟影片 |
| excluded | 34 | 既定去重／八類歧義隔離，未開啟影片 |

原manifest與split assignment未覆寫，沒有啟用80/10/10候選。原八類順序為2p0、2p1、3p0、3p1、ft0、ft1、mp0、mp1；同時輸出八類0至7及Miss=0／Make=1對映。標籤映射一致不等於已人工重新裁定真值。

輸入流程固定為：原生影片完整解碼 → 端點均勻取50幀 → BGR轉RGB → 全圖縮為160×120 → ImageNet常數正規化 → contiguous CPU float32 `[3,50,120,160]`。RGB三通道代表顏色，沒有另行抽取球員、球或籃框。未啟用D3虛擬25Hz、ROI、資料增強或新的內容選幀。

## 驗收證據

- 四項資料／存取邊界檢查通過：固定cohort及標籤、test預先拒絕、跨split／隔離ID拒絕、契約被修改時拒絕。
- 全8739支完整解碼至EOF，原始影格數／尺寸、50幀取樣、RGB輸出及張量shape／dtype／有限值／contiguous均通過。共解碼來源影格1,337,628幀，單片影格數範圍145至180。
- 全量驗收與固定16支先前保存的來源SHA、正規化張量SHA全部一致。逐片來源與張量SHA已保存，便於之後檢查是否改變資料。
- 本次單次全量驗收耗時約133.6秒，僅描述這次CPU／I/O執行，並非訓練或未來工時預測。
- 51個歷史受保護檔案SHA仍一致；原標籤、歷史權重與訓練程式未修改。

本流程累計8771次來源解碼：前期16支對照32次，加上本次全量8739次。累計不同影片為7192 train及1547 validation；test影片0支。另有逐片來源檔雜湊讀取，不是額外影片解碼或推論。

## 保留的內容問題與限制

使用者已選擇保留完整片段，統計卡、特寫、切鏡、遮擋只記為品質觀察。未因觀看結果更換固定16支樣本，也未新增內容篩選器。

| 固定樣本 | 已確認觀察 | 證據 |
| --- | --- | --- |
| 09：ft0_v120_003741 | 抽樣約0至1.96秒是全畫面統計卡，約2.59秒已回到罰球場景；未逐幀判定切換邊界 | [原圖／輸入](../../development/aca_eight_class_data_v1/artifacts/input_foundation_20261115_v1/representative16/09_comparison.png) |
| 10：ft1_v167_001943 | 原片已有前景頭部遮擋部分球場，片末切板凳席；不能據此宣稱整段結果不可見 | [原圖／輸入](../../development/aca_eight_class_data_v1/artifacts/input_foundation_20261115_v1/representative16/10_comparison.png) |

「基準資料處理完成」指這份固定資料清單與輸入流程可重現，且train／validation技術驗收完成。全量解碼不是全量人工看片、事件邊界標註或標籤真值驗證；16支觀察不能推論整體品質比例。這也不代表作者精確前處理已取得、準確率提高、90%目標已達成，或已授權開始訓練。

## 保存位置與使用

- [固定契約](specs/input_foundation_v1.json)、[小型凍結摘要](results/input_processing_foundation_release_v1.json)、[完整進度](results/input_processing_foundation_progress_v1.json)。
- [資料封裝與使用說明](../../development/aca_eight_class_data_v1/README.md)、[唯一loader入口](../../development/aca_eight_class_data_v1/foundation.py)。此入口尚未接入任何訓練排程。
- [全量驗收摘要](../../development/aca_eight_class_data_v1/artifacts/input_foundation_20261115_v1/full_audit_v1/summary.json)、[逐片紀錄](../../development/aca_eight_class_data_v1/artifacts/input_foundation_20261115_v1/full_audit_v1/records.jsonl)。大型逐片證據保留本機ignored artifacts。
- 固定16支50幀對照：[00–03](../../development/aca_eight_class_data_v1/artifacts/input_foundation_20261115_v1/representative16/overview_00_03.png)、[04–07](../../development/aca_eight_class_data_v1/artifacts/input_foundation_20261115_v1/representative16/overview_04_07.png)、[08–11](../../development/aca_eight_class_data_v1/artifacts/input_foundation_20261115_v1/representative16/overview_08_11.png)、[12–15](../../development/aca_eight_class_data_v1/artifacts/input_foundation_20261115_v1/representative16/overview_12_15.png)、[視覺觀察紀錄](../../development/aca_eight_class_data_v1/artifacts/input_foundation_20261115_v1/representative16/visual_review.json)。

模型建立、forward、optimizer更新與訓練均為0，沒有執行test或push。既有訓練與每小時排程維持暫停。
