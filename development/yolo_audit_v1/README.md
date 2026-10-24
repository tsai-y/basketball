# YOLO 人物定位 Audit v1

這是既有 YOLO11m 的偵測／追蹤診斷，不是偵測器訓練，也沒有接入
MixNet-I3D／Transformer。現有模型、特徵快取與品質標註不修改。

## 固定樣本與輸出

只使用固定 train：20 支、10 Make／10 Miss；2p、3p、ft、mp 各 5 支；
各原始類別至少 2 支。有 candidate／無 candidate 各 10 支，盡量分散
source_group 與 FPS。seed=20261024，以 SHA-256 排序，先保存 manifest
再推論。這是刻意平衡的診斷樣本，不代表全資料集的發生率。

每支影片完整逐幀推論，保留所有 conf>=0.25 的 person 框；追蹤採 ByteTrack，
每影片建立新 tracker。未確認的偵測框仍保存，track_id 為 null，不因追蹤失敗
就刪除偵測。記錄本機權重與設定檔 SHA、ultralytics/torch/OpenCV 版本。
imgsz=640、conf=0.25、iou=0.7、classes=[0]、max_det=100、float32。
由於 detector floor 為 0.25，ByteTrack 沒有收到 0.1–0.25 的低信心候選。

影片左側是原畫面，右側是人物框。D 為該幀偵測編號；T 為本影片的追蹤 ID，
T-- 是未確認追蹤。粉紅 ROI 只是「全部 person 框聯集＋10% 寬高邊界」的示意，
沒有球員／觀眾過濾，也沒有用於訓練。無框時 ROI 回到全畫面。
球、籃框是否落在 ROI 內由人工檢查，並非已經偵測到球或籃框。

每影片另固定取 10%、30%、50%、70%、90% 的 5 幀，共 100 張無損 PNG；
這些位置不是出手時刻標註。人物影片的 H264 CRF18 僅供可視化，不作模糊真值。
完整 JSON 保留每幀框、信心值、track ID、ROI 與時間；時間為 frame_index/FPS。

## 人工填表順序

先在 index.html 看 Y001–Y003，練習影片層欄位，再完成其餘影片。
用正常速度、0.25 倍速、暫停與逐格查看均可。分不清時用 uncertain，不硬猜。
XLSX 有兩個工作表；CSV 是同一初始表格的替代格式，請選一種作主要標註檔，
不要分別修改後混合使用。影像／影片相對路徑皆以本次 audit 資料夾為基準。

Video Audit（每影片一列）欄位：

- shot_event_visible：看得到本次投籃事件？yes/no/uncertain。
- pre_release_shooter_missed：看得出投籃者，但出手前曾漏框？yes/no/uncertain/n_a。
- shooter_dropout：投籃者可見期間漏框情形 none/brief/repeated/uncertain/n_a。
  brief 指孤立短暫漏框；repeated 指多次或持續漏框，這是人工描述，不是計算指標。
- non_player_false_boxes：觀眾、裁判、教練或看板人像被框？yes/no/uncertain。
- id_stability：只跟著本次投籃者判斷 stable/switch/lost/repeated_switch/uncertain/n_a。
  不同影片的 ID 無法相比；ID 數多不等於 ID switches 多。
- cut_or_replay：none/cut/replay/both/uncertain。
- outcome_visible：完整原畫面能判斷 Make/Miss？yes/no/uncertain。
- roi_would_lose_task_info：若只保留粉紅 ROI，是否會失去球、籃框、投籃者或結果資訊？yes/no/uncertain。
- overall_suitability：目前人物定位是否適合作為 ROI 輔助？usable/partially_usable/unusable/uncertain。
- first_issue_frame：選填第一個問題的 0-based frame_index；notes 說明現象。

Frame Audit（每固定幀一列）欄位：

- matched_player_boxes：可一對一對應到場上球員的偵測框數。
- non_player_or_duplicate_boxes：框到非場上球員、錯誤位置、或同球員的重複框數。
  裁判在 person 類別上不算錯，但對「球員 ROI」目標算非球員框。
- uncertain_boxes：無法確認的預測框數。前三欄相加必須等於 predicted_person_boxes。
- missed_visible_players：畫面可辨識為場上球員，但沒有對應框的人數。
  完全遮擋的人不算可見；分不清的身體局部請備註，不硬補人數。
- shooter_visible、ball_visible、rim_visible：yes/no/uncertain。
- shooter_detected、ball_in_roi、rim_in_roi：yes/no/uncertain/n_a。
  原畫面看不到或無法辨識對象時，不把它當成偵測／裁切失敗；用 n_a 或 uncertain。
- 不要把尚未查看的空白填 0；空白表示未標註，0 是已確認數量為零。

此表只支援人工視覺對應的初步計數，沒有 IoU 標註，因此不是 mAP。
球員 recall 需用「匹配球員＋漏掉可見球員」作分母；模糊不確定框應另報，
不得默默排除。只有模型輸出時不能算真正 precision/recall。
本表也不是微調 YOLO 所需的逐物件 bounding-box 訓練標註。

## 執行

在 `/home/tsai/projects/basketball`、yolo11 環境：

```bash
python -B -m unittest discover -s development/yolo_audit_v1/tests -p 'test_*.py' -v
python -B -m development.yolo_audit_v1.audit --dry-run
python -B -m development.yolo_audit_v1.audit --run --device 0
```

dry-run 不載入 YOLO、不開影片、不建立輸出。run 是離線推論，沒有訓練函式。
每次建立獨立 artifacts/audit_seed20261024_*，不覆寫舊檔。原片透過 train
allowlist 與唯讀 descriptor 開啟，檢查來源 identity 不變；validation/test 原片不開啟。
完成推論後以 build_annotations.mjs 使用 bundled Artifact Tool 產生 XLSX/CSV。
人工表格需要填完，才可決定 ROI 方法是否適合接入模型。
