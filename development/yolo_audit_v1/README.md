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

## 保留場景 ROI 小型可視化對照

`roi_probe.py` 只讀取本次 audit 已產生的 Y004、Y018、Y012、Y019 診斷影片與
逐幀 detection JSON，不重新讀原始資料集、不重跑 YOLO、不改人工標註、
特徵快取或分類器，不啟動訓練。這是已知失敗案例的開發檢查，非獨立驗證。

三欄依序為完整診斷畫面、舊 person 聯集 ROI、新的保留場景候選 ROI。
ROI 外顯示黑色，保留區不縮放，幀號與 FPS 與來源診斷片一致。

候選幾何規則（未證實有效，未以 validation accuracy 調參）：

- 無人物框、框少於3個、舊 ROI 面積小於35%、寬度小於50%，或其上緣已在
  畫面高度55%以下，退回全畫面。
- 其他情況保留完整寬度與頂部，高度至少90%，並包含舊 ROI 底部。
- 因此新 ROI 必然包含舊 ROI，且至少保留90%面積。它可能幾乎等同全畫面，
  不可把「沒再裁掉資訊」宣稱為「人物偵測改善」。
- 不區分球員與觀眾，也不辨識球、籃框、投籃或轉場。即使轉場中沒框，仍保留該幀。
- 人工複查的位置僅供導覽，沒有作為 ROI 規則輸入；Make/Miss 標籤及品質資訊也不用。

執行例（CPU，輸出自動另建在本次 audit 的 `outputs/roi_context_probe_v1_*`）：

```bash
python -B -m development.yolo_audit_v1.roi_probe --audit-run development/yolo_audit_v1/artifacts/audit_seed20261024_7_aaaj0r --dry-run
python -B -m development.yolo_audit_v1.roi_probe --audit-run development/yolo_audit_v1/artifacts/audit_seed20261024_7_aaaj0r --run
```

輸出含4支三欄影片、24張 PNG、8張複查圖、逐幀 ROI 與備援原因 JSON、設定／來源
SHA-256 與保留面積統計，以及可跳到指定幀和慢放的檢查頁。規則觸發備援與
擴張後自然變成全畫面分開統計。這些是幾何／工程資訊，不能計算 mAP、
真正球員 precision/recall 或分類模型準確率；診斷轉碼畫面也不作模糊真值。

## 自訂 YOLO 三類標註可行性試驗

`target_label_pilot.py` 從既有 train audit 中選8支不同 source_group，
各投籃類型與 Make/Miss 組合各1支，排除曾用於 ROI 修正的 Y004/Y012/Y018/Y019。
SHA-256排序與seed固定，每支取30%、50%、70%的既有無損PNG，共24張。
影像取初始audit PNG的未畫框左半邊，保持320×240，不從H264診斷片重新抽幀。
這仍是開發／標註可行性試驗，不是獨立效能驗證，也不足以宣稱可訓練出可靠偵測器。

類別與規則版本 `active-player-ball-visible-rim-v1`：

- `0 active_player`：球場比賽畫面中參與比賽球員，包含場外發球員；排除裁判、
  觀眾、教練、板凳球員。框可辨識的可見身體範圍，不猜遮擋部位。
- `1 ball`：可辨識的實體籃球；不猜測球的位置，不以標誌或手替代。
- `2 rim`：可辨識的籃圈本體，不包含籃網、籃板或支架。
- 球員特寫、轉場或無法可靠框定的影像可填原因後略過，不能直接當成負樣本。
- 球或圈過小無法框定時記 `uncertain`，這張不得宣告訓練標註完成。
  「沒有可見物件」與「不知道在哪」必須分開。

### 選填持球角色（不改三類偵測）

角色規則 `visible-hand-held-ball-v1`：只在既有 `active_player` annotation
附加 `holding_ball`，值為 `yes/no/uncertain/null`。`null` 或舊檔缺欄位表示
尚未判斷，不預設為 `no`；`yes` 僅指目標幀可見該球員以手持住球，`no` 指
可確認未持球，遮擋／爭搶／運球控制關係不清楚用 `uncertain`。
這不是投籃者身分、跨幀控球權或第四個偵測類別，也不重複畫球員框。

`role_metadata.py` 的 `add_role_metadata` 是非破壞性複製：保留既有角色值，
只為缺欄位的球員填 `null`，不從球的位置、框距離或已接受物件框推定角色。
Python／JS均拒絕非法值及球／籃圈上的此欄位；舊進度仍可讀。
角色未完成或不確定不使已通過的三類物件標註退回pending，兩種完整性分開。

一般框編輯器可在球員列選角色；`holding_ball_editor.html/js` 是只改角色的
專用頁模板，不提供框／類別／場景／物件完整性修改。產生專用頁時需嵌入
當前進度和唯一暫存namespace，避免載入舊頁暫存覆蓋新進度。
本機頁暫存不自動回寫對話中的JSON，操作後需匯出JSON再交回同步。
角色保存在複查JSON，不會自動成為標準YOLO三類偵測訓練目標；目前不訓練。

只將官方 person 框作橘色虛線提案，初始人工框全部為空、狀態全部pending。
人工需要拒絕非球員提案、補漏框並修正邊界，再標球與籃圈。提案接受、
全部類別的有無、完整性確認與影像複查狀態分開保存，不能盲目把預測當真值。

```bash
python -B -m development.yolo_audit_v1.target_label_pilot --audit-run development/yolo_audit_v1/artifacts/audit_seed20261024_7_aaaj0r --dry-run
python -B -m development.yolo_audit_v1.target_label_pilot --audit-run development/yolo_audit_v1/artifacts/audit_seed20261024_7_aaaj0r --prepare
```

輸出在原audit的 `outputs/target_label_pilot_v1_*`，含24張PNG、來源manifest、
空白人工草稿及本機標註頁。先做L001/L002校準即可，不必一次標完。
可拖曳畫框、確認／拒絕提案、改座標、刪除框、逐類記錄有無／不確定及略過原因。
本機暫存可能遺失，離開前請匯出進度JSON；可匯入相同批次續做。
頁面不顯示 Make/Miss 標籤，這些標籤只用於初始抽樣分層。

匯出後檢查（唯讀，例中路徑改為實際輸出與下載的JSON）：

```bash
python -B -m development.yolo_audit_v1.target_label_pilot --pilot-dir /absolute/path/to/pilot --check-review /absolute/path/to/downloaded_review.json
```

此階段沒有訓練函式、不匯出YOLO訓練labels、不產生data.yaml，也不改MixNet-I3D、
分類器、特徵快取、品質標註或先前AI/人工複查紀錄。未讀validation/test或原始影片，
未重跑YOLO。互動頁使用本機檔案，不上傳資料；純JS驗證可用bundled Node執行
`tests/test_label_core.cjs`，但這不等於已驗證實際瀏覽器互動。
