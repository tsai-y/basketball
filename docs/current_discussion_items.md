# Basketball-51 目前待討論與決策事項

更新日期：2026-10-12

本頁彙整下一輪研究或部署開始前需要由團隊確認的事項。現有證據顯示，完整 train 品質掃描、品質感知取樣與視覺前處理 Pilot 已完成；目前真正的阻塞不是再做一次掃描，而是凍結正式前處理政策、定義下一個可比較實驗，以及同步專案現況文件。A1／A2／B／C、固定 manifest／split 與正式 test 產物維持 frozen。

## 會議希望完成的決定

本次討論至少應完成以下四項決定：

1. 指定最新狀態的唯一來源，並同步仍停在 2026-10-07 舊狀態的入口文件。
2. 凍結第一版正式前處理政策，決定 reference、mild spatial、gamma、photometric 與 horizontal flip 的去留。
3. 定義下一個受控實驗的研究問題、模型路線、比較組、驗收指標與輸出位置。
4. 決定未追蹤的 Discord 專案助理是否納入本 repository，以及採用何種部署與 AI 模式。

## 已確認的目前狀態

| 項目 | 目前狀態 | 可直接採用的證據 | 不可過度解讀之處 |
| --- | --- | --- | --- |
| A1／A2／B／C | **Frozen**；已完成一次性正式 test | [Research Baseline v1.3](research_baseline_v1.3.md)、[正式 test protocol](formal_test_evaluation_protocol.md) | 是完整 full-video 流程比較，不是單變因結論，也不代表 pre-release 能力 |
| 完整 train 品質掃描 | **完成**；7,193 支、1,105,146 frames、18,390 candidate frames、1,798 segments | [品質感知取樣方法](data_quality/quality_aware_clip_sampler_v1.md)、[資料品質索引](data_quality/README.md) | candidate 是低局部 Laplacian 候選，不是人工模糊真值或刪除名單 |
| 品質感知取樣 | **完成**；64-frame、epoch 0 時 7,188 支可選零 overlap，5 支 fallback 共 overlap 20 frames | [Quality-aware Clip Sampler v1](data_quality/quality_aware_clip_sampler_v1.md) | 64 frames 是目前 Pilot 設定，不等於已選定所有後續模型的輸入長度 |
| 視覺前處理／增強 | **Pilot 完成，正式政策未定案**；48 支 train、每支 64 frames | [Visual Preprocessing Pilot v1](data_quality/visual_preprocessing_pilot_v1.md) | 沒有 validation accuracy、NR-IQA 或連續播放人工審查，不能宣稱改善模型表現 |
| I3D／YOLO／融合 | **尚未開始**；Gate 1C 與原 Experiment D 仍 paused | [目前研究方向](current_research_direction.md)、[資料品質索引](data_quality/README.md) | Gate 1A／1B 完成不代表已選定 I3D 或可直接開始訓練 |
| Pre-release／SPL | **不在目前範圍** | [目前研究方向](current_research_direction.md) | 既有歷史文件中的長期構想不是目前排程 |
| Discord 專案助理 | **提交候選**；完整測試與 local dry-run 已通過，尚未部署 | `discord_project_assistant/`、本頁的驗證紀錄 | 通過本機驗證不等於已完成真實 Discord／Codex 登入部署、安全審查或團隊採用決策 |

## 決策一：同步目前狀態與唯一來源

### 現況

較新的[資料品質索引](data_quality/README.md)記載完整掃描、v3.1 publication、品質感知 sampler 與 Visual Preprocessing Pilot 已完成；但下列入口仍保留較早的「尚未開始／只整理文件」敘述：

- [目前研究方向](current_research_direction.md)更新於 2026-10-07，仍稱只建立 repository 結構與文件。
- [根目錄 README](../README.md)仍稱沒有開始資料清洗、模型實作或訓練。
- [Development README](../development/README.md)仍稱本輪只建立目錄與文件。
- [文件索引](README.md)仍稱完整 train publication 與清理政策尚未完成；其中 publication 已完成，清理政策確實仍未定案。
- [進度紀錄](progress_log.md)已有 sampler 紀錄，但沒有獨立的 Visual Preprocessing Pilot 完成節點。

這會直接影響人員閱讀與 Discord Bot 的 `/status`、`/next` 回答；Bot 能偵測日期衝突，但不應以衝突偵測取代文件治理。

### 需要決定

- 是否指定 `docs/current_research_direction.md` 為唯一的跨階段「Current」狀態頁，其他 README 只保留摘要與連結。
- 是否將本頁作為會議中的暫時決策登錄，決議後再把結論回寫到各 current 文件。
- 誰負責同步，以及完成日期。

### 建議方向與完成條件

建議保留 `current_research_direction.md` 為唯一狀態來源，將已完成、進行中、下一步、暫停與範圍外項目一次更新；其他入口不再複製易過期的細節。完成條件是上述四份入口對「掃描、sampler、Pilot、Gate 1C、Experiment D」沒有互相衝突的敘述，並補一筆 Visual Preprocessing Pilot 的進度紀錄。

## 決策二：凍結正式視覺前處理政策

### 目前證據

Visual Preprocessing Pilot 已驗證輸出 shape、範圍、決定性、連續 frame index、seed 與 metadata 契約；但只有 48 支 train 影片、稀疏 contact sheets，沒有模型效果或連續播放人工審查。

| 策略 | 目前判讀 | 需要團隊決定的問題 |
| --- | --- | --- |
| reference | 可作研究對照；保留全幅 4:3 語義 | 是否凍結為 validation／test 與第一輪比較的唯一預設 |
| mild spatial | 候選；平均保留 92.74% 原面積，抽查主體仍可見 | 是否先做邊緣人物、球與籃框的人工時序審查，再進入受控 validation 比較 |
| gamma 0.95 | 略提亮，未見新 halo，但可見性收益未證明 | 保留為候選、停止，或要求另一輪人工審查 |
| photometric | 平均 clamp 前超界 2.6104%，個案最高 36.7377% | 是否明確拒絕目前參數作正式預設，只保留程式介面供後續重新校準 |
| horizontal flip | 會反轉比分文字並改變球場方向與慣用手 | 是否排除，除非先證明 Make/Miss 任務的 label invariance |

### 建議方向與完成條件

建議先凍結 `reference` 為唯一正式預設；`mild spatial` 只作下一輪受控候選；目前 photometric、gamma 與 horizontal flip 不進入預設。這是保守研究起點，不是宣稱 reference 最佳。完成條件是一份版本化 policy，明列 train／validation／test 行為、參數、seed、metadata、禁止項目與變更規則。

## 決策三：定義下一個受控實驗

### 需要先回答的研究問題

團隊應先在以下路線中選一個主要問題，避免同時改變 sampler、前處理、backbone 與融合方式：

1. **資料管線比較**：在同一個非 frozen 新實驗入口中，比較 reference 與 mild spatial；模型與取樣固定。
2. **特徵模型選型**：先決定 I3D、YOLO ROI 或既有 C3D-style 特徵的優先順序，再建立各自 protocol。
3. **融合研究**：只有在至少兩個單模型已有可比較的 validation 證據後才啟動；目前尚不具備此條件。

### 必須一起凍結的欄位

- 主要與次要 validation 指標，以及最低有意義差異。
- 比較組、seed 數量、epoch／early stopping、checkpoint 選擇規則。
- clip length 與每影片 clip 數；不能只因 64-frame clean availability 高就直接視為模型最佳設定。
- 是否使用 quality-aware sampler，以及 fallback 5 支影片如何記錄。
- normalization 與 pretrained checkpoint 相容性；若要啟動 I3D，需先重新開啟 Gate 1C 的明確決策。
- 新實驗的程式、設定與輸出位置；不得修改 frozen baseline 或歷史 `runs/`。
- test 封存條件：方法與超參數定案前不得因 test 結果回頭調整。

### 建議方向與完成條件

建議下一步先做「資料管線整合 gate」，在 `development/` 建立新的、與 frozen baseline 分離的入口，先證明 sampler → decoder → transform → tensor → metadata 全流程一致。通過後只比較 reference 與 mild spatial，其他條件固定；是否啟動 I3D／YOLO 留到這個 gate 後另行決定。完成條件是 protocol、config、輸出位置、驗收指標與停止規則在執行前全部寫定。

## 決策四：品質候選、fallback 與 validation 邊界

### 需要決定

- 是否維持「candidate 不刪除資料，只影響 train clip 選擇」的 v1 政策。
- 5 支沒有 64-frame 零 overlap 窗口的影片，是保留並記錄 minimum overlap、人工複核，或從特定實驗排除。
- validation 目前沒有 quality sidecar 且 quality mode disabled；第一輪是否維持 reference sampling，或另立 protocol 以 train 凍結規則建立 validation sidecar。
- 是否需要人工標註一小批 candidate／non-candidate，才能討論正式清理門檻、precision／recall 與錯誤成本。

### 建議方向與完成條件

建議第一輪不刪除任何影片，保留 5 支 fallback 並把 overlap 寫入 sample 與訓練 log；若模型結果對這 5 支敏感，再做預先定義的敏感度分析。正式資料清理需先有人工真值與錯誤成本，不能由 v3 candidate ratio 直接推導。validation 是否建立 sidecar 必須在 protocol 中單獨決定，不可默認沿用 train 行為。

## 決策五：資料切分與結論強度

目前來源代碼尚未核實為真實 game ID，near duplicates 也尚未定量排除；因此下一輪需明確決定這些問題是：

- 只作目前方法比較的已知限制；或
- 在宣稱跨比賽泛化前必須完成的 blocking gate。

建議短期方法比較可在固定 split 上進行，但報告只能稱為既有來源代碼分組下的比較；若目標是對外主張 game-level generalization，則 game identity、`v160`／`v166` 關係與 near-duplicate 稽核必須先完成。依據見[資料集分析](dataset_analysis.md)。

## 決策六：研究產物與版本管理

### 需要決定

- `development/outputs/<實驗識別碼>/` 是否正式建立，哪些產物 tracked、哪些 ignored。
- ignored 的完整掃描、publication、contact sheets 與 runtime logs 如何備份、由誰保存、保存多久。
- clone 後拿不到影片、checkpoint 與大型 artifacts 時，團隊採用哪一套分發與復原流程。

### 建議方向與完成條件

在下一個實驗啟動前，先定義小型 config／summary／metrics 可提交，大型 checkpoint／逐幀資料／影像／runtime log 保持 ignored，並為每個 ignored 必要證據保存 hash 與備份位置。完成條件是新實驗不會寫入 frozen `runs/`，且另一位組員可依文件找到輸入、重建小型摘要並辨識不可重建的本機產物。

## 決策七：Discord 專案助理是否納入正式範圍

### 現況與風險

`discord_project_assistant/` 已完成提交前技術驗證，其設計為 repository 唯讀索引、allowlist、local fallback、最小 Discord 權限與 Codex fail-closed；但正式採用仍需要部署身分、憑證保管、Business 額度、使用者／channel allowlist、單一 instance 與維運責任人。

本次在獨立 `basketball-bot` 環境確認 `discord.py 2.7.1` 與 `PyYAML 6.0.3`，完整測試 **26／26 通過**。Local dry-run 為 **SMOKE PASS**，共索引 59 份文件、建立 1,043 個 chunks、0 skipped；尚未驗證真實 Discord 或 Codex 登入部署。

### 需要決定

- 採用、暫緩，或移到獨立 repository。
- 若採用，先以 `AI_PROVIDER=local` 上線，或在完成 sandbox／額度／登入審查後啟用 Codex。
- Docker 與 systemd 二選一；不得以同一 token 同時跑兩個 instance。
- 誰持有 Bot／Business workspace 身分、誰能執行 `/refresh`、每日與每人額度。
- 是否以獨立變更完成 code／security review，再進行合併與部署。

### 建議方向與完成條件

建議將 Bot 與研究政策更新拆成獨立變更；目前乾淨環境、pinned dependencies、完整測試與 local dry-run 已通過，合併或部署前仍需完成 allowlist、secret、權限與部署 review。第一階段可用 local retrieval，不必立即啟用 AI。完成條件是部署方式唯一、owner 與回復／停用流程明確，且 current 狀態文件已先消除衝突。

## 明確延後的項目

除非本次會議主動改變研究範圍，下列項目維持延後：

- Gate 1C、pretrained weights 下載／轉換與原 Experiment D。
- YOLO 特徵／ROI 與多模型融合實作。
- Pre-release annotation／prediction。
- SPL Open Data 整合。
- 依 candidate 自動刪除、改標或覆寫原始影片。
- 重新訓練、覆寫或依 test 結果調整 A1／A2／B／C。

## 會議決議紀錄

| 決策 | 結論 | 負責人 | 截止日期 | 要更新的文件／產物 |
| --- | --- | --- | --- | --- |
| Current 狀態唯一來源與同步 | 待討論 | 待指定 | 待指定 | `current_research_direction.md`、各 README、`progress_log.md` |
| Visual preprocessing policy v1 | 待討論 | 待指定 | 待指定 | 新 policy／protocol、config |
| 下一個受控實驗 | 待討論 | 待指定 | 待指定 | protocol、config、`development/` 入口 |
| Candidate／fallback／validation 政策 | 待討論 | 待指定 | 待指定 | sampler／evaluation protocol |
| Split 與 near-duplicate gate | 待討論 | 待指定 | 待指定 | dataset audit／研究限制 |
| 產物保存與備份 | 待討論 | 待指定 | 待指定 | output／ignore／backup 規則 |
| Discord 專案助理 | 待討論 | 待指定 | 待指定 | 獨立 review、測試與部署紀錄 |

