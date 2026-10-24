# MixNet-I3D 末端局部微調 v1

目前狀態：完整 RGB 訓練、梯度累積、validation、早停與 checkpoint 流程已實作。
既有兩影片 20 步驗收、新訓練入口的真實 GPU dry-run，以及 72 項新增／回歸測試通過。
**依使用者要求，停在正式訓練前；未執行真實資料的 `--train`，沒有新的完整 validation accuracy。**

## 本版學習什麼

原始 RGB 三窗口 `[B,3,3,64,192,256]` → MixNet-I3D →
`[B,3,8,1536]` 時序特徵 → 既有 WindowTemporalTransformer → Make/Miss logit。

- 凍結 stem 與 `blocks[0]`–`blocks[4]`，前段在 `no_grad` 下運算。
- 解凍 `blocks[5]` 與 `conv_head`，共 2,245,700 個參數，learning rate `1e-5`。
- 所有 BatchNorm 的 affine 參數與 running statistics 都凍結，包含末端的 BN。
- Transformer 保持可訓練，共 1,450,497 個參數，learning rate `1e-4`。
- AdamW，weight decay `1e-4`，gradient clipping `1.0`。
- FP32，不用 AMP／TF32；固定 seed 20261024 與確定性設定。
- 不加 YOLO、RL、八類平衡抽樣或 candidate 品質降權，三窗口位置不變。

這是 warm-start 微調：骨幹由既有 frozen extractor 還原，分類端由固定 seed
20261024 的原版最低 validation loss checkpoint（epoch 6）還原。
初始化會核對兩份 checkpoint 的 SHA-256 與既有來源程式雜湊，不下載新權重。

在前一階段 preflight 中，舊 feature cache **只作更新前的一致性參考**；實際 backward 必須走 RGB →
末端骨幹 → Transformer，不能拿舊快取來更新骨幹。
新 `train_rgb` 不讀特徵快取，會在每輪重新解碼 RGB、重新計算特徵。

## 操作（專案根目錄、yolo11 環境）

```bash
python -B -m unittest discover -s development/partial_finetune_v1/tests -p 'test_*.py' -v
python -B -m development.partial_finetune_v1.preflight --dry-run
python -B -m development.partial_finetune_v1.preflight --steps 20
```

後兩項需要 CUDA GPU。若執行環境沒有暴露 `/dev/nvidia*`，會明確拒絕，不默默
退回 CPU。本次確認宿主 RTX 5070 Ti 正常，在允許 GPU 存取的執行環境完成驗收，
沒有更動驅動。

dry-run 用 train 中第一支 Miss 與第一支 Make：核對原 RGB 特徵／logit 與原 cache
一致，再做 backward；不建立 optimizer、不更新權重、不建立輸出資料夾。
正式 preflight 才對同兩支 train 影片反覆做 20 次 optimizer updates。
本輪不讀 validation／test 影片；初始化 head 曾由既有 baseline 的 validation 選模。

## 實際結果（2026-10-24）

- 更新前 feature 最大絕對差：`0.0`，與舊快取完全一致。
- 兩支 train 影片 eval loss：`0.0306554 → 0.000105058`。
- 最終 P(Make)：Miss `0.00010698`、Make `0.99989665`。
- 凍結權重與所有 BN buffers：逐項 bitwise unchanged。
- `blocks[5]`、`conv_head` 與 Transformer：梯度有限且權重確實更新。
- PyTorch peak allocated：`5.009 GiB`，不包含其他程式或全部 CUDA reserved memory。
- 同兩支已解碼影片的每步中位時間約 `0.125 s`；兩影片初次解碼約 `0.490 s`。
  這不是全資料訓練的吞吐量或可靠工時估計。

結果：[summary.json](artifacts/preflight_seed20261024_hxchdp42/summary.json)。
每次執行建立獨立資料夾，內含 config、history、summary 與 `engineering_only.pt`。
兩影片反覆擬合的權重只供工程驗收，**禁止當作正式微調的初始化 checkpoint**；
正式訓練必須重新由原 frozen extractor 與原版 head 初始化。

## 完整訓練設定（已固定）

- Train：7,193 支，每輪 shuffle、無放回，全部看一次，不丟最後一批。
- Validation：1,547 支，原始分布、固定順序，每輪完整 forward；不更新參數或 BN。
- 每支影片三個固定 64-frame RGB 窗口，保持與舊版相同的 reference transform。
- GPU micro-batch 2，累積 32 個 micro-batch，等效 batch 64。
- 7,193 = 112 × 64 + 25，因此每輪 113 次 optimizer updates；最後一組除以實際 25 支，
  並非除以 64。每個 micro-batch 先用 summed BCE 再除以該組實際影片數。
- 每組累積結束後才做 gradient clipping 與 optimizer.step，micro-batch 間保留梯度。
- 骨幹末端 lr `1e-5`、Transformer lr `1e-4`，其餘骨幹與所有 BN 凍結。
- 最多額外 50 輪；完整 validation loss 監控，patience 10、min_delta `0.0001`。
- 預測閾值固定 0.5；最低 validation loss 選模，不挑 accuracy 最高的一輪。
- 包含微調前 epoch 0 的完整 validation：如果後續都變差，`best.pt` 保留 epoch 0。
- 實際最低 loss 選模與 min_delta 早停分開，微小改善可能更新 best 卻不重置 patience。
- 每 50 個 micro-batch 顯示進度（預設相當於 100 支影片），不只在 epoch 結束才輸出。
- DataLoader workers 0、CPU threads 4、OpenCV threads 1，先採穩定的單程序解碼。
- 不使用 YOLO、RL、quality weighting 或八類平衡抽樣；test 不載入。

這版 `--seed` 控制微調 shuffle／dropout，但 warm-start head 固定來自 seed 20261024
的既有 checkpoint；更改 `--seed` 不會自動改成另一份 baseline head。

## 交接：由使用者在本機啟動

新入口必須明確選 `--dry-run` 或 `--train`，不帶模式會直接拒絕。
安全預檢（已執行通過；可自行重跑）：

```bash
conda activate yolo11
cd /home/tsai/projects/basketball
python -B -m development.partial_finetune_v1.train_rgb --dry-run
```

dry-run 檢查全部 train/validation 名單、類別數與 source disjoint；只解碼兩支 train、
兩支 validation，執行 train backward 與 validation forward，確認全部 state 不變。
不建立 optimizer、不啟動正式訓練、不寫 run 或 checkpoint。
本次另以 mock 封鎖 optimizer、`run_training` 與 run 目錄建立，實際 GPU dry-run 仍通過。
因此 probe 的 loss 不能當作完整 validation loss，亦未遍歷解碼全部 8,740 支影片。

以下才會真正開始全量流程，**本次沒有替使用者執行**：

```bash
conda activate yolo11
cd /home/tsai/projects/basketball
python -B -m development.partial_finetune_v1.train_rgb \
  --train --epochs 50 --seed 20261024 --micro-batch 2
```

啟動後先看見 `Epoch 00`：它會先完整驗證原始 warm-start 權重，尚不更新參數；
完成並儲存 epoch 0 後才開始 epoch 1 訓練。
這版每輪重新解碼影片，預期比舊快取分類訓練慢，耗時以第一輪實測為準。

若 GPU 顯存不足，可改 `--micro-batch 1`；程式會自動累積 64 次，仍保持有效 batch 64。
重跑會建立新 run，不覆蓋舊結果；更動 micro-batch 仍應記錄，dropout 的隨機運算分組可能不同。

## 全量訓練輸出與中斷限制

每次建立 `artifacts/rgb_seed20261024_<random>/`：

- `config.json`：超參數、兩份初始化 checkpoint 身分、完整 split 名單雜湊及程式來源雜湊。
- `history.csv`：從 epoch 0 起記錄 validation；epoch 1 起包含 train loss 與 optimizer updates。
- `best.pt`：最低 validation loss 的完整骨幹＋Transformer（可能是 epoch 0）。
- `last.pt`：最後完成一輪的完整模型（可能不是最佳），epoch 1 完成後才產生。
- `summary.json`：所選 best 的整體／逐類／逐投籃類型指標，並對照 initial validation。
- `validation_predictions.csv`：所選 best 的全部 1,547 支預測。

每次完整 validation 得到的 logits／labels／IDs 與模型一起存入 checkpoint；最終報告
核對身分與指標後使用這份已保存的完整預測，不必再重解碼一輪影片。
best／last 使用 atomic replace，只替換目前新 run 中對應的檔案；原 baseline 不變。

**目前不支援無縫續訓（未保存 optimizer 與完整 RNG 狀態）。**
`Ctrl+C` 或程序中斷會失去尚未完成的那輪進度；已完成的 best／last 仍保留。
重新執行指令會由原 frozen extractor＋原 baseline head 重新開始，不是接續 last。
不要以 `engineering_only.pt` 或已擬合兩支影片的權重初始化正式實驗。

## 驗證範圍

27 項局部微調測試，另加 MixNet-I3D 14 項、資料管線 13 項、temporal baseline 18 項，
合計 72 項通過。新增 optimizer 更新測試使用小型合成向量與測試模型，不是正式 RGB 訓練。
涵蓋與大 batch 等價的梯度累積（含最後 25 支）、micro-batch 1／2、錯誤輸入與順序拒絕、
不丟影片、驗證不更新權重、早停、epoch 0 保留、best／last／history／prediction 對應。

warm-start、GPU、有效 batch size 與額外訓練步數都需如實記錄，不能將結果直接
說成「只改解凍範圍的嚴格單變因實驗」。若要單獨歸因於解凍骨幹，還需相同
warm-start／訓練預算的 head-only continuation 對照。
