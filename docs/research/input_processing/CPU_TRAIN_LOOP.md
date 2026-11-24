# 八類 baseline：訓練迴圈與中斷恢復的 CPU 驗收

日期：2026-11-17。**已完成隔離的八類迴圈核心與 CPU 合成驗收，8 項檢查全部通過。正式訓練與每小時排程維持暫停。**

目前流程進度：基準資料驗收 → 八類資料與指標介面 → 單一訓練規格 → CPU ImageNet 初始化與模型接線 → **本輪：訓練迴圈與恢復核心** → 尚待：真實 loader／GPU 端到端接線及正式啟動檢查。

## 本輪實作

新增 [loop_core.py](../../../development/aca_eight_class_data_v1/loop_core.py) 與 [test_loop_core.py](../../../development/aca_eight_class_data_v1/test_loop_core.py)，仍放在既有八類資料模組。沒有新增模型候選或大型產物目錄。

- 使用八類 targets、未加權 CE、Adam lr=1e-4、micro1／effective16；每個有效批次完成後裁剪一次梯度並更新一次。尾批除以實際支數，正式資料的尾批8與本輪24筆合成資料的尾批8具有相同計算路徑。
- 依 seed＋epoch 產生固定訓練順序；透過 records 核對每筆 ID 與 split。validation 依固定順序完整遍歷，只有完成後才更新歷史與選模。
- 嚴格較低的完整八類 validation CE 才換最佳輪次；同分保留較早輪、累加早停計數，patience5，30輪上限。
- 指標狀態獨立序列化，驗證八類順序、8×8混淆矩陣、樣本數及有限 CE；沒有修改已凍結的 adapter。
- 每25次更新、訓練輪結束、完整 validation 提交，以及訓練中要求暫停的 optimizer 邊界存檔。中斷可能重播尚未保存的更新，不累加丟失的工作。
- 載入前核對 schema、plan／程式／合成資料身分、CPU runtime、模型形狀／dtype、Adam 設定與狀態、cursor、指標及歷史。舊二類 checkpoint 不相容。

本輪對象是 CPU 合成案例專用 `FixtureLoop`，沒有正式啟動 CLI、真實影片 loader 或 CUDA 執行入口。`summed_ce` 使用裝置上的八類 logits／targets，指標另以 detached CPU logits 累積；CUDA 路徑尚未執行驗證。正式 GPU 版本仍須接入 CUDA RNG、worker loader、資源與啟動檢查，不能直接把這個 CPU schema 當成正式 run。

## 中斷一致性

同一份 checkpoint 包含當前模型、Adam、Python／NumPy／torch CPU RNG、cursor、更新數、訓練指標、完整歷史、最佳輪次／CE、早停計數與最佳模型權重。先寫同目錄暫存檔，flush／fsync 後 atomic replace，再同步目錄。

完整 validation 的成績與最佳權重一起提交，沒有獨立 best.pt／history.json 的先後不一致問題。代價是每份 checkpoint 多存一份最佳模型權重；正式接線前需將此空間及寫入成本納入資源檢查。

未完成 validation 時保留訓練輪末 checkpoint，不提交部分結果；明確恢復後從 validation 第一筆重新執行。暫停或失敗的記憶體物件不能直接繼續，必須新建物件並明確呼叫 resume，避免使用部分變動的狀態。報告應由 authoritative checkpoint 重新導出。

## 驗收結果

| 檢查 | 結果 |
| --- | --- |
| micro1累積16＋尾批8 | 與獨立實體批次16＋8的平均CE更新比較，參數最大允許絕對誤差3e-8，通過；不是對所有模型宣稱逐位等價 |
| train順序及validation覆蓋 | 每筆 ID／split核對；train24筆、validation10筆各遍歷一次 |
| 訓練中斷恢復 | 使用含torch／Python／NumPy隨機性的極小模型；恢復後模型、Adam、指標、歷史、最佳權重及三種CPU RNG與不中斷執行逐項完全相同 |
| validation中斷恢復 | 第3筆中斷，不產生history／best；恢復後重做全部10筆，最終狀態與不中斷執行相同 |
| 同分與早停 | 固定同分案例保留epoch1，epoch6時bad_epochs=5停止，不延長輪數 |
| 指標及checkpoint損壞 | 非法矩陣／CE、舊schema、不同plan、錯誤cursor／count、Adam設定、非有限權重及dtype不符均拒絕 |
| 非有限輸入／錯誤ID | 失敗後保留上一份磁碟checkpoint；拒絕直接重入失敗物件 |
| 原子存檔／選模提交失敗 | 注入replace前及validation commit前故障；舊checkpoint保持一致，可明確恢復並重新完整提交 |

8項 unittest 通過，最終執行約0.213秒（不含Python啟動）。沒有建立ResNet或載入ImageNet權重，沒有讀取真實影片或test，CUDA未初始化。只對極小模型做少量工程參數更新，checkpoint皆由TemporaryDirectory清理。64個基準及51個歷史受保護檔案SHA全部一致。

這些合成數字不是研究 validation／test 成績，也不能推論accuracy有提升。50幀RGB、整段取樣、160×120、無裁切／增強、標籤與資料split皆保持原樣。

[機器可讀驗收紀錄](eight_class_cpu_loop_qa_v1.json)

## 下一個有限任務

把既有真實資料 adapter、固定八類 ResNet 與 GPU 接入同一套迴圈，完成一次有明確樣本／更新上限的端到端預檢，並核對GPU RNG恢復及worker順序。完成後才進入單一baseline正式啟動判定；不重新比較模型、不開始多seed佇列。

原 [訓練規格](BASELINE_TRAIN_SPEC.md) JSON 的 implementation_pending／not_implemented 是定稿當時快照，依 SHA 契約保持不變；本頁及QA紀錄記載後續實作進度，不代表解除 training_authorized=false。

[上一輪CPU模型接線](CPU_MODEL_WIRING.md) · [資料驗收](BASELINE_ACCEPTANCE.md) · [研究工作台](../../../workspace/README.md)

2026-11-17後續：上述有限端到端預檢已完成，見[真實輸入與GPU預檢](GPU_READINESS.md)。本頁保留CPU輪次快照；正式全量啟動及一般CUDA續訓入口尚未接入。
