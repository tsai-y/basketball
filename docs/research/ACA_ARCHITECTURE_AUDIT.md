# ACA 架構核對結果（2026-11-03）

目前狀態（2026-11-08）：工程預檢及後續單seed四組已完成，訓練/排程已暫停；作者精確設定仍未核實。最新決策見 [複刻核對與研究決策](ACA_REPRODUCTION_CHECKPOINT.md)。以下保留11/03架構核對時的結論、草案與時點界線。

本輪已完成公開來源、圖與公式、既有實作及使用者 TAM→ACA 提案的核對。
結論：核心拓撲相符，但作者精確設定仍無法確認；這輪不改模型、不訓練、不執行 test。
本分支不再只是一份待辦入口。既有實驗狀態見 [ACA 實驗索引](aca_experiments.md)。

## 結論與接續決定

1. 保留既有 dense 候選作「依圖重建的專案參考」，不宣稱作者原版；DW3D 明確保留為效率假設。
2. 不採用貼文程式直接替換舊模型：它同時改了 gating、r、β、卷積、bias 與 BN，不能解釋為只新增兩個模組。
3. 下一個工程步驟是建立同協定的 baseline／LSTA-only／TSCI-only／ACA 四組設定與預檢，而非再新增架構或直接開跑十二組訓練。
4. 四組從全新配對 ImageNet 初始化開始；舊五輪→十五輪延長結果及高解析度結果保留為歷史，不拿來當新的受控四組。
5. SpaceJam 暫緩。既有 test 使用歷史保留，本輪結果與後續架構決定不依 test 成績調整。

此處 dense 是為維持既有候選與明確依圖假設的連續性，不是已證實優於 DW3D 或最接近作者程式。
作者精確復刻仍需可靠實作／設定證據；現有證據足以規劃專案受控對照，無需靠湊參數量推定原設定。

## 公開來源與補充檔的新證據

已視覺核對 [原論文 PDF](https://public-pages-files-2025.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1471327/pdf) 的印刷頁 3–6，另核對頁 8 的設定。
Fig.1 支持 conv1 後插入、LSTA/TSCI 並聯求和；Fig.2 支持 gating、空間平均、短期加權與動態時間核；Fig.3 支持 C/H/W 三路、7×7 Conv2D、BN/Sigmoid、逆置換後平均。

本次由 [Europe PMC 官方補充檔服務](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11461453/supplementaryFiles) 取得原文標示的 `Data_Sheet_1.ZIP`。
內層 **34 個項目**為 LaTeX 稿件、模板、參考文獻、logo、圖表與示例 PDF，沒有找到模型程式。
只讀取目錄及兩份 TeX 文字，沒有編譯或執行；不能把它當成「作者程式已取得」，也不能據此宣稱作者從未在別處公開程式。
搜尋到的語者辨識、雲層移除同名 ACA-Net 與本論文無關，未當作依據。

## 結構對照

| 項目 | 證據定位 | 既有 dense／DW3D | 貼文預設 | 判定 |
| --- | --- | --- | --- | --- |
| 插入與融合 | Fig.1 | 16 bottleneck，conv1/BN/ReLU 後，LSTA+TSCI | 同一主拓撲 | 大方向一致；BN/ReLU、stride 的精確位置仍屬重建選擇 |
| Gating | Fig.2、Eq.1 | dense 三個卷積；DW3D 只替兩個 3³ | dense 1³ + depthwise 兩個 3³ | 分組方式無作者實作佐證 |
| Gate Sigmoid | Fig.2／Eq.1 不完全一致 | 依圖採用 | 採用 | 明列圖／公式差異，不偷偷刪除 |
| 短期權重 | Fig.2、Eq.3 | r16、k3/k3、有 bias、沒有 BN | r4、k3/k1、無 bias、有 BN | 圖未列短期 BN；r/kernel/bias 仍待核實 |
| 動態核 | Fig.2、Eq.5 | 50→12→3，有 bias、沒有 BN | 50→100→3，無 bias、通道 BN | β/K/bias/BN 未由作者程式確認 |
| 加權的空間特徵 | Eq.4 與 Fig.2 形狀 | G×V | G×V | Eq.4 的 P 已壓空間；採 G 保留空間是明列的形狀解讀 |
| 時間卷積算子 | Eq.6 | unfold/einsum | grouped conv2d | 合成數值等價，算子包裝不是目前差異來源 |
| TSCI | Fig.3 | 三路逐幀共享 2D gates，還原後平均 | 同方向 | 本身無跨時間卷積；訓練 BN 另有統計範圍問題 |

表內既有設定是程式實測，未指定設定不能自動以 TAM 預設補成作者 ACA 規格。
候選訓練會凍結所有 BN（連新 TSCI 的 affine/buffers）；這是專案的小批次選擇，不能說是論文要求。
累積至 effective batch16 也不等於 physical batch16 的 BN 統計。

## 參數量：統一八類分類頭後比較

以下是 meta device 的總參數核算，包含可能被凍結的參數；不是可訓練參數、GPU 效能或分類成績。
與論文八類口徑一致後，結果仍不一致；論文「約 26.8M」不能用作唯一辨識條件。

| 實作／診斷版本 | 八類總參數 | 其中 gating |
| --- | ---: | ---: |
| ResNet50 | 23,524,424 | 0 |
| LSTA-only dense | 93,171,364 | 69,160,960 |
| TSCI-only | 23,529,224 | 0 |
| 既有 dense ACA | 93,176,164 | 69,160,960 |
| 既有 DW3D | 25,476,580 | 1,461,376 |
| 所有 gate 都 depthwise（診斷） | 24,222,884 | 207,680 |
| 貼文預設（僅核對） | 26,342,312 | 1,461,376 |

ResNet50 的 16 個窄通道插入點分別為 C=64/128/256/512、blocks=3/4/6/3。
以無 bias、IN 無 affine 的既有候選計算：

- dense gating：Σ blocks×55C² = **69,160,960**。
- dense 1³ + depthwise 兩個 3³：Σ blocks×(C²+54C) = **1,461,376**。
- 所有 gate 都 depthwise：Σ blocks×55C = **207,680**。

所以 **「所有 gate 都 depthwise」與「只改兩個 3³ 的 DW3D」不是同一版本**。
貼文與 DW3D 的 gating 相同，但短期與動態核設定不同，因此總量又不同。
八類改二類只少 12,294 個參數，不能解釋 dense 與約 26.8M 的巨大差異；凍結 BN 也不能改變總參數量。
所有報表用固定候選核算，沒有搜尋額外層或超參數湊 26.8M。

## TAM Global BN：可執行，但不等價

[固定作者 TAM 原碼](https://github.com/liu-zhy/temporal-adaptive-module/blob/5fcb46fe671ead2f7958d8ffc3321a0e274d6491/ops/temporal_module.py) 先把 P 摺成 `[N*C,T]`，Linear 後是 `[N*C,βT]`，用 `BatchNorm1d(βT)`。
貼文對 `[N,C,βT]` 直接使用 `BatchNorm1d(C)`，會對每個 C 在 N×βT 上取統計。
它不是 shape error，但不能稱為「直接沿用官方 G 分支」。

用相同 FC 權重、初始 BN affine、合成輸入 `[2,4,5]` 比較 train-mode BN，輸出動態核最大絕對差約 **0.157725**。
這是算子差異，不是準確率差 15.77 個百分點，也沒有證明哪種 BN 在真實資料較好。
grouped conv2d 與既有 einsum 的合成最大絕對差僅 **2.22e-16**；獨立逐影片 conv3d 與跨影片梯度隔離測試亦通過。

另依既有圖解實作，`Sigmoid(ReLU(...))` 的控制權重範圍是 `[0.5,1)`，不是能任意接近 0 的 gate。
這不是自動刪格、定位球或保證背景抑制；本輪沒有因此更改圖解實作。

## 實驗協定的界線

[論文 §3.3](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1471327/full) 與本專案的類別、split、輸入前處理、BN、batch、停止細節並未全部等同。
因此目標固定為 Make/Miss 專案上的 ACA 候選對照，不能把已有八類論文成績直接當二類目標的基準。
TANet L 當片統計屬另一個探索性改良，本輪不移植到沒有 L-BN 的既有 ACA 候選。

下一步先準備 [四組共同協定草案](results/aca_controlled_comparison_draft_v1.json)，使用既有 dense 設定，三個 seed 預先固定。
共同 trainer、配對初始化、尾批平均、checkpoint/resume、可訓練範圍與十二組完整輸入 GPU 預檢已完成；最新狀態見上方連結。
不得將已有 dense／DW3D／baseline 的不同 run 直接組成公平消融；架構核對階段未啟動工程預檢或正式訓練；預檢階段當時尚未正式訓練；後續啟動與暫停歷史見最新狀態連結。

## 驗收與可追溯性

- [核算與算子結果](results/aca_architecture_audit_v1.json)：7 個固定版本×二／八類，共 14 次完整尺寸 meta shape/count。
- [來源、補充檔目錄與保護檔 SHA](results/aca_architecture_sources_v1.json)：包含論文、補充包、使用者附件、固定 TAM 與既有 89 個 ACA 來源檔。
- 5 項新核對測試通過；既有 89 個 ACA 來源檔 SHA 全部不變。工具本身另記來源 SHA。
- CPU 合成算子及小張量梯度檢查；影片開啟、checkpoint 載入、optimizer updates、GPU 工作均為 0。
- 原 PDF、LaTeX、ZIP 與取用證據僅保存在 `development/aca_architecture_audit_v1/artifacts/evidence_20261103/`，不納入 Git。

本輪只完成架構核對與共同協定草案，未產生新 validation/test 分數，也未宣稱找到作者精確設定。
