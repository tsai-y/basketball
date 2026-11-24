# ACA 架構缺口與放行條件

日期：2026-11-18。缺口盤點已定稿；作者精確架構尚未定稿。本頁沿用已保存逐層核對，核實現有model.py及modules.py與該證據的SHA一致，不重新建立模型。

## 核心缺口

| 項目 | 論文或既有證據 | 現有程式 | 放行所需證據／決定 |
| --- | --- | --- | --- |
| 插入與融合 | 每個bottleneck第一個1×1後、LSTA與TSCI並聯；見原文§2.1及Fig.1 | 16點，conv1/BN/ReLU後，兩路相加 | BN/ReLU相對位置、stride與殘差順序需精確定義 |
| Gate | 卷積與InstanceNorm；細節見既有Fig.2核對 | 三個C→C Conv3d，1³＋3³＋3³，dense、無bias | 作者groups／通道／算子表；不得靠參數接近推定depthwise |
| 圖式歧義 | 既有核對記錄Fig.2與Eq.1／4差異 | sigmoid接在IN/ReLU後；回乘完整G | 作者說明或明列採用哪個解讀 |
| 短期／動態核 | 兩時間分支有依據，具體值未核實 | r16、兩個k3、β0.25、K3、FC有bias | r、β、K、bias、padding及邊界處理 |
| TSCI | 三路空間／通道交互與平均；見原文§2.3 | NT折幀，三個獨立7×7 gate、BN後sigmoid | 作者維度、共享範圍、BN設定與初始化 |
| BN與batch | 作者報batch16、多GPU；未取得完整BN設定 | 骨幹與新TSCI BN全部凍結，micro1累積16 | 新BN是否訓練、SyncBN／每GPU batch；累積不等於BN統計等效 |
| 整體參數 | 作者報約26.8M；本地逐層核算見reference | 八類dense ACA 93,176,164 | 可追溯的差異解釋；不以總量湊數驗收 |
| 輸入與評估 | 精確crop、split清單、metric平均規則未取得 | 凍結50幀RGB全圖；7192/1547；macro-F1明定 | 作者transform／split或明列專案協定；不自行換切分 |

原文來源：[§2–3](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1471327/full)。詳細程式形狀及先前圖式核對：[既有重建規格](ACA_RECONSTRUCTION_SPEC.md)、[架構稽核](ACA_ARCHITECTURE_AUDIT.md)。

## 參數差異的定位

在既有16個窄通道插入點，C=64/128/256/512、blocks=3/4/6/3。dense gate合計Σblocks×55C²＝69,160,960參數；八類baseline為23,524,424，其他LSTA與TSCI為490,780，因此ACA合計93,176,164。

二類換八類只增加12,294，凍結BN也不改變總參數數量，均不能解釋此差距。既有DW3D與貼文預設也只是不同假設；接近論文數量不構成作者身分證明。這個算術定位已完成，不再用更多seed回答架構問題。

## 作者程式查核的停止範圍

2026-11-18重新查閱出版社頁，進行以下三個限定查詢：`site:github.com "1471327"`、`site:github.com "LSTA" "TSCI" basketball`、`"ACA-Net" basketball author code Zhang Xu`。此次回傳結果未核實作者模型程式庫；不推論全網不存在。

沿用2026-11-03補充包清單：內層34項為稿件、模板、圖片，當時未找到模型原碼；本輪未重新下載或執行補充包。[保存證據](results/aca_architecture_sources_v1.json)

不再做無限關鍵字搜尋。若獲得作者連結，驗證其作者／論文關聯、版本、授權及設定，再更新差異表；不直接執行陌生程式。

## 兩條明確的後續路線

| 路線 | 下一個有限工作 | 代價與邊界 |
| --- | --- | --- |
| 優先作者證據（建議） | 使用下方草稿請求原碼與關鍵設定；收到後核對 | 等待外部回覆；期間保留baseline，不啟動猜測版訓練 |
| 明列假設的重建 | 製作唯一候選的逐層規格、參數表、BN及共同協定供審閱 | 不能稱作者精確複刻；路線選擇本身不授權任意超參數或正式訓練 |

2026-11-18使用者已明確選擇「明列假設的重建」，授權提出唯一候選與共同協定供審閱。[提案已完成](ACA_RECONSTRUCTION_PROPOSAL.md)，尚未批准實作或訓練；作者詢問草稿保留，未寄送。

## 作者詢問草稿

未寄送，供使用者審閱或自行寄送。需要代寄時另有明確指示才可送出。

Subject: Request for ACA-Net implementation and experimental protocol (DOI 10.3389/fnbot.2024.1471327)

Dear ACA-Net authors,

I am reproducing your basketball action recognition study. Could you share the official implementation or clarify the following details?

1. The exact LSTA and TSCI layer definitions, insertion locations, and fusion operations, including convolution groups/channels, normalization, biases, and the values of r, beta, and K. Our literal dense interpretation has about 93.18M parameters with an eight-class head; we would like to understand the difference from the reported 26.8M without guessing architectural substitutions.
2. Backbone and newly added normalization behavior, per-GPU batch size or SyncBN, and pretrained/new-module initialization.
3. Basketball-51 split manifests/seed and exact temporal sampling, cropping, resizing, normalization, and augmentation transforms. For SpaceJam, please clarify the relationship between the 16-frame dataset description and the T=8 experimental input.
4. Early-stopping details, optimizer settings, evaluation sampling, and multiclass precision/recall/F1/AUC averaging conventions.

A versioned repository, configuration file, or model summary would be very helpful. Thank you for your time.

## 不能由本輪得出的結論

現有八類baseline已完成，只提供專案validation參考。它不驗證ACA架構，不使舊二類四組變成八類消融，也不證明來源資料或任務協定與作者完全相同。訓練、影片解碼、權重載入、test與新模型候選均未在本輪執行。
