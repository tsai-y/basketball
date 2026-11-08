# ACA dense 既有重建規格 v1

日期：2026-11-08。規格識別：`aca_dense_reference_v1`。

**這份文件固定「現有程式究竟做了什麼」，不新增模型，也不把現有版本升格為作者精確版。**
對應已完成的 Make/Miss 四組單 seed 消融；93M dense 版本是保存及核對的參考，不因本規格而決定再次訓練。訓練與每小時排程維持暫停。

- [機器可讀規格與逐層參數](specs/aca_dense_reference_v1.json)：文件資料，不是 trainer 可執行設定。
- [本輪驗收快照](results/aca_reconstruction_spec_qa_v1.json)：整數核算、AST、來源 SHA 與既有稽核對照。
- [研究決策與已完成成績](ACA_REPRODUCTION_CHECKPOINT.md)：保留四組結果及其限制。

## 1. 證據界線與固定決定

[原論文 §2.1–2.3、Fig.1–3、Eq.1–8](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1471327/full)提供拓撲依據。已重新目視核對保存的三張圖；補充稿件文字也明載 InstanceNorm。以下「固定」均指本專案既有版本；沒有作者證據的值仍是重建假設。

| 項目 | 證據與判定 | 既有版本固定行為 |
| --- | --- | --- |
| 插入範圍 | §2.1 明載每個 bottleneck 第一個 1×1 conv 後 | ResNet50 全部 16 個 block；conv1/BN/ReLU 後、conv2 前；BN/ReLU 相對位置仍是專案決定 |
| 並聯融合 | Fig.1 為相加 | `LSTA(X)+TSCI(X)`；不除以2，不額外加一次 X |
| Gate 與 Norm | §2.2 有 InstanceNorm；Fig.2 有 1³、3³、輸出3³及 Norm/ReLU | 三層均 C→C、groups=1、bias=False；IN 無 affine／running stats；這些具體旗標未由作者碼確認 |
| Gate sigmoid | Fig.2 有，Eq.1 未列 | 採圖中的 sigmoid；放在 control 的 IN/ReLU 之後 |
| 短期回乘 | Eq.4 的 P 已池化，與完整空間圖有歧義 | 採 `G*V` 的空間 broadcast，維持 N,C,T,H,W；不假稱這是已核實作者碼 |
| TSCI | Fig.3 支持三支路、7×7、BN及平均 | 三支路獨立參數；inverse permutation 後平均；BN 凍結是專案決定 |
| 未核實的數值 | r、β、K、groups、bias、padding 等缺少完整作者設定 | r16、β0.25、兩個短期 k3、動態 K3；不依參數總數改成 depthwise |

論文圖中的通道縮放、排列與融合有依據，並不代表每個預設值都已核實。舊報告將「插入位置」概括列為待核對；本輪收斂為：**每個 bottleneck 第一個1×1後有文字依據，精確 BN/ReLU/stride 位置仍未核實。**

## 2. 骨幹與 16 個插入點

來源：[model.py](../../development/aca_net_v1/model.py)、[modules.py](../../development/aca_net_v1/modules.py)。程式指紋見 JSON；文件前來源 commit 為 `0f227fd`。

- 輸入 `N×3×50×120×160`，RGB；所有幀使用同一組2D骨幹權重，framewise 展成 `N*T` 執行。
- torchvision ResNet50 v1.5，stage block 數 3/4/6/3，窄通道 64/128/256/512，輸出通道為4C。階段下採樣在首 block 的 **conv2**；此為 torchvision 選擇，不是作者版本認證。
- Stem：7×7 conv stride2 pad3 → BN/ReLU → 3×3 maxpool stride2 pad1。空間依序120×160 →60×80 →30×40，T始終50。
- 每個 block：保留 identity；conv1/BN/ReLU → 模組 → conv2/BN/ReLU → conv3/BN → 與原 identity/projection 相加 → ReLU。
- 最後對 T/H/W 取平均得到2048維，接有 bias 的 `Linear(2048,2)`，輸出 logits；沒有 dropout。八類頭只作參數核算，沒有建立或訓練八類新模型。

下表的 shape 為 `C×T×H×W`，省略 N；模組輸入與輸出 shape 相同。transition block 的模組位於 stride2 **之前**。

| 位置 | 模組 shape | block 最後輸出 shape | conv2 stride | 骨幹參數 | ACA 模組參數 |
| --- | --- | --- | ---: | ---: | ---: |
| layer1.0 | 64×50×30×40 | 256×50×30×40 | 1 | 75,008 | 227,835 |
| layer1.1 | 64×50×30×40 | 256×50×30×40 | 1 | 70,400 | 227,835 |
| layer1.2 | 64×50×30×40 | 256×50×30×40 | 1 | 70,400 | 227,835 |
| layer2.0 | 128×50×30×40 | 512×50×15×20 | 2 | 379,392 | 908,351 |
| layer2.1 | 128×50×15×20 | 512×50×15×20 | 1 | 280,064 | 908,351 |
| layer2.2 | 128×50×15×20 | 512×50×15×20 | 1 | 280,064 | 908,351 |
| layer2.3 | 128×50×15×20 | 512×50×15×20 | 1 | 280,064 | 908,351 |
| layer3.0 | 256×50×15×20 | 1024×50×8×10 | 2 | 1,512,448 | 3,630,279 |
| layer3.1 | 256×50×8×10 | 1024×50×8×10 | 1 | 1,117,184 | 3,630,279 |
| layer3.2 | 256×50×8×10 | 1024×50×8×10 | 1 | 1,117,184 | 3,630,279 |
| layer3.3 | 256×50×8×10 | 1024×50×8×10 | 1 | 1,117,184 | 3,630,279 |
| layer3.4 | 256×50×8×10 | 1024×50×8×10 | 1 | 1,117,184 | 3,630,279 |
| layer3.5 | 256×50×8×10 | 1024×50×8×10 | 1 | 1,117,184 | 3,630,279 |
| layer4.0 | 512×50×8×10 | 2048×50×4×5 | 2 | 6,039,552 | 14,517,719 |
| layer4.1 | 512×50×4×5 | 2048×50×4×5 | 1 | 4,462,592 | 14,517,719 |
| layer4.2 | 512×50×4×5 | 2048×50×4×5 | 1 | 4,462,592 | 14,517,719 |

表中骨幹含原 block 的 BN affine 與必要 projection，不含 ACA；Stem另有9,536參數。形狀以卷積尺寸公式核算，未做本輪 forward。

## 3. LSTA 的確切算子

令 `B_k(U)=ReLU(IN(Conv3d_k(U)))`。三個 B 各自擁有權重，stride1、各維 padding=k//2，dense groups1，bias=False。IN為每個樣本/通道於 T/H/W 計算統計，eps=1e-5、affine=False、track_running_stats=False；train/eval均使用當片統計。

```text
control = sigmoid(B3_control(X))
G       = X + B3_output(B1_value(X) * control)       # N,C,T,H,W
P       = mean_HW(G)                              # N,C,T
V       = sigmoid(Conv1d_h_to_C(ReLU(Conv1d_C_to_h(P))))
Z       = G * V[:,:,:,None,None]                   # N,C,T,H,W
theta   = softmax_K(FC12_to_3(ReLU(FC50_to_12(P)))) # N,C,3
Y[t]    = sum(k=0..2, theta[k] * Z[t+k-1])         # 越界 Z=0
```

- 短期 h=`max(1,C//16)`，兩層 Conv1d 都有 bias、kernel3、padding1，中間 ReLU，末端 sigmoid；沒有 BN1d。
- 長期 hidden=`max(1,int(50*0.25))=12`；FC 50→12→3 均有 bias。同一 block 的 FC 權重跨 C 共用，每個通道因輸入不同而生成各自的核；不同 block 不共用權重。不是每個通道另有一套 FC。
- theta沿K做 softmax；每影片、每通道一個核，該通道各空間位置共用。運算是時間 cross-correlation，不翻轉 kernel；時間兩端各補1個0，邊界不重新正規化。這些邊界細節是專案重建選擇。
- sigmoid 接在 ReLU 後，因此理想實數下 control∈[0.5,1)，不是任意[0,1]抑制；浮點可能飽和到1。這裡記錄現況，不藉文件改寫 gate。

這裡的 LSTA InstanceNorm 與先前 TANet 的 L/G BatchNorm 議題不同；不沿用 TANet 的推論規則。

## 4. TSCI、融合與 BN

將 X 折成 `NT,C,H,W`。三個獨立 SpatialGate 都先沿當下 axis1 做 mean/max concat，得到2通道，再做無bias的7×7 Conv2d(2→1,pad3) → BatchNorm2d(1) → sigmoid，回乘**各支路原特徵**。

| 支路 | gate 前形狀 | permutation / inverse | 池化後形狀 |
| --- | --- | --- | --- |
| C | NT,C,H,W | identity / identity | NT,2,H,W |
| H | NT,H,C,W | 0,2,1,3 / 0,2,1,3 | NT,2,C,W |
| W | NT,W,C,H | 0,3,1,2 / 0,2,3,1 | NT,2,C,H |

三支路 inverse 後平均 `/3`，恢復 N,C,T,H,W。gate權重跨幀共用，跨block獨立；沒有顯式時間卷積。在目前BN凍結規則下，也沒有訓練批次統計造成的跨幀耦合。

ACAModule最終輸出 `LSTA(X)+TSCI(X)`。LSTA內有一次 X 殘差，外層原ResNet block也保留identity；不在模組外再額外加 X，也不把兩支路平均。

BN規則來源：[baseline_preflight.py](../../development/aca_net_v1/baseline_preflight.py)、[四組初始化](../../development/aca_four_arm_v1/experiment.py)、[共同更新](../../development/aca_four_arm_v1/engine.py)。

| 類型 | 個數 | 訓練與validation | 參數／buffers |
| --- | ---: | --- | --- |
| 骨幹BN2d | 53 | eval | ImageNet affine及統計固定；53,120個affine參數不更新 |
| 新TSCI BN2d | 48 | 自初始化起eval | γ1、β0、mean0、var1、counter0均不更新；96個affine固定 |
| LSTA IN3d | 48 | 均用當片統計 | 無affine、無running buffers；不受freeze_batchnorm影響 |

新BN仍有 `1/sqrt(1+eps)` 因子，不是數值上完全identity。BN預設eps1e-5、momentum0.1；凍結時momentum不參與更新。凍結BN affine不會切斷經過BN的輸入梯度。沒有SyncBN；不能宣稱累積physical1就等同作者physical16的BN行為。**包含新BN在內全部凍結，是專案規則，並未核實為論文要求。**

固定初始化為 ImageNet V1，SHA見JSON；骨幹/head及重疊LSTA/TSCI使用既有配對初始化。這次沒有載入任何權重，也沒有用舊候選權重初始化新任務。

## 5. 可手算的參數預算

計數含 frozen affine，不含 buffers、gradient或Adam狀態。每個窄通道 C 的完整模組：

```text
gate    = C² + 27C² + 27C²                  = 55C²
short   = (3Ch+h) + (3hC+C), h=C//16        = 6Ch+h+C
dynamic = (50×12+12) + (12×3+3)             = 651
TSCI    = 3×(2×1×7×7 + 2×1)                = 300
```

| Stage | C / h | Blocks | Gate / block | Short / block | Dynamic / block | TSCI / block | 模組 stage合計 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| layer1 | 64 / 4 | 3 | 225,280 | 1,604 | 651 | 300 | 683,505 |
| layer2 | 128 / 8 | 4 | 901,120 | 6,280 | 651 | 300 | 3,633,404 |
| layer3 | 256 / 16 | 6 | 3,604,480 | 24,848 | 651 | 300 | 21,781,674 |
| layer4 | 512 / 32 | 3 | 14,417,920 | 98,848 | 651 | 300 | 43,553,157 |
| 全部 | — | 16 | 69,160,960合計 | 475,564合計 | 10,416合計 | 4,800合計 | **69,651,740** |

骨幹：Stem9,536 + stage1 215,808 + stage2 1,219,584 + stage3 7,098,368 + stage4 14,964,736 = **23,508,032**（無head）。
以 I 為 block 輸入通道，conv權重與BN affine合計為 `IC+13C²+12C`；若需projection另加 `4IC+8C`。分類頭參數=`2049×類別數`。

| 既有分支 | 二類總參數 | 二類可訓練參數 | 八類總參數（僅算術） |
| --- | ---: | ---: | ---: |
| Baseline | 23,512,130 | 23,459,010 | 23,524,424 |
| LSTA | 93,159,070 | 93,105,950 | 93,171,364 |
| TSCI | 23,516,930 | 23,463,714 | 23,529,224 |
| ACA | **93,163,870** | **93,110,654** | **93,176,164** |

二類換八類只多12,294參數，無法解釋與論文約26.8M的差距。groups、通道、插入範圍等會影響參數，但缺少作者證據時，不用接近26.8M作為「已複刻」的判準，也不新增候選去湊數。

## 6. 本輪驗收與下一關

以純Python標準庫核對：來源SHA、dataclass AST、16個位置及尺寸、逐block/stage計數、八個既有meta稽核總數，以及十二組已存readiness參數數量。沒有匯入torch、建立模型、讀影片、推論、更新optimizer或執行test；原權重／plan／程式不改。

可重做唯讀算術檢查：

```bash
python3 scripts/audit_aca_reconstruction_spec.py
```

這不是新的forward、梯度或工程readiness驗收；舊readiness的有效範圍仍是舊程式與原契約。任何未來實作更動需另記新版本，不能覆寫既有run。

| 關卡 | 本輪判定 |
| --- | --- |
| G1「明列假設的既有重建規格」 | 已定稿及算術核對；不等同確認其符合作者參數預算 |
| G1「作者精確實作」 | 仍未核實；groups、具體超參數、BN規則及參數量差距仍待作者證據 |
| G2 原論文任務／資料／評估契約 | 下一步先做八類標籤、資料清單、split、取幀/crop及評估定義的唯讀對照；本輪不建立新split、不改標籤 |
| G3–G5 新工程、正式比較、多seed／SpaceJam | 不自動啟動；先有明確任務契約與待回答問題 |

下一個可驗收產物是 **G2任務契約表**：哪些原始欄位可用、哪些與論文不同、哪些證據仍缺；不是立即再訓練一個93M模型。Make/Miss已完成結果保留為獨立專案探索，不冒充原八類任務成績。整個專案已有歷史test接觸，不能因整理規格就稱為全新未見holdout。
