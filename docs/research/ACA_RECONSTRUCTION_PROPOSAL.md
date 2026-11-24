# ACA 唯一重建候選與共同協定提案

後續更新：使用者已指示接續有限工程；[CPU及有限ACA真實GPU驗收通過](ACA_ENGINEERING_STATUS.md)。下文與JSON保留提案快照，正式訓練仍未啟動。

日期：2026-11-18。使用者已選擇「明列假設的重建」，本提案供審閱。範圍為Basketball-51八類及固定四組消融；未核實為作者精確版。這次只有規格與參數算術，尚未實作、建立模型或啟動訓練。

## 建議採用的唯一候選

採ResNet50＋LSTA／TSCI並聯。LSTA gate保留dense 1×1×1混合通道，兩個3×3×3改為depthwise（groups=C），其餘時間分支值維持既有r16、β0.25、K3。新TSCI BN正常訓練，骨幹BN繼續凍結。預算為**25,476,580參數**，其中可訓練25,423,460。

這個gate拓撲已有本地DW3D來源可供參考；本次選擇它的理由是保留通道混合與時空局部處理、降低dense gate成本，且不新增一批候選。它不是作者碼，也不是因接近26.8M就獲得認證。與舊DW3D相比，本提案定義八類共同協定、配對初始化及可訓練的新TSCI BN；舊二類結果與權重不沿用。

[機器可讀提案](specs/aca_reconstruction_proposal_v2.json)是文件規格，launch_enabled=false，不是可啟動的run plan。

## 論文依據與重建假設

| 項目 | 本提案 | 證據等級／原因 |
| --- | --- | --- |
| 骨幹與整體方向 | ResNet50、LSTA/TSCI並聯，全部16個bottleneck第一個1×1之後 | 拓撲有論文依據 |
| 具體位置 | conv1→BN→ReLU→模組→conv2；transition block在stride2前 | 重建假設，沿用既有骨幹接線 |
| Gate | 1³ dense；control與output的3³各groups=C | 資源與機制取捨，作者groups未核實 |
| Norm／激活 | IN無affine/running stats；control在IN/ReLU後sigmoid | 沿用既有圖解解讀，圖／公式歧義保留 |
| 短期／動態核 | r16、k3/k3、β0.25、K3，有bias、無BN1d | 固定專案假設；不移植TAM預設或搜尋湊參數 |
| TSCI | 三路mean/max、7×7卷積、BN、sigmoid、還原後平均 | 主拓撲有依據，精確排列與BN訓練規則為本提案定義 |
| BN | 骨幹凍結，新TSCI訓練；validation一律用BN已存統計 | 保留baseline相容性，避免新BN一直停在初始化統計 |
| 輸入／split | 沿用已凍結50幀RGB及7192/1547 | 專案控制比較；不是作者精確split/crop |

原文來源：[ACA-Net §2–3](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1471327/full)。既有逐層證據見[重建規格](ACA_RECONSTRUCTION_SPEC.md)。PyTorch的groups=C定義見[Conv3d](https://docs.pytorch.org/docs/2.11/generated/torch.nn.Conv3d.html)；訓練與推論統計見[BatchNorm2d](https://docs.pytorch.org/docs/2.11/generated/torch.nn.BatchNorm2d.html)。官方算子文件只支持實作語義，不支持作者採用這些設定。

## 逐層架構

輸入N×3×50×120×160。ResNet50 v1.5使用3/4/6/3 blocks；所有2D卷積跨時間共用，stride只降低空間。Stem為7×7 stride2→BN/ReLU→3×3 maxpool stride2；時間始終50。窄通道依序64/128/256/512。

每個block先計算X=ReLU(BN(conv1(input)))，插入M(X)，再conv2/BN/ReLU→conv3/BN→原identity/projection相加→ReLU。LSTA與TSCI看到同一個X；模組外不額外再加X。最後T/H/W平均得到2048維，接有bias的Linear(2048,8)，無dropout。

令B_k(U)=ReLU(IN(Conv3d_k(U)))，各層獨立、無bias、stride1、padding=k//2、dilation1；k1使用groups1，兩個k3各groups=C，IN eps1e-5、affine=False、track_running_stats=False。

```text
control = sigmoid(B3_control(X))
G       = X + B3_output(B1_value(X) * control)
P       = mean_HW(G)                                      # N,C,50
V       = sigmoid(Conv1d_h_C(ReLU(Conv1d_C_h(P))))        # h=C//16，兩層k3/pad1、有bias
Z       = G * V[...,None,None]
theta   = softmax_K(Linear12_3(ReLU(Linear50_12(P))))     # FC有bias，跨C共用
LSTA[t] = sum(theta[k] * Z[t+k-1], k=0..2)               # 兩端補0，不翻核、不重新正規化
```

採G回乘V以保留空間形狀，明列為公式歧義的重建解讀。control在實數下為[0.5,1)，浮點可能飽和到1；不把此gate解釋為能完全消除背景。動態核每片／每通道獨立，在該通道所有空間位置共用；不同block不共享FC權重。

TSCI先折成NT,C,H,W。三路排列為identity、(0,2,1,3)、(0,3,1,2)；各自在axis1做mean/max依序串接成2通道，Conv2d(2,1,7,pad3,bias=False)→BN→sigmoid，乘回該路原特徵。逆排列分別identity、(0,2,1,3)、(0,2,3,1)，三路平均後恢復N,C,T,H,W。每路、每block獨立參數，權重跨幀共用。

| 消融組 | M(X) | 額外殘差 |
| --- | --- | --- |
| baseline | X | 無 |
| LSTA | LSTA(X) | 只有LSTA內部已定義的X殘差 |
| TSCI | TSCI(X) | 無 |
| ACA | LSTA(X)+TSCI(X) | 不額外加X、不除以2 |

這四組是同一候選的模組移除消融，不是四個超參數版本。沒有容量匹配控制，因此若有提升，不能宣稱已完全排除參數量與融合尺度的影響。

## BN 與初始化

骨幹53個BN固定ImageNet affine與buffers；每次model.train()後只把骨幹BN設eval。LSTA IN在train與validation皆使用當片統計且無buffers。

TSCI的48個新BN使用eps1e-5、momentum0.1、affine=True、track_running_stats=True；初始化γ1、β0、mean0、var1、counter0。train更新affine與統計；validation使用已存統計且不得更新buffers。它們的統計來自單片折入batch的50幀與空間位置，不能視為16支獨立影片或作者多GPU的BN等效。未來工程驗證須特別確認這個train/eval差異；不在看過validation分數後切換當片統計。

所有組從同一份固定ImageNet V1開始。先按既有baseline8初始化的完整順序建立共同初始骨幹與八類head，再複製到各組；不能用已訓練的baseline last.pt初始化新組。新增模組不改變head初始值。

block索引0–15按layer1.0至layer4.2排序；LSTA用隔離CPU RNG seed+1000+index，TSCI用seed+2000+index。使用固定PyTorch2.11的Conv／Linear建構預設，直接建立目標groups，不先建dense再替換；LSTA建立順序value、control、output、短期兩層、動態兩層，TSCI為channel、height、width。隔離RNG結束後還原共同全域狀態。共有模組初始張量必須逐一相等，資料順序使用獨立generator。

## 參數預算

每個C的gate=C²+54C，short=6Ch+h+C（h=C//16），dynamic=651，TSCI=300。

| Stage | Blocks | C | Gate／block | Short／block | ACA模組stage合計 |
| --- | ---: | ---: | ---: | ---: | ---: |
| layer1 | 3 | 64 | 7,552 | 1,604 | 30,321 |
| layer2 | 4 | 128 | 23,296 | 6,280 | 122,108 |
| layer3 | 6 | 256 | 79,360 | 24,848 | 630,954 |
| layer4 | 3 | 512 | 289,792 | 98,848 | 1,168,773 |

| 組別 | 總參數 | 可訓練參數 |
| --- | ---: | ---: |
| baseline | 23,524,424 | 23,471,304 |
| LSTA | 25,471,780 | 25,418,660 |
| TSCI | 23,529,224 | 23,476,104 |
| ACA | 25,476,580 | 25,423,460 |

各組僅凍結53,120個骨幹BN affine；TSCI新BN affine可訓練，與舊「全部BN凍結」計數不同。ACA比論文約26.8M少約1.32M；保留差異，不新增層湊數。以上是算術預算，實作後必須重數。

## 四組共同實驗協定

| 項目 | 固定提案 |
| --- | --- |
| 資料 | 同一凍結契約、7192 train／1547 validation、相同排序與SHA；不改標籤或排除疑義片 |
| 輸入 | 原生整段端點均勻50幀RGB、160×120、ImageNet正規化；無crop、augmentation、虛擬25Hz |
| 目標 | 八類class_id，未加權CE；無label smoothing、oversampling或二類輔助loss |
| 第一個seed | 20261024；後續預定20261025、20261026，單seed比較完成後另行放行 |
| 初始化 | 固定ImageNet V1、同骨幹與head、共有模組配對；不用舊任務checkpoint |
| 更新 | Adam lr1e-4，betas(.9,.999)、eps1e-8、wd0、amsgrad/foreach/fused=False，無scheduler |
| 批次 | micro1／effective16；449個完整批＋尾批8，共450更新；按實際批量平均梯度，clip norm1一次後step |
| 停止 | 最多30輪、patience5、min_delta0，最低完整八類validation CE；相同CE留較早輪並計bad |
| 評估 | 1547支完整validation，每片一次；accuracy、macro-F1（8類zero_division0）、混淆矩陣及recall |
| 二類診斷 | 八類argmax後映射Make/Miss，不能以機率加總替換，也不用它選模 |
| 數值與runtime | 沿用baseline紀錄：PyTorch2.11/CUDA12.8、FP32、AMP/TF32off、deterministic、2個spawn workers；版本差異先核對 |
| 保存 | 每25更新、train輪末與完整validation保存模型／Adam／RNG／BN buffers／history／best_model；每run獨立plan與attempt |
| 中斷 | 明確要求後才恢復同run；不完整validation全部重做；不自動續訓、延長或改規則 |
| test | 本階段關閉；全專案歷史test接觸如實揭露 |

完整細節由機器可讀提案繼承[既有baseline訓練契約](specs/eight_class_baseline_train_v1.json)中的data、optimizer、batch、metrics、stopping、runtime欄位；不沿用其歷史implementation_pending文字作目前狀態。

## 現有 baseline 是否需要重跑

**提案目標是沿用，現在不重跑；實作驗收前仍為有條件相容。** 新TSCI BN只存在於TSCI／ACA，沒有改變baseline的骨幹BN。沿用須核對：有序資料／輸入SHA一致、初始骨幹與head逐張量相同、baseline算子順序與模式不變、共同數值迴圈／loss／optimizer／尾批／選模一致、runtime可比。

現有seed24初始化器與正式plan只允許baseline，不得以替換variant字串繞過。未來應隔離擴充模組factory與模式處理，保留舊入口與run；若共同算術或初始化改變到無法證明相容，先回報原因，才決定是否重跑。其他兩個seed也需要明示新契約，不靜默放寬seed24限制。

## 審閱後的有限下一步

先實作且只驗證：四組meta shape／參數數量、共有初始化相等、獨立動態卷積參考與梯度、BN train/eval及validation buffer不變、baseline相容性。通過後才做固定24 train＋8 validation的有限GPU接線／尾批／恢復核對；臨時權重不當正式成績。已通過的資料全量驗收不重跑。

預檢須估計定稿模型顯存與每輪成本；資源不足或數值失敗就停止回報，不自動改精度、解析度、groups或BN。正式訓練與test皆需後續明確啟動指示。單seed順序：沿用合格baseline→LSTA→TSCI→ACA；不因分數不升就修改候選。三seed與SpaceJam屬後段，現在不排程。

這次請審閱的兩個主要取捨是：**保留dense1³＋兩個depthwise3³的gate；凍結骨幹BN而訓練新TSCI BN。** 其餘值以可追溯、減少變因及沿用資料基礎為原則固定。
