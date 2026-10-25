# ACA-Net 論文復刻 v1：架構與可重現性關卡

狀態：模型原型、50幀資料管線、ImageNet初始化、正式baseline入口／續訓、
零更新GPU就緒檢查與完整ACA候選GPU前向／反向均完成；正式訓練尚未開始。
正式baseline可待授權後啟動；精確ACA復刻仍受參數量差異阻擋。
分支 `codex/aca-net-reproduction`。
目標：先建立 ResNet50 baseline、LSTA、TSCI 與完整 ACA 的可審核實作，
再於專案固定 split 上比較完整影片 Make／Miss。90% 為研究目標。

## 來源與定位

Zhang, Y., Zhang, F., Zhou, Y., and Xu, X. (2024).
*ACA-Net: adaptive context-aware network for basketball action recognition*.
DOI: [10.3389/fnbot.2024.1471327](https://doi.org/10.3389/fnbot.2024.1471327)。
[出版社全文](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1471327/full)，
[PMC 含公式全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11461453/)。

2026-10-25 核對 §2、Figures 1–3、§3.3、Tables 3–4。
標題／DOI／GitHub 搜尋未找到可驗證的作者官方實作連結；不能據此宣稱作者沒有公開程式。
PMC 列出 `Data_Sheet_1.ZIP`，本次下載返回網頁驗證頁；另試出版社舊格式路徑為 404。
**補充 ZIP 內容仍未核實**，也未把搜尋到的同名雲層移除模型當成本文程式。

此實作是依公開圖與公式重建的候選。未通過作者程式／參數核對前，
不能標示為精確復現，也不能以 synthetic smoke 宣稱分類效能。

## 已核對的結構

| 元件 | 論文位置 | 此處對應 |
| --- | --- | --- |
| ResNet50，16 個 bottleneck | §2.1 / Figure 1 | `model.VideoResNet50` |
| 第一個 1×1 Conv2D 後插 ACA | Figure 1 | `VideoBottleneck` 的 conv1/BN/ReLU 後 |
| LSTA、TSCI 平行，再相加 | Figure 1 右下 | `ACAModule` |
| 3D gating、短期權重、動態時間卷積 | §2.2 / Figure 2 / Eqs. 1–6 | `LSTA` |
| C/H/W 三方向池化與 7×7 Conv2D | §2.3 / Figure 3 / Eqs. 7–8 | `TSCI` |
| 三個 TSCI 輸出平均 | Eq. 8 | 還原維度後平均 |

輸入 `[N,3,T,H,W]`；ResNet 的 2D 層按影格共用權重；模組內保留時間維度。
末端將時間和空間平均成 2048 維，再經分類層輸出 logits。
支援 `baseline / lsta / tsci / aca` 四組，以及 8 類或 2 類分類頭。

## 必須公開的重建假設

1. Figure 2 的 gating 有 Sigmoid，Eq. 1 沒寫：預設依圖加入；`gate_sigmoid` 明確記錄。
2. Eq. 4 的 P 已為 `[N,C,T]`，V 卻擴為五維：以 G 乘 broadcast V，保留 Eq. 6 所需空間。
3. 未指定 r、短期 Conv1D kernel、K、β：候選固定 `16 / 3 / 3 / 0.25`。
4. Feed-forward 依 Figure 2 的 `[N,C,T] → [N,C,βT] → [N,C,K]`，
   以共用兩層 Linear 沿每個 channel 的 T 運算，Softmax 沿 K。
5. InstanceNorm 的 affine／running stats、conv bias、temporal padding 未交代：
   本版 IN 無 affine、無 running stats，3D conv 無 bias，時間使用對稱 zero padding。
6. ResNet50 使用 torchvision v1.5 stride 位置；BN/ReLU 相對 ACA 的插入位置、
   最後影片 pooling 均需作者程式佐證。
7. **參數量是正式ACA復刻訓練前的阻擋關卡（不阻擋獨立baseline）**：依文字使用 dense 3D conv 時，
   本版核算為 93.18M，與論文約 26.8M 不相符。差異可能來自未公開設定、
   本重建的解讀或文中規格，尚不能判定唯一原因。預設明確為 `gate_groups=dense`；
   `depthwise` 僅供後續假設診斷，不能改用它後宣稱已還原原方法。

## 論文協定與專案協定

| 項目 | 論文報告 | 本專案下一階段 |
| --- | --- | --- |
| 標籤 | Basketball-51 八類 | Make／Miss 二類；八類頭保留作架構對應 |
| 切分 | 80/10/10，未提供 split IDs／group 規則 | 重用既有清理後固定 split |
| 影像 | 50 幀、120×160 | 固定均勻50幀、全畫面縮放；詳見POLICY，與作者未公開前處理有差異 |
| 初始化 | ImageNet ResNet50，checkpoint 未指定 | 固定torchvision ImageNet V1，完整SHA與載入覆蓋率已核對 |
| 訓練 | Adam、lr 1e-4、batch 16、最多30輪、validation loss 早停 | baseline已固定設定；ACA待結構核對 |
| 成績 | 八類 accuracy 92.05% | 不能直接換算成本專案二類泛化成績 |

不重抽資料切分。原始資料從 `/home/tsai/projects/basketball` 唯讀讀取；
此 worktree 不包含 ignored 原片、舊特徵快取或模型 checkpoint。
模型 synthetic smoke 使用合成 tensor；新資料管線另提供有界的 4 train＋4 validation 驗收。
未來 output 限此模組 `artifacts/` 的新 run；不寫入 YOLO 或歷史 `runs/`。

## 執行驗收

在此 worktree 根目錄，沿用 `yolo11` 環境：

```bash
/home/tsai/miniconda3/envs/yolo11/bin/python -B -m unittest discover \
  -s development/aca_net_v1/tests -p 'test_*.py' -v
/home/tsai/miniconda3/envs/yolo11/bin/python -B -m development.aca_net_v1.smoke --synthetic
```

Smoke 在 CPU 驗證小通道 ACA 的真實 forward/backward、完整 ResNet50 baseline 的
小輸入 backward，並用 meta device 核算完整 50×120×160 形狀與參數。
**meta 不是完整 ACA 的數值推論、GPU 吞吐量或顯存實測。**
此synthetic smoke無 optimizer、無權重下載、無檔案輸出。
正式baseline CLI及其獨立啟動開關詳見下方與 [TRAINING_RUNBOOK.md](TRAINING_RUNBOOK.md)。

`load_imagenet_state` 要求完整torchvision ResNet50權重與統計量，保留來源1000類head供形狀核對，
轉移時保留新分類頭與新模組初始化；來源若只缺舊格式BN計數器，則明確補零並回報。
其他missing／unexpected key、shape／dtype或非有限值均拒絕。
單元測試中的隨機 ResNet state 只驗證載入機制，不代表已載入 ImageNet。

## 本次驗收紀錄（2026-10-25）

環境：Python 3.10.20、PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128；CPU。
已執行上述 unit tests 與 synthetic smoke。實際驗收重點包括：

- 每影片獨立的動態卷積，與逐影片 `conv3d(groups=C)` 數值相符；autograd gradcheck 通過。
- TSCI 三方向還原、逐幀語意，以及 LSTA／TSCI 平行求和通過。
- LSTA gating／短期分支／動態核生成均收到有限梯度。
- ResNet50 baseline 與原生 torchvision 逐幀 logits 平均數值一致。
- 完整 50 幀形狀在 meta device 通過；ACA 模組與完整 baseline 的小輸入 backward 通過。

| Variant（8類頭） | 本重建精確參數量 |
| --- | ---: |
| ResNet50 baseline | 23,524,424 |
| LSTA-only | 93,171,364 |
| TSCI-only | 23,529,224 |
| LSTA＋TSCI | 93,176,164 |

LSTA-only 與 ACA 使用上述候選超參數及 dense gating。表中參數為程式核算，
不是宣稱作者模型的參數。未載入 pretrained weights；影片讀取 0、optimizer updates 0。

## 接續順序

1. 已將參數差異定位至3D gating；作者實際設定仍待補充 ZIP／程式佐證。
2. 已完成一次解碼的50幀管線及有界真實資料驗收。
3. 正式baseline入口、完整checkpoint／resume與零更新GPU readiness已完成；等待使用者授權正式訓練。
4. 完整ACA候選已通過GPU數值forward/backward，但仍須核對93.18M／26.8M差異；不能當成忠實復刻直接訓練。
5. 架構選定後，以三個固定 seed 比較 validation loss、accuracy、macro-F1、
   Make／Miss recall 與各投籃類型，最後另依既有 test protocol 評估。

YOLO在另一checkout；本次使用者已暫停該支線，GPU檢查時未見YOLO運算程序。
正式訓練前仍應再次確認GPU占用，不自動中止其他程序。

## 參數分解與真實資料管線

完整架構說明、張量形狀、LSTA／TSCI 操作與尚待核對處見 [ARCHITECTURE.md](ARCHITECTURE.md)。

```bash
# 純參數／形狀核算：不開影片
/home/tsai/miniconda3/envs/yolo11/bin/python -B -m development.aca_net_v1.audit_architecture

# 固定 manifest/split 核對與選樣：影片讀取0、輸出檔案0
/home/tsai/miniconda3/envs/yolo11/bin/python -B -m development.aca_net_v1.pipeline_smoke \
  --source-root /home/tsai/projects/basketball --dry-run

# 有界真實資料驗收：每次建立獨立 artifacts/pipeline_smoke_<random>/
/home/tsai/miniconda3/envs/yolo11/bin/python -B -m development.aca_net_v1.pipeline_smoke \
  --source-root /home/tsai/projects/basketball --run
```

50 幀取樣公式：`index_i = floor(i * (frame_count - 1) / 49)`，含第一／最後一幀。
全片循序解碼一次，保留50幀；不足50幀直接拒絕，不重複補幀。
原圖320×240完整縮為160×120（INTER_AREA、RGB），不做裁切。
先除255，再使用 ImageNet mean/std；輸出 float32 `[3,50,120,160]`。
這是明確記錄的專案前處理選擇，不代表已還原論文未公開的 crop／取樣規則。

真實 smoke 選 2p0／2p1／ft0／ft1，各split四個不同source group，共8支影片。
僅驗收少量案例，不代表所有投籃類型／來源／FPS；沒有遍歷解碼完整train／validation。
test／excluded／跨split ID於影片path存取前拒絕；沿用既有禁止symlink／hardlink的descriptor讀取。
每支原片在解碼前後做受保護的SHA核對；8次影片解碼之外，另有16次唯讀位元組hash讀取。
每split第一支送入完整50幀ResNet50（隨機權重、CPU、eval）驗證輸出`[1,2]`與有限值，
不計算準確率。所有sample的時間索引、tensor/source SHA與shape寫入report。

產物：`config.json`、`architecture_audit.json`、`report.json`、`index.html`與8張稀疏contact sheets。
8張預覽不代表完整影片／全部50幀已完成事件可見性人工審查。

### 2026-10-25 真實資料驗收結果

- 26項模型／資料單元測試通過，含時間卷積、維度還原、權重轉移、完整參數拆解、
  label／split、單次解碼、RGB正規化、截斷／多幀／symlink／hardlink拒絕。
- [診斷頁](artifacts/pipeline_smoke_amc_te5p/index.html)、
  [機器報告](artifacts/pipeline_smoke_amc_te5p/report.json)、
  [參數與形狀追蹤](artifacts/pipeline_smoke_amc_te5p/architecture_audit.json)。
- 4 train＋4 validation，共8支原片、8次解碼；另有16次受保護的SHA位元組讀取。
- 每支影片輸出float32 `[3,50,120,160]`，來源SHA前後一致。
- 兩支完整50幀輸入通過隨機權重ResNet50 CPU前向，輸出有限`[1,2]`。
- 本小樣本原片FPS有23.976與29.97；按原始frame index均勻取樣，沒有重採樣至25FPS。
- AI看過全部8張稀疏對照圖，共64個預覽位置；觀察到部分特寫、轉場與轉播圖卡。
  詳見[AI輸入檢查](artifacts/pipeline_smoke_amc_te5p/AI_input_visual_review.json)。
- 尚未載入ImageNet、沒有optimizer updates或分類成績；test影片讀取0。

## 2026-10-25 ResNet50 ImageNet／GPU工程驗收

固定權重見 [imagenet_v1.json](imagenet_v1.json)：官方
`resnet50-0676ba61.pth`，102,530,333 bytes，完整SHA-256
`0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a`。
實檔位於ignored `artifacts/weights/`；不隱式下載或自動改用另一個權重版本。
V1是本專案選定的ImageNet初始化，作者實際checkpoint身分仍未公開核實。

原checkpoint有267個tensor，含1000類分類頭的2個tensor；265個骨幹權重／統計tensor全數轉移。
來源缺53個舊格式`num_batches_tracked`，已明確補零；新二類Linear獨立初始化。
GPU影片輸出與原生torchvision逐幀輸出的最大絕對差為`8.94e-8`。

```bash
# 安裝已從官方取得的payload；完整SHA相符才接受，拒絕覆寫不同內容
/home/tsai/miniconda3/envs/yolo11/bin/python -B -m development.aca_net_v1.weights \
  --install-from /path/to/resnet50-0676ba61.pth

# 真實train forward/backward；optimizer updates=0，無產物輸出
/home/tsai/miniconda3/envs/yolo11/bin/python -B -m development.aca_net_v1.baseline_preflight \
  --source-root /home/tsai/projects/basketball --dry-run

# 有界small-overfit；新建獨立run，最多80步、預設固定40步
/home/tsai/miniconda3/envs/yolo11/bin/python -B -m development.aca_net_v1.baseline_preflight \
  --source-root /home/tsai/projects/basketball --run --steps 40 --micro-batch 1
```

本輪固定seed20261025；只使用先前4支train影片，標籤`[0,1,0,1]`，不開validation／test。
每支完整50幀；micro-batch1、累積4支影片為一次更新，Adam lr1e-4、clip norm1。
骨幹與新分類頭更新；所有BN affine與buffers凍結，FP32、無AMP／TF32。
32項單元測試通過，涵蓋舊格式載入、微批梯度等價、BN凍結及GPU不可用拒絕。

| 項目 | 實測 |
| --- | ---: |
| 更新次數 | 40 |
| 同4支train影片eval loss | 0.651084 → 0.0（FP32數值飽和） |
| 同4支影片accuracy | 75% → 100% |
| 每次4影片更新中位時間，排除前兩步 | 0.260708秒 |
| GPU-resident輸入的訓練吞吐量 | 15.34影片／秒 |
| PyTorch峰值allocated | 2.066 GiB |
| PyTorch峰值reserved | 2.326 GiB |

計時包含forward／backward／gradient clipping／Adam，輸入已在GPU；不含解碼、資料載入或checkpoint IO。
這不能直接換算正式epoch時間或完整顯存使用，亦不含其他桌面程式的GPU記憶體。
Stem、末端stage與分類頭權重均確實更新，所有凍結BN狀態逐項不變。
checkpoint重載後logits逐位一致，原片與原始ImageNet檔案SHA未變。

[本輪summary](artifacts/baseline_preflight_seed20261025_81_s6pas/summary.json)、
[40步history](artifacts/baseline_preflight_seed20261025_81_s6pas/history.json)、
[完整config](artifacts/baseline_preflight_seed20261025_81_s6pas/config.json)。
run亦保存本輪Python來源與權重pin的副本。

**此為工程學習驗收，100%只適用這4支反覆訓練的影片。**
`engineering_only.pt`標記禁止作為正式訓練初始化，也未保存optimizer／RNG以供續訓。
正式baseline由固定ImageNet V1＋新分類頭重新開始；完整checkpoint／resume已另行實作如下。

## 訓練前最後一關（正式訓練尚未啟動）

- `train.py` 預設只產生計畫；需明確 `--start-training` 及相符的readiness報告才能開始。
- `baseline_config.json`：seed20261025、Adam1e-4、有效batch16、micro-batch1、workers2、
  最多30epochs、validation loss連續5epochs未嚴格改善早停、clip1、無scheduler／augmentation。
  BN affine與running buffers全部凍結；FP32、無AMP／TF32。patience／凍結BN／clip是專案選擇，非作者已核實設定。
- 每epoch450次更新，最後有效batch9支；依實際支數除loss，不丟棄、不重複補樣。
- `last.pt` 在optimizer邊界原子替換，含model、Adam、Python／NumPy／torch／CUDA RNG、
  epoch、cursor、部分epoch指標與history；預設每25次更新、epoch結束與SIGINT／SIGTERM停止時存檔。
- 非預期強制中止可能需重做最後已保存位置後的最多25次更新；驗證中止會重跑該輪validation，不重訓已完成的train。
- `best.pt` 僅依最低validation CE選擇；相同loss保留較早epoch。指標包含accuracy、macro-F1、
  Make／Miss recall、confusion及2p／3p／ft／mp分組；本入口無test模式。
- 續訓核對完整設定、程式SHA、固定manifest／split、全部原片stat身分、權重SHA與runtime。
  不接受早期`engineering_only.pt`，也不暗中改設定或重新抽split。

### 實測readiness

44項單元測試通過，新增覆蓋預設CLI不訓練、有效batch尾批梯度、epoch順序、完整續訓、
validation中斷、最低loss選模／tie與早停、過期readiness及工程checkpoint拒絕。

[完整報告](artifacts/training_readiness_3i_lwbki/report.json)、
[固定計畫](artifacts/training_readiness_3i_lwbki/plan.json)。

1. 8,740支train／validation原片通過guarded open＋fstat（不讀影片bytes），共5,490,358,259 bytes。
   此為存在性／身分檢查，**不是全部影片解碼驗收**。
2. 4train／4validation以正式DataLoader（workers2、spawn、pin memory）完成零更新檢查；
   所有模型權重與buffers逐項不變，8支原片SHA前後一致；test原片開啟0。
3. CUDA小型合成模型經同一`fit`迴圈，19筆資料拆16＋3、啟用dropout、真實SIGINT中斷／磁碟重載後，
   model、Adam、RNG、history與cursor與連續跑逐位一致。只有這個合成玩具模型執行optimizer更新（合計8次），
   **本輪真實ResNet50／ACA optimizer更新均為0**。
4. 完整dense ACA二類候選93,163,870參數，以一支完整50幀train輸入通過GPU前向／反向，
   gating、short-term、dynamic kernel及TSCI皆收到有限且非零梯度；權重／buffers不變。
   單次forward/backward約0.375秒，峰值allocated3.707GiB、reserved4.146GiB；
   不含Adam狀態、並非穩態benchmark或正式訓練顯存預估。
5. 正式全資料訓練未開始；沒有新增正式成績。仍未核實作者ACA設定，亦未完成所有影片的事件可見性人工審查。

啟動與安全停止／續訓指令見 [TRAINING_RUNBOOK.md](TRAINING_RUNBOOK.md)。
