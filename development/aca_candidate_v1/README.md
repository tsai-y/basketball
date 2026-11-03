# ACA dense候選：獨立訓練支線

2026-10-26 更新：使用者選擇先進行修正前 dense ACA 候選。
**CPU準備與GPU預檢已通過，已啟動五輪全train／validation篩選實驗；不測test、不自動延長至30輪。**
新增內容位於本目錄；`development/aca_net_v1/`原有程式全部保持不變。

## 本次五輪篩選實驗

獨立設定為 `pilot_config.json`，僅將 epochs 改為5；seed、資料、初始化、凍結BN、Adam、lr、batch、clip等與DW3D實驗相同。
原本 `config.json` 的30輪設定未改。五輪只是篩選，不能把尚未收斂的結果當成模型優劣定論；是否續跑需另行決定，不根據test選擇。

- CPU準備：`artifacts/cpu_preparation_0isjljqz/report.json`。
- GPU預檢：`artifacts/gpu_preflight_z4nq3vr6/report.json`，status=passed；一次4片工程更新、各分支梯度、凍結BN、checkpoint logits及CUDA續跑一致性通過。
- 正式run：`artifacts/aca_train_seed20261025_hmqhr2pg/`；模型重新從ImageNet及全新ACA開始，不承接工程或DW3D權重。
- 輸出日誌：`artifacts/dense_pilot_launch_seed20261025/console.log`；啟動PID、命令及plan指紋保存在同目錄`launch.json`。
- 預檢plan SHA256：`cf3b6864b54218ff789313158a64901d27bd620461fe3e96b8a9bcea12f6470a`。

下方保留原30輪入口的說明與準備紀錄。執行本次五輪入口必須加 `--config development/aca_candidate_v1/pilot_config.json`；本次GPU報告僅適用五輪plan，不能替原30輪設定解鎖。

## 研究定位

這是依現有`aca_net_v1.model.VideoResNet50`重建的dense ACA候選，
不是作者程式，也不是已核實的精確復刻。
ResNet50共16個bottleneck，內部插入平行的LSTA＋TSCI；輸入為完整影片均勻50幀、120×160。
候選架構固定r16、short kernel3、dynamic kernel3、beta0.25、dense gating、gate sigmoid。
二類版本93,163,870參數，與論文約26.8M仍有差距；不暗中換depthwise湊參數。
本實驗目的是測試這個明確標示的候選是否有效，不保證test達90%。

## 與baseline的對照與差異

- 固定7,193train／1,547validation與原始source-group切分；本入口沒有test功能。
- 同樣50幀、全畫面160×120、ImageNet正規化，無augmentation／品質降權／YOLO。
- 骨幹從完整SHA固定的ImageNet V1重新開始，不使用正在訓練的baseline權重。
- 新二類head刻意與同seed baseline的**初始head**逐項相同；不是從已訓練head繼承。
- ACA額外參數重新初始化；不是從先前工程overfit checkpoint載入。
- seed20261025、Adam lr1e-4／wd0、clip norm1、有效batch16、micro1、workers2、FP32無AMP／TF32。
- 最多30輪，validation CE連續5輪未嚴格改善早停；最低CE保存best，tie保留較早epoch。
- 凍結所有BN affine與running buffers，包含新TSCI的BN；可訓練93,110,654參數。
  此BN設定與patience／clip等屬專案選擇，不是作者已核實設定。
- 每epoch450次更新，最後有效batch9支依實際支數平均；不丟棄或重複補樣。

本候選與baseline仍不是已完成的受控消融：新增模組會改變初始化後的特徵分布，
須等完整validation與後續LSTA-only／TSCI-only比較後，才能歸因效能差異。

## 已完成的CPU驗收

- 候選16項測試通過，原baseline44項回歸測試亦通過；全程`CUDA_VISIBLE_DEVICES=''`。
- 合成小型線性分類器及小型ACA＋dropout均經同一candidate訓練迴圈；
  真實SIGINT、保存、重建與重載後，model、Adam、RNG、history、cursor與連續跑逐位一致。
- 有效batch16＋尾批3的梯度與直接batch mean參考一致；早停、loss tie、validation中斷恢復通過。
- baseline／engineering／synthetic checkpoint禁止被正式ACA續訓入口接受。
- CPU初始化93M參數、完整ImageNet轉移、有限值與配對head通過；完整50幀meta輸出`[1,2]`通過。
- 全部8,740支train／validation原片做guarded open＋fstat身分檢查，不解碼、不讀原片payload。
- CPU準備報告中的`cuda_context_initialized=false`、`optimizer_steps=0`、`test_videos_opened=0`。

CPU報告：[report.json](artifacts/cpu_preparation_y10_f60j/report.json)，
完整計畫：[plan.json](artifacts/cpu_preparation_y10_f60j/plan.json)。
meta是形狀驗收，不代表完整模型的CPU/GPU數值訓練或實際訓練速度。
先前`aca_net_v1`曾完成完整候選GPU零更新backward；本入口的新配對head、
Adam更新／顯存及checkpoint roundtrip仍需下方新GPU預檢，不借用baseline的readiness報告。

## 現在可做：CPU準備（不搶GPU）

```bash
cd /home/tsai/.codex/worktrees/aca-net-reproduction/basketball

CUDA_VISIBLE_DEVICES='' /home/tsai/miniconda3/envs/yolo11/bin/python -B \
  -m development.aca_candidate_v1.train \
  --source-root /home/tsai/projects/basketball --prepare
```

每次建立獨立`artifacts/cpu_preparation_<random>/`；預設不帶模式只印出計畫，不寫run。
CPU準備僅透過`nvidia-smi`查詢GPU硬體身分，不建立Python CUDA context。
計畫記錄程式、設定、權重、原片stat及runtime身分；變更後需要重做預檢。

## 等目前baseline完成後：兩步啟動

目前固定前置run為：
`development/aca_net_v1/artifacts/baseline_train_seed20261025_yqk5pn63/`。
它必須有`status=completed`的summary，而且GPU不能有其他非桌面運算程序。
程式只會拒絕占用，不會幫你kill其他程序。至少需要8GiB可用磁碟。

### 第一步：GPU工程預檢（尚未執行）

```bash
/home/tsai/miniconda3/envs/yolo11/bin/python -B \
  -m development.aca_candidate_v1.train \
  --source-root /home/tsai/projects/basketball \
  --preflight --accept-candidate-architecture
```

預檢將執行：

1. 合成CUDA玩具模型的candidate迴圈續訓一致性測試。
2. 4支真實train影片的一次有界**工程**Adam更新；確認所有梯度有限、各ACA分支有梯度、BN不變。
3. 4支validation工程推論，以及工程checkpoint重載logits逐位一致。
4. 來源SHA前後一致、含Adam狀態的峰值顯存與工程耗時紀錄。

工程checkpoint禁止作為正式初始化，四影片分數也不是泛化成績。
成功報告保存在獨立`gpu_preflight_<random>/report.json`，並建立便利鏡像
`artifacts/latest_gpu_preflight.json`。未通過就不能開始下一步。
新入口目前沒有完整epoch實測工時，不以舊單次backward秒數估完整訓練時間。

### 第二步：正式訓練（尚未執行）

```bash
/home/tsai/miniconda3/envs/yolo11/bin/python -B \
  -m development.aca_candidate_v1.train \
  --source-root /home/tsai/projects/basketball \
  --start-training --accept-candidate-architecture
```

預設驗證上述`latest_gpu_preflight.json`；也可用`--preflight-report /absolute/path/report.json`明確指定。
正式訓練再次從固定ImageNet＋新ACA＋配對新head開始，不接工程更新後權重。
輸出新建於本目錄`artifacts/aca_train_seed20261025_<random>/`，不覆寫baseline。
此處兩個指令須由使用者明確執行；沒有背景排程或自動串接。

## 停止與續訓

按一次Ctrl-C或送SIGTERM，等目前有效batch完成後保存`last.pt`並退出。
checkpoint含model、Adam、Python／NumPy／torch／CUDA RNG、epoch、cursor、部分epoch指標與history。
每25次更新、epoch結束及安全停止時原子保存；強制中止最多重做最後25次未保存的更新。
validation中斷會重跑該輪validation，不重複已完成train更新。

```bash
# 換成終端實際列印的ACA run目錄；不要使用baseline run
/home/tsai/miniconda3/envs/yolo11/bin/python -B \
  -m development.aca_candidate_v1.train \
  --source-root /home/tsai/projects/basketball \
  --resume /ABSOLUTE/aca_train_seed20261025_RUN_DIRECTORY \
  --start-training --accept-candidate-architecture
```

schema與baseline分離，且續訓核對完整計畫；修改程式／設定／來源／runtime後不能靜默續跑。
合成模型的逐位一致不等於完整93M模型已完成真實中斷續訓實驗。

## 仍待完成

GPU工程預檢、正式ACA全量訓練、完整validation、凍結後的正式test入口與一次性評估，
以及作者設定核對／LSTA與TSCI消融。沒有宣稱新準確率，也沒有改動既有test結果。
