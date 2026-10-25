# Baseline啟動前交接

本次停在正式訓練前。下方`--start-training`指令**尚未執行**；需要使用者下一步授權。
這是專案二分類ResNet50比較基準，不是已驗證忠實復刻的ACA，也不是論文八類92.05%的重現結果。

## 目前設定

設定檔：`baseline_config.json`；來源原片唯讀取自`/home/tsai/projects/basketball`。
固定7,193train／1,547validation；test在本入口禁止。
ImageNet V1＋新二類分類頭初始化，不使用4影片overfit checkpoint。
Adam1e-4、有效batch16（micro1累積）、FP32、凍結全部BN、無augmentation／scheduler。
最多30epochs、validation CE連續5輪未嚴格改善早停；不依validation accuracy挑checkpoint。
每epoch450更新（449×16＋9）；模型權重、split、設定及來源身分不可在續訓時靜默變更。

## 唯讀計畫與零更新檢查

在專用worktree，不切回YOLO checkout：

```bash
cd /home/tsai/.codex/worktrees/aca-net-reproduction/basketball

# 預設只列計畫：不建run、不解碼、不建optimizer、不更新模型
/home/tsai/miniconda3/envs/yolo11/bin/python -B -m development.aca_net_v1.train \
  --source-root /home/tsai/projects/basketball

# 真實模型零optimizer更新；合成玩具模型另做續訓測試
/home/tsai/miniconda3/envs/yolo11/bin/python -B -m development.aca_net_v1.readiness \
  --source-root /home/tsai/projects/basketball
```

readiness預設亦檢查完整dense ACA候選；它只通過數值／GPU工程關卡，不能取代作者設定核對。
每次檢查建立獨立`artifacts/training_readiness_<random>/`。程式／設定／資料／runtime改變，
舊報告即失效，需重新執行readiness；不能手改報告雜湊繞過。
原片身分使用全部train／validation的dev／inode／size／mtime／ctime，不等於完整內容SHA或完整解碼審查。

## 正式啟動（未執行）

先確認YOLO仍暫停、GPU無其他訓練占用，磁碟至少保留5GiB。
本次有效報告如下；若未來重新檢查，請使用新輸出目錄的report。

```bash
/home/tsai/miniconda3/envs/yolo11/bin/python -B -m development.aca_net_v1.train \
  --source-root /home/tsai/projects/basketball \
  --preflight-report /home/tsai/.codex/worktrees/aca-net-reproduction/basketball/development/aca_net_v1/artifacts/training_readiness_3i_lwbki/report.json \
  --start-training
```

終端會列印新run完整路徑：`artifacts/baseline_train_seed20261025_<random>/`。
先確認首輪train／validation正常、loss有限與batch數正確，再判讀學習曲線。
本次只驗收4train／4validation的實際解碼，尚未逐支完整解碼8,740支影片；正式遇到壞片會停止，
不會靜默排除／重標籤／跨split補樣。遇到錯誤先診斷，不改固定split以求更高成績。

## 安全停止與續訓

按一次Ctrl-C或送SIGTERM，等當前有效batch完成；程序保存`last.pt`後離開。
強制kill／斷電不保證保存當前進度；只從最後完整checkpoint重做，最多25次更新。
驗證中斷會重跑該輪validation，已完成train不重複更新。

```bash
# 將 /ABSOLUTE/RUN/DIRECTORY 換成終端實際印出的baseline_train_*目錄
/home/tsai/miniconda3/envs/yolo11/bin/python -B -m development.aca_net_v1.train \
  --source-root /home/tsai/projects/basketball \
  --preflight-report /home/tsai/.codex/worktrees/aca-net-reproduction/basketball/development/aca_net_v1/artifacts/training_readiness_3i_lwbki/report.json \
  --resume /ABSOLUTE/RUN/DIRECTORY \
  --start-training
```

`last.pt`為續訓真實來源；`history.json`只在epoch完成後更新，若最後寫入中斷，可能暫時比checkpoint落後。
`best.pt`依最低validation CE保存。兩者含完整optimizer與RNG；不要拿舊`engineering_only.pt`替代。
checkpoint採同目錄temporary file＋fsync＋atomic replace，避免保存半個檔案；暫存舊檔不會被當成last。
使用相同軟硬體與依賴環境；目前逐位一致驗收針對合成CUDA分類器，並非宣稱整個ResNet50已完成中斷訓練實驗。

## ACA未解除的關卡

完整候選已通過真實50幀GPU forward/backward，但參數量93.18M仍不符合論文26.8M。
不能因GPU跑得動就稱精確復刻；正式CLI目前只接受baseline，拒絕ACA變體。
應取得作者程式／補充設定，或將明確標示的候選及消融另立實驗協定，再開始ACA訓練。
目前沒有使用test選模，也沒有可宣稱達成90%的新準確率。
