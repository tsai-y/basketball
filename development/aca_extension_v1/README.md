# Dense ACA：五輪後延長至總15輪

這是使用者看過五輪validation結果後授權的訓練預算延長，不是事前即固定15輪的獨立實驗，也不是作者原版ACA-Net。模型為93,163,870參數的dense ACA候選。

## 本次啟動紀錄

- GPU預檢：`artifacts/gpu_preflight_qqyb7xte/report.json`，status=passed；真實影片optimizer更新數為0。
- 正式run：`artifacts/aca_extend15_seed20261025_wuaps2g9/`。
- 日誌與啟動命令：`artifacts/extension_launch_seed20261025/console.log`及`launch.json`。
- 正式run初始checkpoint已另外核對：model、Adam與RNG和父run逐位一致；progress僅finished從true改為false，其餘不變。
- 每10分鐘的完成提醒只讀此run；正常保持安靜，完成/確認異常才通知，並自行停用提醒；不自動重試或續跑。

## 唯一訓練設定變更

總epochs上限 **5 → 15**，不是另外增加15輪。從第5輪`last.pt`接續第6輪，最多再跑10輪。patience仍為5，依最低validation CE選模，ties保留較早輪；其他seed、Adam、lr、batch、BN、資料、前處理、checkpoint節奏與FP32設定不變，沒有新scheduler。

固定父run：
`/home/tsai/.codex/worktrees/aca-net-reproduction/basketball/development/aca_candidate_v1/artifacts/aca_train_seed20261025_hmqhr2pg/`

承接完整model、Adam moments及step、Python/NumPy/torch/CUDA RNG、epoch=5、optimizer_steps=2250、history、best loss與bad_epochs。只解除父run因max_epochs=5而設定的finished旗標；不接受已early-stop、部分輪或工程checkpoint。父run目前第5輪最佳，因此bad_epochs本來就是0，不是人為重設。

父run所有輸出唯讀，並記錄config、history、summary、last、best與gpu_preflight的SHA256。舊dense／DW3D／baseline程式不改；原strict resume契約不放寬。新的輸出、source snapshot與provenance獨立存於本模組`artifacts/aca_extend15_seed20261025_<random>/`。

新run先繼承父run最佳checkpoint；若第6輪之後都變差，仍保留第5輪best，而不是以新run最後一輪取代。從第5輪重啟至總15輪的連續性以合成dropout模型驗證；這不等於已實測完整真實影片不中斷15輪的逐位軌跡。

## 預檢與啟動

```bash
cd /home/tsai/.codex/worktrees/aca-net-reproduction/basketball
/home/tsai/miniconda3/envs/yolo11/bin/python -B -m development.aca_extension_v1.train \
  --source-root /home/tsai/projects/basketball --preflight --accept-candidate-architecture
```

預檢先核對父run的原始plan與目前來源/程式/runtime完全一致，再核對只允許epochs變更。GPU檢查包含合成5→15輪接續與不中斷軌跡一致、真實parent模型/Adam/RNG重載、兩支train影片零更新backward、全模型及凍結BN不變、checkpoint logits與Adam moments逐位一致。工程checkpoint禁止作為正式初始化；正式訓練重新從原父run載入。

```bash
/home/tsai/miniconda3/envs/yolo11/bin/python -B -u -m development.aca_extension_v1.train \
  --source-root /home/tsai/projects/basketball --start-training --accept-candidate-architecture \
  --preflight-report /ABSOLUTE/EXTENSION_GPU_PREFLIGHT/report.json
```

不帶模式僅輸出plan。正式啟動需要CUDA、沒有其他非桌面GPU compute、至少8GiB磁碟，且延長版預檢report指紋吻合。原五輪預檢不能直接解鎖此入口。

`config.json`保存總15輪plan；`continuation.json`記錄父run與繼承進度；`history.json`包含原5輪及後續輪；`best.pt`/`last.pt`沿用嚴格checkpoint機制。人工中斷後可用同一命令加`--resume /ABSOLUTE/EXTENSION_RUN`，僅接受同plan延長版run，不自動續跑或重試。

所有成績均為validation，test沒有介面、不讀test影片。沒有自動融合、YOLO處理、commit或push。延長訓練不保證達到90%。
