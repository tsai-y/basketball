# 主機重啟後的明確授權續訓

原 campaign：`artifacts/three_seed_cew8zqhi`。
20261024／20261025 已完成；20261026 在 epoch5 的6512／7193支時因主機重啟中斷。
最後完整 checkpoint 為 epoch4、1800 optimizer updates；先前未保存的407次更新
不計入續訓後的 global_step。使用者於2026-10-31明確授權續訓。

`resume_campaign_v1.py` 是獨立的單次恢復監督程序；原 training、train、readiness、
作者 TAM、recipe、三 seed plan 及 contract 全部保持相同 SHA。
`--prepare <campaign>` 檢查新 boot ID、原程序身分已失效、checkpoint／history／
最佳 epoch 一致、runtime未變、尚未達15輪或patience5，並驗證已完成seed及原始
保護鏈。它將原 launcher／state／日誌和 best／last checkpoint 複製到
`resume_7qmvw180/before_resume/`，保存來源和備份 SHA。

`--run <resume session>` 取得與原 campaign 共用的 GPU 檔案鎖，再跑同一套 GPU
工程預檢。預檢權重不作正式初始化。將確認已失效 boot 的 stale active.lock 移入
恢復紀錄後，呼叫原正式入口 `--start --readiness-report ... --resume <run>/last.pt`。
原入口恢復模型、BN buffers、Adam 與全部 RNG 狀態，以 seed+epoch 的相同順序
重播完整 epoch5；繼續原15輪上限與patience5，以同一 L validation 規則選模。

root state.json 保留先前 jobs，將被重啟打斷的 job 標記 interrupted_by_host_reboot、
未知 returncode=null；新增本次 readiness／training job、PID／start ticks／boot ID／
命令列及獨立 stdout_log。launcher.json 指向新監督程序；原 launcher 已存入備份。
progress.log 以追加方式保留中斷與重播紀錄；讀目前 job 的 stdout_log 的唯一 OUTPUT
行找回同一正式 run，再以 progress.log 最後一個 OUTPUT 後的記錄判斷目前進度。

已完成的兩個 seed 以 SHA 保護，不重跑。最後 seed 完成後，沿用原 campaign
summarize 從已存 validation logits 產生三 seed 報告。失敗即停止，無自動重試或
自動續訓；再次主機重啟仍需使用者明確授權。無 test、門檻修改或新選模規則。

檢查：既有8項L policy測試與5項恢復測試通過；恢復測試包含 checkpoint 邊界、
拒絕完成／早停／runtime不符、原程序身分檢查、失敗不重試及完成seed雜湊保護。
本次現場 GPU 預檢與實際續訓結果另保存在 resume session 與 campaign state。
