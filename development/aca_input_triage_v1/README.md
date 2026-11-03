# 既有24部模型輸入的助理視覺初判

`observations.json` 是助理在本對話實際看過既有PNG後撰寫的非盲式觀察，不是程式辨識結果，不是第二位人類標註。已知道使用者答案及模型預測，因此不可作獨立真值或評分。原片沒有重新解碼，僅讀既有validation影格頁與JSON；不讀test影片、不改人工欄位／標籤／分割／模型／成績。

每部查看完整ACA50張、原片／ACA／主線三時點對照及指定主線連續頁，疑點另看列明的原片或主線頁。不是逐片完整影片播放，也不是看完每部3×64張主線輸入；輸出逐頁列出實際查看範圍。儲存PNG不變，檢視工具可能稍微縮放大型contact sheet。

初判分類：3部關鍵關係較清楚（R07/R12/R17）、18部脈絡保留但細節有限、3部需要使用者確認模型輸入可辨識性（R11/R14/R21）。未要求立即複查不等於品質合格，不從定向樣本估計母體比例。R02及R24沿用使用者已確認事項，不重問、不刪除。

`build.py` 僅驗證／發佈既有觀察、引用原PNG，沒有解碼器或ML推論。保護既有人工作業、全部抽幀產物及實驗檔案SHA256前後不變。每次輸出新目錄，保留舊報告；沒有自動重新判圖或啟動訓練。

```bash
cd /home/tsai/.codex/worktrees/aca-net-reproduction/basketball
/home/tsai/miniconda3/envs/yolo11/bin/python -B -m unittest development.aca_input_triage_v1.test_build
/home/tsai/miniconda3/envs/yolo11/bin/python -B -m development.aca_input_triage_v1.build
```

「人可判讀」不保證網路可利用；主線同時改變時間密度與解析度，不能直接當解析度消融。下一輪可先評估固定ACA50張／全畫面、僅提高解析度的成本，再決定是否訓練；尚未建立高解析度訓練版，也不保證test90%。
