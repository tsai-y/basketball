# 24 部 validation 影片人工檢視清單

固定檢視三模型各1/3、門檻0.5的既有預測，建立清單與本機來源影片連結，不解碼、不複製影片、不跑GPU、不更新模型、不讀test、不修改原標籤。不把建立清單當成人工檢視完成。

配額：罰球沒進卻判進8部、罰球進卻判沒進4部、罰球沒進判對4部、罰球進判對4部；其餘4部為非罰球且三模型都判錯，固定2p兩部、3p一部、mp一部。分層輪流挑選，優先避免同層重複來源，再平衡整份清單的來源；相同條件用 `SHA256(aca_review_v1_seed20261026:video_id)` 排序，不使用模型信心或人工觀感挑案例。配額故意增加錯誤及罰球比重，只供診斷，不用這24部估計準確率。

透過原凍結manifest/split契約核對所有1547個預測ID與標籤，核對三模型平均機率及原checkpoint/來源指紋。僅對選出的validation來源做symlink/regular-file存在性檢查，不開啟影片或讀影片bytes。來源連結指向共享原資料，不在worktree複製。

```bash
cd /home/tsai/.codex/worktrees/aca-net-reproduction/basketball
/home/tsai/miniconda3/envs/yolo11/bin/python -B -m unittest development.aca_review_v1.test_build
/home/tsai/miniconda3/envs/yolo11/bin/python -B -m development.aca_review_v1.build --build-list
```

每次建立新的 `artifacts/review24_<random>/`：

- `index.html`：可開啟原片的靜態檢視清單，含每片8項問題；沒有影片預載入、沒有自動儲存表單，適合列印或搭配JSON填寫。展開細節才顯示標籤與預測，但影片檔名含原標籤，因此不是盲審。
- `review_template.json`：24個case完整預測、來源、選入理由與空白人工欄位，reviewed=false、其他尚未確認事項為null。
- `sampling_summary.json`：配額、各層母體數、來源群組分布、固定抽樣規則、所選ID與來源指紋。

人工檢查包含：結果與依據、是否同意資料集標籤、球/籃框可見度、結果關鍵時間、是否過早截斷、後續ACA50帧及主線三視窗的事件保留情形。後兩種輸入對照本次尚未產生或檢視，不預先填入結果。
