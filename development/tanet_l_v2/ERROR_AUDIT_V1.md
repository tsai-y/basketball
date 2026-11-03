# L v2 完成後：共同誤判、分歧與固定24片診斷

來源為已完成的 `artifacts/three_seed_cew8zqhi`，三 seed 各依最低完整 validation CE
選定 best。這輪只讀已存 logits／預測與24支 validation 原片；沒有模型forward、
optimizer、重訓、test、門檻／標籤修改或特徵快取整合。

`diagnose_errors_v1.py --prepare` 以 canonical validation ID 嚴格連接三 seed 和主線，
重算 best logits 指標，核對逐片 CSV、contract 與原保護鏈。完整1547支的共同誤判、
seed分歧、全對及各seed分類指標按來源、出手類型、標籤及交叉分組輸出。
「共同誤判率」的分母是該組影片數；「平均seed錯誤率」為該組三seed錯誤次數／3N。
只用資料集既有標籤，尚無新人工真值。

抽樣seed20261031，先保存 sampling_plan.json，再允許 `--run <audit output>` 讀像素：
共同誤判12支（每label6支：2p與ft各2、3p與mp各1）、分歧8支（每label×出手類型1支）、
全對4支（每出手類型1支，Miss／Make交替）。來源分散優先，再以SHA排序；不以模型
信心高低排序。任何配額缺少便停止，不在看畫面後替換樣本。共24支、兩標籤各12支。

原生320×240順序解碼只開每支原片一次，唯讀guard拒絕跨split與test。
保存所有原生PNG、有損原片預覽、固定12幀總覽、完整原生縮圖、實際50幀RGB縮放影格、
原始frame index／時間戳與normalized tensor SHA。均勻取樣、resize、normalize直接
沿用固定訓練函式；這是輸入重建，不是新模型forward的重現測試。

index.html 提供隱藏標籤／分數、原生逐幀滑桿、最近抽樣影格對照、50幀總覽、固定樣本
分組篩選與人工筆記匯出。筆記只保存在瀏覽器localStorage，未回寫資料集；匯出的JSON
也不是自動標籤補丁。human_review_template.json 的所有確認欄位預設false／null。

AI初審將另存 review.json／review.html，記載實際看過的影格證據與不確定之處。
在沒有明確出手／結果時間標註前，抽樣間隔不能直接證明漏取事件；縮圖或模糊球路也
不能直接證明原標籤錯誤。固定24片的分層比例不代表整體資料分布或模型泛化能力。

輸出在 `artifacts/error_audit_v1_*`，保留 sampling_plan.json、report.json、
distributions.json、全片預測及各組分母CSV、24片回看素材、來源存取紀錄與QA。
原訓練程式、作者TAM、資料清單、checkpoint與正式訓練結果均以SHA保護，不覆寫。
