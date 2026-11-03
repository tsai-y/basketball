# 主線 MixNet-I3D 與 ACA 的 validation 融合

唯讀使用既有 checkpoint 內保存的 validation logits，不解碼任何影片，不跑GPU、不建立optimizer、不更新模型、不讀test、不覆寫舊產物。主線固定使用 `rgb_seed20261024_bkegdbsu/best.pt`：正式RGB微調已早停，但最低 CE 選出 epoch0，代表原初始化模型，不宣稱微調帶來提升，也不跨seed挑最好者。

先核對 checkpoint schema、config、來源程式與初始化SHA256、原始凍結manifest、1547個validation ID/labels、checkpoint保存的logits與summary，再以 Artifact Tool 唯讀讀入主線 CSV，檢查逐列機率/預測和checkpoint一致。history包含epoch0，依最低CE核對選模。ACA則重用上一輪已驗證的逐片JSONL並重新核對原產物指紋。依video ID對齊，不假設列序相同。

本輪固定新增四種候選：主線+Dense各50%、主線+DW3D各50%、三模型各1/3、主線50%+Dense25%+DW3D25%；保留原ACA各50%作參考。不搜尋權重、seed、分類門檻，不依投籃類型另設規則。融合機率大於0.5為Make，恰好平手為Miss。主線單模型沿用原logit>=0的Make平手規則。CE使用log-probabilities穩定計算。全數結果公開，探索性選擇依最低validation CE而非最高accuracy，並保留原單模型及ACA融合。

在本目錄 `node_modules` 建立指向 Codex bundled runtime Node packages 的 symlink 後：

```bash
cd /home/tsai/.codex/worktrees/aca-net-reproduction/basketball
/home/tsai/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node --test development/aca_mainline_fusion_v1/test_fuse.mjs
# 不帶旗標只顯示預先固定方案
/home/tsai/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node development/aca_mainline_fusion_v1/fuse.mjs
# CPU checkpoint核對及既有validation預測融合；不推論影片
/home/tsai/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node development/aca_mainline_fusion_v1/fuse.mjs --execute
```

每次建立 `artifacts/mainline_fusion_<random>/`，保存來源provenance、主線已驗證逐片預測、各組融合逐片預測、完整report；原來源與本輪程式指紋前後必須相同。共同錯誤的label oracle僅為不可部署診斷，不是實際成績。所有候選都重複使用開發validation，不是獨立test、不能保證90%。

## 2026-10-26 完成結果

輸出 `artifacts/mainline_fusion_4Lbbzt/`。10項Node測試、3項CPU checkpoint回歸測試通過；所有1547部影片的主線CSV機率與原checkpoint逐筆scalar sigmoid精確一致。兩個初期工具比對錯誤（JSON整數鍵、向量/純量sigmoid ULP差異）均已定位並修正，沒有改來源檔或放寬機率比較。五組融合另經獨立ID、機率、預測、判對數核算通過。

| 方法 | Accuracy | Macro-F1 | CE loss |
| --- | ---: | ---: | ---: |
| 原ACA各50% | 82.22% | 82.07% | 0.39032 |
| Dense+主線各50% | 83.06% | 82.95% | 0.38987 |
| DW3D+主線各50% | 82.42% | 82.36% | 0.40850 |
| 三模型各1/3 | 83.84% | 83.74% | 0.38024 |
| 主線50%+兩ACA各25% | 83.26% | 83.17% | 0.39031 |

依最低CE選出三模型各1/3，判對1297/1547；相較原ACA融合救回59部、新增判錯34部，淨增25部（1.62個百分點）。主線可獨立判對兩ACA共同錯誤中的90/199部，但三者仍同時判錯109部。三模型融合對罰球沒進的recall仍只有24.44%，不能宣稱解決此弱點。

`freeze.mjs` 對本次完成報告與所有來源指紋再次確認後，僅以exclusive write保存 `candidate.json`，含三份checkpoint/config指紋、best epoch、前處理、各1/3權重、類別映射、閾值及所選逐片預測指紋。這是已驗證validation預測的候選設定，不等於已實作完整影片的三分支推論入口，未執行test、未更新任何模型、不保證90%。
