# ACA 架構核對 v1（唯讀）

只核算固定候選的參數／meta shape，並以 CPU 合成張量核對動態時間卷積與 Global BN。
沒有訓練、資料集、checkpoint、GPU 或 test 評估入口。ProposalLSTA 僅供核對貼文，不能當作作者實作。
完整結論見 [架構核對報告](../../docs/research/ACA_ARCHITECTURE_AUDIT.md)。

```bash
CUDA_VISIBLE_DEVICES='' /home/tsai/miniconda3/envs/yolo11/bin/python -B -m unittest development.aca_architecture_audit_v1.test_audit
```

完整 CLI 需要原始使用者附件，以及另一 TANet 分支保存的 SHA 固定作者檔：

```bash
CUDA_VISIBLE_DEVICES='' /home/tsai/miniconda3/envs/yolo11/bin/python -B -m development.aca_architecture_audit_v1.audit --proposal /ABSOLUTE/Pasted-text.txt --tam /ABSOLUTE/temporal_module.py --output /ABSOLUTE/NEW-result.json
```

輸出必須不存在，不覆寫已保存結果；附件只做 AST 預設值核對，不執行附件程式。
缺少原附件時仍可用 parameter_audit() 重算固定版本；不能宣稱已重新核對附件。
作者 TAM 必須符合固定 SHA；工具不自動下載，也不靜默改用另一個版本。
既有模型程式與設定不改；本工具的合成 forward/gradient 不代表真實影片效能。
