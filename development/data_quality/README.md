# 自動品質篩檢（規劃中）

下一階段先研究 Basketball-51 full-video Make/Miss 資料的自動品質篩檢。**原始資料保持不變**；固定 manifest／split、baseline 與歷史 runs 不修改、不搬移。

預計產出：

| 產物 | 用途 |
| --- | --- |
| 品質分數表 | 以既有 video ID 關聯，記錄分數、品質方法版本及資料來源 |
| 候選排除清單 | 提供待審查的影片 ID 與理由；不直接刪除影片或改寫固定 manifest |
| 報告 | 記錄方法、版本、使用的 split、觀察、限制及人工審查結果 |

**本輪尚未實作品質篩檢／清理演算法、計算品質分數、決定門檻或排除任何影片。** 具體品質指標、門檻與審查規則待後續研究；影像增強策略也尚待確認。

未來輸出位置依 [development 規則](../README.md) 先行記錄，每次實驗附 manifest、split、前處理版本、設定與輸出路徑。不得用 test 影片內容或分布制定品質政策。閱讀 [目前研究方向](../../docs/current_research_direction.md) 與 [repository map](../../docs/repository_map.md) 後再展開下一輪工作。
