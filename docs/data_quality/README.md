# Basketball-51 資料品質研究索引

> - **給誰看：** 想理解資料品質研究現況與證據的一般組員。
> - **回答什麼：** v1、v2、v3/v3.1、Sampler v1 與 Visual Preprocessing Pilot 各自解決什麼問題、該讀哪些文件。
> - **不包含：** CLI、程式清單、sidecar schema 細節或完整研究報告內容。
> - **上一層入口：** [研究文件索引](../README.md)。
> - **建議下一份：** [Quality-aware Clip Sampler v1](quality_aware_clip_sampler_v1.md)。

本頁是資料品質研究的繁體中文導航。需要重現、稽核或維護程式時，改讀[唯一技術入口](../../development/data_quality/README.md)。

## 一般組員怎麼讀

必讀：

1. [目前研究方向](../current_research_direction.md)
2. 本頁
3. [Quality-aware Clip Sampler v1](quality_aware_clip_sampler_v1.md)

依工作選讀：

- 要了解候選如何形成：讀 [v3 方法](blur_screening_method_v3.md) 與 [v3.1 publisher contract](blur_screening_publisher_v3_1.md)。
- 要追溯門檻來源：讀 [v2 方法](blur_screening_method_v2.md) 與 [v2 報告](blur_screening_pilot_v2.md)。
- 要評估前處理候選：讀 [Visual Preprocessing 方法](visual_preprocessing_method_v1.md) 與 [Pilot 報告](visual_preprocessing_pilot_v1.md)。
- 只有做歷史稽核時，才需要閱讀 v1 或各 smoke report。

## 研究階段

| 階段 | 狀態 | 回答的問題 | 主要文件 |
| --- | --- | --- | --- |
| Quality Screening v1 | 完成／歷史 | 全畫面清晰度指標與小樣本候選分布是否值得研究 | [方法](blur_screening_method_v1.md)、[Pilot 報告](blur_screening_pilot_v1.md) |
| Quality Screening v2 | 完成／歷史 | 合成模糊與 temporal calibration 如何提供 v3 凍結參數 | [方法](blur_screening_method_v2.md)、[Pilot 報告](blur_screening_pilot_v2.md) |
| Quality Screening v3/v3.1 | 完成 | train-only 完整掃描、恢復、驗證與發布契約 | [方法](blur_screening_method_v3.md)、[smoke 報告](blur_screening_smoke_v3.md)、[publisher contract](blur_screening_publisher_v3_1.md) |
| Quality-aware Clip Sampler v1 | 完成 | 如何用 tracked sidecar 選擇連續 train clips，並記錄 fallback | [方法](quality_aware_clip_sampler_v1.md)、[smoke 報告](quality_aware_clip_sampler_smoke_v1.md) |
| Visual Preprocessing Pilot v1 | Pilot 完成；正式政策未定案 | 候選 transform 的介面、決定性與有限 train 樣本診斷 | [方法](visual_preprocessing_method_v1.md)、[Pilot 報告](visual_preprocessing_pilot_v1.md) |

## 版本關係

```text
v1 指標探索
  └─> v2 指標校準
        └─> v3/v3.1 完整 train 掃描與 publication
              └─> tracked train-only sidecar
                    └─> Quality-aware Clip Sampler v1
                          └─> Visual Preprocessing Pilot v1
```

v1/v2 是歷史證據，不是目前完整掃描入口；v3/v3.1 完成後，正式技術介面是 tracked sidecar 與 sampler 契約。視覺 Pilot 只提供前處理候選的工程證據，不代表模型效果或正式政策已定案。

## 重要判讀邊界

- `candidate` 不等於確認模糊、排除名單或刪除授權；未標記也不等於人工確認清晰。
- 所有品質掃描與 sidecar 都是 train-only；validation/test quality mode 維持停用。
- 原始影片、固定 manifest／split、A1/A2/B/C、歷史 runs、設定、sidecar 與報告數值不由索引文件回寫。
- I3D 特徵擷取、YOLO 特徵／ROI 與多模型融合尚未開始。

## 實作者入口

configs、scripts、samplers、transforms、tests、sidecars、ignored artifacts、CLI 與重現方式集中在[資料品質技術入口](../../development/data_quality/README.md)。完整 train 主機操作只讀 [v3/v3.1 runbook](../../development/data_quality/runbooks/blur_train_v3_tmux.md)；不要從本頁推導命令。
