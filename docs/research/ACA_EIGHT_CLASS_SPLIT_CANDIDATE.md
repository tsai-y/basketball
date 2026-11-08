# ACA 八類：資料品質規則與80/10/10來源分組候選

日期：2026-11-08。**D1品質規則及D2切分候選已完成；候選尚未啟用，訓練與每小時排程維持暫停。**

**後續更新：[D3單一前處理已定稿並通過17項合成測試](ACA_EIGHT_CLASS_PREPROCESSING.md)，下一步是D4真實解碼與八類adapter。下文保留D1/D2完成時的決策快照；候選未啟用。**

這次實際產出一份獨立metadata manifest；不是只提出下一輪清單。原始影片、二類manifest／split及既有run全部保留。新候選不是作者原split，也不代表原論文精確複刻已完成。

- [固定政策](specs/aca_eight_class_split_policy_v1.json)
- [候選摘要、完整來源群組assignment與SHA](results/aca_eight_class_split_candidate_v1.json)
- [驗收與負向案例](results/aca_eight_class_split_candidate_qa_v1.json)
- [本機完整候選manifest](../../development/aca_eight_class_data_v1/artifacts/sourcegroup80_seed20261108_v1/candidate_manifest.csv)（ignored，可由腳本重建）
- [先前G2盤點快照](ACA_EIGHT_CLASS_DATA_CONTRACT.md)、[研究決策總覽](ACA_REPRODUCTION_CHECKPOINT.md)

## 1. D1：統一隔離歧義，不猜測真值

以已保存的29組位元完全相同清單為依據，不開影片、不看模型成績：

1. 同內容但original_class不同：**整組隔離**，不論Make/Miss是否相同。
2. 同內容、八類標籤也相同：按video_id字典序留一份，其餘列為duplicate_copy。
3. 其餘影片保留原八類與二類標籤，沒有人工裁決或自動改標。

新候選隔離5組異類內容的10支影片，另排除24支同類重複副本，總共34支。原版排除32支，因此只新增隔離兩支原本保留的歧義來源：

- `2p1/2p1_v166_005133_x264.mp4`：舊Train，與mp1同內容。
- `3p0/3p0_v121_000328_x264.mp4`：舊Test，與mp0同內容。

這是保守的八類品質規則，不宣稱哪個原標籤正確。新候選沒有重新納入任何舊版excluded影片；也不預測隔離兩支就能提升多少accuracy。規則只處理既有已知完全重複清單，近重複／重播仍未全面核實。

## 2. D2：固定規則產出一份重建切分

| 項目 | 固定值 |
| --- | --- |
| split seed | 20261108，與model訓練seed分開 |
| 來源群組 | 50群；v160/v166維持合併 |
| 群組配額 | Train40／Validation5／Test5 |
| 影片比例目標 | 80%／10%／10%；以完整群組分配，不能保證逐影片精確比例 |
| metadata assignment預算 | 512個，預先固定；沒有使用模型預測、loss或accuracy |
| 平衡目標 | 影片總數比例誤差，加八類等權比例誤差 |
| class coverage | 三個集合都必須涵蓋八類 |
| 排名／平手 | SHA256固定排序；Fraction精確比較目標值，再以assignment SHA與trial index決勝 |

目標式為 `Σ_s |N_s/N-r_s| + (1/8)Σ_s,c |N_sc/N_c-r_s|`。這是中繼資料分層，不是512次模型實驗；沒有根據90%目標或任何測試成績挑選切分。完整規則在政策JSON中。

最終選到trial index 42（零起始）；512個候選均滿足八類覆蓋。只保留一份輸出，不因訓練分數改seed或重挑切分。

| 集合 | 來源群組 | 影片數 | 占10,277支 |
| --- | ---: | ---: | ---: |
| Train | 40 | 8,230 | 80.0817% |
| Validation | 5 | 1,027 | 9.9932% |
| Test | 5 | 1,020 | 9.9251% |
| 隔離／排除 | — | 34 | 不計入比例 |

所有已納入來源群組只出現在單一集合；已知完全重複沒有多份保留；八類都有資料：

| 類別 | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| 2p0 | 1,133 | 144 | 137 |
| 2p1 | 1,571 | 199 | 170 |
| 3p0 | 1,701 | 222 | 206 |
| 3p1 | 944 | 108 | 123 |
| ft0 | 465 | 54 | 46 |
| ft1 | 1,353 | 174 | 189 |
| mp0 | 617 | 74 | 92 |
| mp1 | 446 | 52 | 57 |

來源群組仍是檔名代碼，不是獨立核實過的真實比賽ID。這是明示假設的來源分組重建，不把比例接近80/10/10當作取得了作者原清單。

## 3. 舊切分與候選切分的關係

| 舊集合 → 候選 | Train | Validation | Test | Excluded |
| --- | ---: | ---: | ---: | ---: |
| train | 5,797 | 375 | 1,020 | 1 |
| validation | 1,127 | 420 | 0 | 0 |
| test | 1,306 | 232 | 0 | 1 |
| excluded | 0 | 0 | 0 | 32 |

**候選Test的1,020支全部原屬舊Train。** 因此，任何曾在舊Train學習的checkpoint都不能拿來在這份Test宣稱泛化，也不能把舊四組成績直接當新八類對照。候選政策明確將舊checkpoint排除於新切分比較；未來若採用，只能由固定ImageNet等預先聲明初始化全新訓練。

同時，既有專案已接觸過test，重新分配同一資料不會恢復全專案未見狀態。這不表示新模型一定看過其候選Test，而是必須如實區分「對新模型的訓練集合隔離」與「研究流程從未接觸」。若需驗收新比賽90%泛化，仍需另外保留新的、未參與開發的來源；本輪沒有引入或下載新資料。

## 4. 完整性、重做方式與啟用界線

候選CSV仍保留全部10,311筆，每筆含original_split、original_excluded與原排除原因；新增cohort_status、cohort_exclusion_reason、class_id及candidate_split。原original_class與binary_label保持一致，不搬動或複製MP4。

完整metadata位於 `development/aca_eight_class_data_v1/artifacts/sourcegroup80_seed20261108_v1/`，由Git忽略；小型政策、摘要及驗收會保存於研究文件。來源SHA不符即拒絕；既有輸出目錄拒絕覆寫。

```bash
# 本機已有候選時：唯讀重做驗收
python3 scripts/build_aca_eight_class_candidate.py --verify development/aca_eight_class_data_v1/artifacts/sourcegroup80_seed20261108_v1

# 產物不存在時，或使用新的直接子目錄重建；不會啟動訓練
python3 scripts/build_aca_eight_class_candidate.py --output development/aca_eight_class_data_v1/artifacts/sourcegroup80_seed20261108_v1
```

已驗證：10,311筆可重做manifest、相同類別標籤、34筆排除、群組隔離、八類覆蓋、舊新切分交叉表與SHA。另核對輸入順序反轉結果不變、拒絕覆寫、拒絕遭改動的CSV及來源。這些是metadata驗收，不是模型accuracy或test成績。

目前沒有loader／trainer讀取候選CSV，政策`activate_existing_trainer=false`、`training_authorized=false`。舊資料指紋、模型程式、權重、Adam/RNG及排程不變；新模型、GPU工作、optimizer更新、影片開啟及test均為0。

## 5. 下一步收斂

| 項目 | 目前狀態 |
| --- | --- |
| D1 八類品質規則 | 已固定，候選中完整隔離已知同內容異類群組；未聲稱真值裁決 |
| D2 切分 | 一份固定規則候選已產出並驗收；作者原IDs／seed仍未取得 |
| D3 前處理 | 下一步定稿FPS、crop、取幀及train/eval規則，缺證即明示假設；不開多組搜尋 |
| D4 八類工程 | D3定稿後新增隔離adapter，驗證8類CE、8×8矩陣、二類top-1映射與資料邊界 |
| G1作者精確架構 | groups／BN與參數量差距仍未核實，不因完成split而消失 |
| 正式訓練／多seed／SpaceJam | 維持暫停；不啟用新候選或續跑八組舊佇列 |

下一個具體產物為D3的一份可重做前處理規格，再做D4隔離工程驗證；不再反覆處理相同兩組標籤，也不把新切分當作提升分數的方法。
