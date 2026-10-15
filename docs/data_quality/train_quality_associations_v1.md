# Train-only v3.1 Candidate／Sampler 資料品質 Audit v1

## Executive summary

本 audit 精確覆蓋固定 train **7,193** 支影片，只讀 manifest、v3.1 sidecar、sampler config 與 epoch=0 derived selections；影片 open/probe/decode 均為 **0**。研究任務仍是 **full-video Make/Miss classification**，不是 pre-release prediction，也不宣稱出手前預測能力。

- Candidate 與 label：有清楚的描述性關聯：Make 的 mean candidate ratio 高 1.24 個百分點、affected rate 高 15.53 個百分點，Cliff’s delta=0.162；rank effect 偏小，且 source-cluster CI 必須一併解讀。
- 控制 source/shot：source_group×shot_type 內的 overlap-weighted 差異仍存在（mean 0.93、affected risk 12.29 個百分點），但較未調整差異縮小；這是描述性調整，不是因果估計。
- Sampler 時間位置：epoch 0 有小幅前移（Make−Miss selected-center=-0.0116，95% CI [-0.0187, -0.0041]）；20 個 deterministic epochs 的 quality-aware 差異範圍為 [-0.0117, 0.0046]，相對 unrestricted baseline 的平均 difference-in-differences=-0.0043。三項檢查均在 ±0.05 警示尺度內，未見實質 label 位移。
- DataLoader gate：不足以阻止整合。
- 下一輪：維持 v3.1 進入可觀測、train-only DataLoader 整合；另做 z=-2.5 derived sensitivity，並以人工標註校準 candidate 的語義。

固定 totals 已重驗：1,105,146 decoded frames（僅來自既有 metadata）、18,390 candidate frames、1,798 candidate segments、1,698 affected、5,495 unaffected。

## Candidate–label 結果

統計單位是影片；frame 不被當作獨立樣本。零膨脹以 affected risk difference、全分布 Cliff’s delta、per-video mean difference 與 affected-only mean difference共同描述。95% CI 是對 35 個 source groups 做 cluster bootstrap；不以 p-value 單獨下結論。

| Label | n | mean | median | P25 | P75 | P90 | P95 | affected count/rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Make | 3,772 | 2.25% | 0.00% | 0.00% | 3.89% | 8.90% | 10.96% | 1,169/30.99% |
| Miss | 3,421 | 1.01% | 0.00% | 0.00% | 0.00% | 4.79% | 7.53% | 529/15.46% |

| Effect (Make−Miss) | estimate [source-cluster 95% CI] |
| --- | ---: |
| Mean candidate ratio | 1.24% [0.97%, 1.51%] |
| Affected risk | 15.53% [12.19%, 19.04%] |
| Affected-only mean ratio | 0.71% [0.44%, 0.97%] |
| Cliff’s delta | 0.1622 [0.1276, 0.1979] |

Cliff’s delta 把零值 ties 保留在全分布中；正值表示隨機抽一支 Make 的 candidate ratio 高於 Miss 的機率較高。這些是描述性關聯，不代表 label 造成 candidate。

## Source／shot type／FPS 分層

### Source group

| Group | n Make/Miss | Mean Make/Miss | Mean Δ | Affected Make/Miss | Affected Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| v108 | 117/101 | 1.23%/0.87% | 0.35% | 14.53%/11.88% | 2.65% |
| v111 | 130/105 | 1.88%/0.53% | 1.35% | 28.46%/8.57% | 19.89% |
| v115 | 119/95 | 1.61%/2.07% | -0.46% | 27.73%/34.74% | -7.01% |
| v116 | 103/99 | 3.07%/0.69% | 2.38% | 36.89%/11.11% | 25.78% |
| v117 | 114/99 | 1.07%/0.28% | 0.79% | 16.67%/5.05% | 11.62% |
| v118 | 114/114 | 1.50%/0.41% | 1.08% | 20.18%/7.02% | 13.16% |
| v119 | 121/106 | 0.99%/0.61% | 0.38% | 15.70%/10.38% | 5.33% |
| v120 | 113/120 | 1.06%/0.74% | 0.32% | 14.16%/13.33% | 0.83% |
| v122 | 128/102 | 1.58%/0.54% | 1.04% | 21.88%/10.78% | 11.09% |
| v124 | 104/96 | 0.78%/0.48% | 0.31% | 12.50%/8.33% | 4.17% |
| v127 | 108/106 | 1.76%/0.62% | 1.14% | 24.07%/7.55% | 16.53% |
| v128 | 113/106 | 1.32%/0.97% | 0.35% | 18.58%/15.09% | 3.49% |
| v130 | 22/19 | 1.43%/2.09% | -0.66% | 31.82%/31.58% | 0.24% |
| v131 | 106/108 | 1.68%/0.74% | 0.94% | 22.64%/10.19% | 12.46% |
| v133 | 105/74 | 3.05%/1.08% | 1.97% | 40.00%/13.51% | 26.49% |
| v145 | 82/84 | 0.95%/0.72% | 0.23% | 15.85%/13.10% | 2.76% |
| v147 | 97/86 | 2.36%/1.06% | 1.30% | 32.99%/17.44% | 15.55% |
| v150 | 93/66 | 2.26%/0.78% | 1.48% | 29.03%/12.12% | 16.91% |
| v155 | 86/105 | 3.33%/1.75% | 1.58% | 47.67%/27.62% | 20.06% |
| v156 | 127/103 | 3.37%/1.30% | 2.06% | 48.03%/18.45% | 29.58% |
| v157 | 101/103 | 3.51%/1.18% | 2.32% | 43.56%/16.50% | 27.06% |
| v158 | 92/88 | 0.39%/0.41% | -0.02% | 7.61%/6.82% | 0.79% |
| v159 | 104/97 | 3.72%/2.02% | 1.70% | 52.88%/24.74% | 28.14% |
| v160_v166 | 198/141 | 1.72%/1.24% | 0.48% | 24.24%/18.44% | 5.80% |
| v161 | 111/93 | 1.46%/0.45% | 1.01% | 20.72%/10.75% | 9.97% |
| v163 | 88/78 | 3.00%/1.42% | 1.57% | 42.05%/24.36% | 17.69% |
| v164 | 100/70 | 3.36%/1.29% | 2.07% | 44.00%/18.57% | 25.43% |
| v167 | 117/101 | 1.84%/0.97% | 0.87% | 28.21%/18.81% | 9.39% |
| v169 | 88/116 | 3.82%/0.79% | 3.03% | 51.14%/12.93% | 38.21% |
| v174 | 113/82 | 3.60%/1.66% | 1.94% | 46.90%/21.95% | 24.95% |
| v175 | 119/88 | 3.10%/0.86% | 2.24% | 37.82%/10.23% | 27.59% |
| v181 | 96/118 | 2.93%/1.35% | 1.58% | 43.75%/20.34% | 23.41% |
| v182 | 124/104 | 4.18%/2.05% | 2.13% | 51.61%/26.92% | 24.69% |
| v183 | 116/128 | 3.31%/1.39% | 1.92% | 43.97%/21.09% | 22.87% |
| v184 | 103/120 | 2.85%/0.99% | 1.86% | 39.81%/14.17% | 25.64% |

### Shot type

| Group | n Make/Miss | Mean Make/Miss | Mean Δ | Affected Make/Miss | Affected Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2p | 1386/1033 | 2.05%/1.02% | 1.04% | 29.00%/16.17% | 12.84% |
| 3p | 818/1440 | 2.01%/0.63% | 1.39% | 29.71%/10.49% | 19.22% |
| ft | 1175/400 | 2.88%/2.98% | -0.10% | 36.68%/37.75% | -1.07% |
| mp | 393/548 | 1.57%/0.59% | 0.98% | 23.66%/10.95% | 12.72% |

### FPS

| Group | n Make/Miss | Mean Make/Miss | Mean Δ | Affected Make/Miss | Affected Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2997/100 | 862/761 | 2.39%/0.88% | 1.51% | 32.02%/11.96% | 20.06% |
| 2997/125 | 2910/2660 | 2.21%/1.05% | 1.16% | 30.69%/16.47% | 14.22% |

### Source group × shot type 調整

在同一 source_group×shot_type 且兩個 label 都存在的 138 個 cells，以 `n_make*n_miss/(n_make+n_miss)` overlap weight 標準化。調整後 candidate-ratio mean difference 為 **0.93% [0.60%, 1.26%]**，affected-risk difference 為 **12.29% [7.71%, 16.74%]**。這是受共同支撐範圍限制的描述性調整，不能解釋為因果效果。

## Candidate 時間位置

每段使用 `center=(start+end)/(2*frame_count)`、`length=(end-start)/frame_count`；center 依三等分標記前／中／後。Segment 表只做描述，因同一影片可有多段；label effect 先在每支 affected video 內聚合，再以影片為單位、source-group cluster bootstrap。

| Label | affected videos | segments | center mean/median | length mean/median | front/middle/rear |
| --- | ---: | ---: | ---: | ---: | ---: |
| Make | 1,169 | 1,231 | 0.7017/0.9178 | 6.90%/6.85% | 25.91%/3.57%/70.51% |
| Miss | 529 | 567 | 0.6874/0.9384 | 6.12%/5.52% | 26.63%/6.17%/67.20% |

Affected-video 平均 segment center 的 Make−Miss 差為 **0.0111 [-0.0517, 0.0626]**；每影片 rear-share 差為 **3.06% [-4.69%, 9.65%]**。candidate segments 後段占比高於前段；需連同 source/shot/FPS 分層解讀。

### Source group

| Group | affected videos Make/Miss | segments Make/Miss | Mean center Make/Miss | Rear rate Make/Miss |
| --- | ---: | ---: | ---: | ---: |
| v108 | 17/12 | 17/15 | 0.4718/0.4418 | 41.18%/40.00% |
| v111 | 37/9 | 41/9 | 0.7112/0.9377 | 68.29%/100.00% |
| v115 | 33/33 | 37/39 | 0.4464/0.4541 | 32.43%/33.33% |
| v116 | 38/11 | 41/11 | 0.7879/0.8689 | 78.05%/90.91% |
| v117 | 19/5 | 20/5 | 0.5962/0.5096 | 55.00%/40.00% |
| v118 | 23/8 | 23/8 | 0.2408/0.9448 | 17.39%/100.00% |
| v119 | 19/11 | 19/11 | 0.4083/0.7902 | 36.84%/81.82% |
| v120 | 16/16 | 17/16 | 0.3987/0.7100 | 35.29%/68.75% |
| v122 | 28/11 | 29/11 | 0.3444/0.8801 | 31.03%/90.91% |
| v124 | 13/8 | 13/8 | 0.3480/0.5372 | 30.77%/50.00% |
| v127 | 26/8 | 26/10 | 0.8454/0.4769 | 88.46%/40.00% |
| v128 | 21/16 | 23/16 | 0.5399/0.8356 | 52.17%/87.50% |
| v130 | 7/6 | 7/6 | 0.8679/0.5571 | 85.71%/50.00% |
| v131 | 24/11 | 25/11 | 0.8070/0.8419 | 84.00%/81.82% |
| v133 | 42/10 | 45/11 | 0.6837/0.5327 | 68.89%/54.55% |
| v145 | 13/11 | 13/11 | 0.7376/0.9166 | 69.23%/90.91% |
| v147 | 32/15 | 32/15 | 0.7042/0.5240 | 75.00%/53.33% |
| v150 | 27/8 | 29/9 | 0.6659/0.6488 | 68.97%/66.67% |
| v155 | 41/29 | 45/29 | 0.7971/0.8056 | 84.44%/86.21% |
| v156 | 61/19 | 64/20 | 0.7999/0.8140 | 84.38%/80.00% |
| v157 | 44/17 | 45/18 | 0.7981/0.7043 | 82.22%/66.67% |
| v158 | 7/6 | 7/6 | 0.6528/0.5838 | 57.14%/66.67% |
| v159 | 55/24 | 60/27 | 0.6588/0.6716 | 70.00%/66.67% |
| v160_v166 | 48/26 | 53/29 | 0.5849/0.5705 | 49.06%/51.72% |
| v161 | 23/10 | 25/10 | 0.6175/0.8997 | 60.00%/90.00% |
| v163 | 37/19 | 40/20 | 0.8689/0.6839 | 90.00%/60.00% |
| v164 | 44/13 | 44/13 | 0.7897/0.7579 | 81.82%/76.92% |
| v167 | 33/19 | 34/19 | 0.7920/0.8019 | 82.35%/78.95% |
| v169 | 45/15 | 47/16 | 0.8038/0.7894 | 80.85%/81.25% |
| v174 | 53/18 | 55/21 | 0.8100/0.6848 | 85.45%/66.67% |
| v175 | 45/9 | 47/11 | 0.8566/0.6742 | 89.36%/63.64% |
| v181 | 42/24 | 43/28 | 0.6972/0.6452 | 67.44%/60.71% |
| v182 | 64/28 | 69/34 | 0.7939/0.6321 | 82.61%/58.82% |
| v183 | 51/27 | 55/27 | 0.7522/0.7012 | 78.18%/74.07% |
| v184 | 41/17 | 41/17 | 0.7277/0.6884 | 73.17%/70.59% |

### Shot type

| Group | affected videos Make/Miss | segments Make/Miss | Mean center Make/Miss | Rear rate Make/Miss |
| --- | ---: | ---: | ---: | ---: |
| 2p | 402/167 | 412/171 | 0.8700/0.8134 | 91.75%/83.04% |
| 3p | 243/151 | 248/159 | 0.8496/0.7494 | 89.11%/74.21% |
| ft | 431/151 | 478/176 | 0.4433/0.4601 | 38.08%/39.77% |
| mp | 93/60 | 93/61 | 0.8901/0.8280 | 93.55%/83.61% |

### FPS

| Group | affected videos Make/Miss | segments Make/Miss | Mean center Make/Miss | Rear rate Make/Miss |
| --- | ---: | ---: | ---: | ---: |
| 2997/100 | 276/91 | 291/98 | 0.7654/0.7238 | 78.35%/72.45% |
| 2997/125 | 893/438 | 940/469 | 0.6821/0.6797 | 68.09%/66.10% |

## Sampler 選取位置

`normalized_start = start_frame / (frame_count - clip_length)`；selected center 使用 `(start+end)/(2*frame_count)`。Sampler 本身不讀 label，但本節直接檢查輸出分布。

| Label | n | start mean/median | end mean/median | normalized start mean/median | center mean/median |
| --- | ---: | ---: | ---: | ---: | ---: |
| Make | 3,772 | 43.62/42.00 | 107.62/106.00 | 0.4861/0.4821 | 0.4919/0.4889 |
| Miss | 3,421 | 45.32/45.00 | 109.32/109.00 | 0.5059/0.5122 | 0.5035/0.5068 |

Epoch 0 的 Make−Miss normalized-start 差為 **-0.0199 [-0.0325, -0.0071]**；selected-center 差為 **-0.0116 [-0.0187, -0.0041]**。以 ±0.05 個影片長度作為整合前的實質偏移警示尺度，epoch 0 的 source-cluster CI 完整落在警示區間內。

### Source group

| Group | n Make/Miss | Mean normalized start Make/Miss | Mean center Make/Miss | Center Δ |
| --- | ---: | ---: | ---: | ---: |
| v108 | 117/101 | 0.4907/0.5517 | 0.4948/0.5290 | -0.0342 |
| v111 | 130/105 | 0.5038/0.5007 | 0.5025/0.5004 | 0.0021 |
| v115 | 119/95 | 0.4923/0.5480 | 0.4957/0.5270 | -0.0313 |
| v116 | 103/99 | 0.4904/0.4892 | 0.4938/0.4930 | 0.0008 |
| v117 | 114/99 | 0.5029/0.5099 | 0.5016/0.5055 | -0.0039 |
| v118 | 114/114 | 0.5423/0.5309 | 0.5237/0.5174 | 0.0064 |
| v119 | 121/106 | 0.5541/0.4589 | 0.5304/0.4769 | 0.0535 |
| v120 | 113/120 | 0.4715/0.4962 | 0.4840/0.4979 | -0.0139 |
| v122 | 128/102 | 0.4950/0.5129 | 0.4972/0.5073 | -0.0100 |
| v124 | 104/96 | 0.4879/0.5090 | 0.4932/0.5051 | -0.0118 |
| v127 | 108/106 | 0.4967/0.5192 | 0.4979/0.5124 | -0.0145 |
| v128 | 113/106 | 0.4439/0.4895 | 0.4685/0.4941 | -0.0256 |
| v130 | 22/19 | 0.5127/0.4994 | 0.5072/0.4996 | 0.0075 |
| v131 | 106/108 | 0.4552/0.5084 | 0.4711/0.5054 | -0.0343 |
| v133 | 105/74 | 0.4707/0.5114 | 0.4836/0.5064 | -0.0228 |
| v145 | 82/84 | 0.4795/0.4961 | 0.4885/0.4978 | -0.0093 |
| v147 | 97/86 | 0.4386/0.4814 | 0.4655/0.4896 | -0.0240 |
| v150 | 93/66 | 0.5269/0.5248 | 0.5151/0.5139 | 0.0012 |
| v155 | 86/105 | 0.4708/0.4841 | 0.4836/0.4911 | -0.0075 |
| v156 | 127/103 | 0.4505/0.5420 | 0.4722/0.5236 | -0.0514 |
| v157 | 101/103 | 0.4655/0.4959 | 0.4806/0.4977 | -0.0171 |
| v158 | 92/88 | 0.5148/0.4847 | 0.5095/0.4902 | 0.0194 |
| v159 | 104/97 | 0.4595/0.5156 | 0.4739/0.5101 | -0.0362 |
| v160_v166 | 198/141 | 0.5065/0.4833 | 0.5036/0.4906 | 0.0130 |
| v161 | 111/93 | 0.4718/0.5414 | 0.4841/0.5233 | -0.0391 |
| v163 | 88/78 | 0.4547/0.5145 | 0.4745/0.5082 | -0.0336 |
| v164 | 100/70 | 0.4783/0.5548 | 0.4860/0.5353 | -0.0493 |
| v167 | 117/101 | 0.5091/0.4816 | 0.5051/0.4897 | 0.0154 |
| v169 | 88/116 | 0.4775/0.4905 | 0.4874/0.4947 | -0.0073 |
| v174 | 113/82 | 0.4889/0.4877 | 0.4938/0.4931 | 0.0007 |
| v175 | 119/88 | 0.4915/0.5069 | 0.4945/0.5044 | -0.0099 |
| v181 | 96/118 | 0.4735/0.4983 | 0.4852/0.4991 | -0.0139 |
| v182 | 124/104 | 0.4647/0.5269 | 0.4802/0.5151 | -0.0349 |
| v183 | 116/128 | 0.4699/0.5054 | 0.4831/0.5031 | -0.0199 |
| v184 | 103/120 | 0.4943/0.4876 | 0.4968/0.4930 | 0.0038 |

### Shot type

| Group | n Make/Miss | Mean normalized start Make/Miss | Mean center Make/Miss | Center Δ |
| --- | ---: | ---: | ---: | ---: |
| 2p | 1386/1033 | 0.4752/0.5016 | 0.4860/0.5011 | -0.0150 |
| 3p | 818/1440 | 0.4668/0.5027 | 0.4808/0.5018 | -0.0210 |
| ft | 1175/400 | 0.5107/0.5161 | 0.5057/0.5087 | -0.0030 |
| mp | 393/548 | 0.4909/0.5152 | 0.4946/0.5088 | -0.0142 |

### FPS

| Group | n Make/Miss | Mean normalized start Make/Miss | Mean center Make/Miss | Center Δ |
| --- | ---: | ---: | ---: | ---: |
| 2997/100 | 862/761 | 0.4865/0.5086 | 0.4913/0.5055 | -0.0142 |
| 2997/125 | 2910/2660 | 0.4859/0.5052 | 0.4921/0.5029 | -0.0108 |

### Multi-epoch 反事實敏感度

以相同 video、seed、epoch 與 hash 規則重跑 **20 個 deterministic epochs**。Quality-aware 使用正式 candidate segments；unrestricted baseline 傳入空 segments，因此每影片 `quality-aware center − unrestricted center` 隔離的是候選窗口限制造成的位移。這是 deterministic sensitivity，不把 epochs 當成獨立樣本。

| 指標 | mean across epochs | min | max |
| --- | ---: | ---: | ---: |
| Quality-aware Make−Miss center | -0.0062 | -0.0117 | 0.0046 |
| Unrestricted Make−Miss center | -0.0019 | -0.0084 | 0.0107 |
| Quality-induced difference-in-differences | -0.0043 | -0.0061 | -0.0019 |

Gate 同時要求：(1) epoch 0 source-cluster CI、(2) 全部 20 個 quality-aware label shifts、(3) 全部 quality-induced difference-in-differences 都位於 ±0.05 內。三項目前均通過。

## Fallback 結果

驗收值完全一致：**7,188** 支 zero-overlap、**5** 支 fallback、fallback 總 overlap **20 frames**。

| video_id | label | shot | source | start:end | overlap |
| --- | --- | --- | --- | ---: | ---: |
| `ft0/ft0_v166_002702_x264.mp4` | Miss | ft | v160_v166 | 0:64 | 1 |
| `ft0/ft0_v181_002018_x264.mp4` | Miss | ft | v181 | 65:129 | 8 |
| `ft1/ft1_v111_005842_x264.mp4` | Make | ft | v111 | 22:86 | 5 |
| `ft1/ft1_v133_011206_x264.mp4` | Make | ft | v133 | 0:64 | 5 |
| `ft1/ft1_v169_004222_x264.mp4` | Make | ft | v169 | 82:146 | 1 |

## Split 限制

50 個 source groups 在 train/validation/test 間沒有交叉，且 v160/v166 維持共同 `v160_v166` source group。本 split 只能明確稱為 **source-group holdout (not confirmed game-level holdout)**；沒有足夠證據稱為已確認 game-level holdout。本輪沒有做影片內容 near-duplicate scan，亦沒有讀取 validation/test 影片。

## 對下一階段 DataLoader 整合的建議

1. 可讓 v3.1 quality-aware sampler 進入 **train-only、可觀測且可回退** 的 DataLoader 整合；保留原始連續 64-frame window，不變更 label、split 或影片。
2. 每個 sample/log 必須保留 `video_id`、start/end、normalized start、candidate overlap、fallback、source、shot type、FPS、policy/run identity；每個 epoch 監控 label/source/shot 的位置分布。
3. validation/test 維持 quality mode disabled；本 audit 不授權把 train sidecar 規則套到 held-out data。
4. 另做 z=-2.5 的 **derived sensitivity**，比較關聯與選窗穩定度，但不要在本輪重調 v3.1 門檻。
5. 建立小型、按 source/shot/label/時間位置分層的人工標註集，校準 candidate 是否真的對應可見模糊；人工標註是語義驗證，不是刪除資料授權。

## 結論（必要問題）

- **Candidate 與 label 是否存在明顯描述性關聯？** 有清楚的描述性關聯：Make 的 mean candidate ratio 高 1.24 個百分點、affected rate 高 15.53 個百分點，Cliff’s delta=0.162；rank effect 偏小，且 source-cluster CI 必須一併解讀。
- **控制 source group 與 shot type 後是否仍存在？** source_group×shot_type 內的 overlap-weighted 差異仍存在（mean 0.93、affected risk 12.29 個百分點），但較未調整差異縮小；這是描述性調整，不是因果估計。
- **Sampler 選中窗口是否因 label 實質偏移？** epoch 0 有小幅前移（Make−Miss selected-center=-0.0116，95% CI [-0.0187, -0.0041]）；20 個 deterministic epochs 的 quality-aware 差異範圍為 [-0.0117, 0.0046]，相對 unrestricted baseline 的平均 difference-in-differences=-0.0043。三項檢查均在 ±0.05 警示尺度內，未見實質 label 位移。
- **是否足以阻止 DataLoader 整合？** 否；但整合必須 train-only 且持續記錄診斷欄位。
- **下一輪？** 維持 v3.1 進入可觀測、train-only DataLoader 整合；另做 z=-2.5 derived sensitivity，並以人工標註校準 candidate 的語義。

## 重現與限制

```bash
python3 -B development/data_quality/scripts/audit_train_quality_associations_v1.py --check-only
python3 -B development/data_quality/scripts/audit_train_quality_associations_v1.py --refresh-artifacts
python3 -B -m unittest development/data_quality/tests/test_train_quality_associations_v1.py
```

完整 machine summary 見 `train_quality_associations_v1.json`；中間 per-video／per-segment CSV 位於 ignored `development/data_quality/artifacts/train_quality_associations_v1/`。乾淨 checkout 可直接執行 `--check-only`；ignored CSV 不存在時會回報為 optional missing，不影響 tracked report 驗證，需要實體 CSV 時再用 `--refresh-artifacts` 重建。Audit 不訓練模型、不評估 test、不做 augmentation、不改門檻，也不開啟、probe 或 decode 影片。關聯都是描述性；candidate 不是人工確認的 blur。
