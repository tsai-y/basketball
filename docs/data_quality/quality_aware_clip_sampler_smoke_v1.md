# Basketball-51 Quality-aware Clip Sampler v1 — metadata-only smoke

此固定 smoke 僅使用 train manifest 與品質 sidecar；candidate 是低局部 Laplacian 候選，不是模糊真值或排除名單。

## 結果

| Metric | Value |
| --- | --- |
| videos / clips | 20 / 20 |
| Make / Miss | 10 / 10 |
| sources | 20 |
| candidate-free clips | 15 |
| fallback clips / videos | 5 / 5 |
| total candidate overlap | 20 frames |
| clean availability 16 / 32 / 64 | 20 / 20 / 15 |
| video opens / probes / decodes | 0 / 0 / 0 |
| validation / test access | 0 / 0 |

五支沒有 64-frame 零重疊窗口的影片均使用 minimum-overlap fallback；其餘 15 支選到零 overlap 連續窗口。

## 固定選樣

| Video | Label | Source | Category | Selected [start,end) | Overlap | Fallback |
| --- | --- | --- | --- | --- | --- | --- |
| `2p0/2p0_v115_003225_x264.mp4` | Miss | v115 | affected_clean | [56,120) | 0 | false |
| `2p0/2p0_v116_012904_x264.mp4` | Miss | v116 | affected_clean | [72,136) | 0 | false |
| `2p0/2p0_v128_002950_x264.mp4` | Miss | v128 | affected_clean | [33,97) | 0 | false |
| `2p0/2p0_v174_000256_x264.mp4` | Miss | v174 | affected_clean | [66,130) | 0 | false |
| `2p1/2p1_v150_000903_x264.mp4` | Make | v150 | unaffected | [66,130) | 0 | false |
| `2p1/2p1_v163_001335_x264.mp4` | Make | v163 | unaffected | [0,64) | 0 | false |
| `2p1/2p1_v175_005517_x264.mp4` | Make | v175 | unaffected | [86,150) | 0 | false |
| `3p0/3p0_v108_003858_x264.mp4` | Miss | v108 | affected_clean | [78,142) | 0 | false |
| `3p1/3p1_v120_004108_x264.mp4` | Make | v120 | unaffected | [70,134) | 0 | false |
| `3p1/3p1_v122_012723_x264.mp4` | Make | v122 | unaffected | [14,78) | 0 | false |
| `ft0/ft0_v127_002040_x264.mp4` | Miss | v127 | affected_clean | [57,121) | 0 | false |
| `ft0/ft0_v159_011506_x264.mp4` | Miss | v159 | affected_clean | [113,177) | 0 | false |
| `ft0/ft0_v166_002702_x264.mp4` | Miss | v160_v166 | fallback | [0,64) | 1 | true |
| `ft0/ft0_v181_002018_x264.mp4` | Miss | v181 | fallback | [65,129) | 8 | true |
| `ft1/ft1_v111_005842_x264.mp4` | Make | v111 | fallback | [22,86) | 5 | true |
| `ft1/ft1_v131_001017_x264.mp4` | Make | v131 | unaffected | [48,112) | 0 | false |
| `ft1/ft1_v133_011206_x264.mp4` | Make | v133 | fallback | [0,64) | 5 | true |
| `ft1/ft1_v155_011026_x264.mp4` | Make | v155 | unaffected | [53,117) | 0 | false |
| `ft1/ft1_v169_004222_x264.mp4` | Make | v169 | fallback | [82,146) | 1 | true |
| `mp0/mp0_v184_000641_x264.mp4` | Miss | v184 | affected_clean | [64,128) | 0 | false |

## Timeline

以下只依 frame index 繪製，沒有擷取影片畫面。`█` 是 candidate，`[──]` 是所選連續 clip。

```text
## 2p0/2p0_v115_003225_x264.mp4 clip 0
Frames:     0──────────────────────────────────────────────────────────────146
Candidates:                  ███                                          █████████ 
Selected:                              [──────────────────────────────]             
Overlap: 0
Fallback: false

## 2p0/2p0_v116_012904_x264.mp4 clip 0
Frames:     0──────────────────────────────────────────────────────────────180
Candidates:                                                                      ███
Selected:                               [─────────────────────────]                 
Overlap: 0
Fallback: false

## 2p0/2p0_v128_002950_x264.mp4 clip 0
Frames:     0──────────────────────────────────────────────────────────────146
Candidates:                                                                    █████
Selected:                   [──────────────────────────────]                        
Overlap: 0
Fallback: false

## 2p0/2p0_v174_000256_x264.mp4 clip 0
Frames:     0──────────────────────────────────────────────────────────────146
Candidates:                                                                  ███████
Selected:                                   [──────────────────────────────]        
Overlap: 0
Fallback: false

## 2p1/2p1_v150_000903_x264.mp4 clip 0
Frames:     0──────────────────────────────────────────────────────────────146
Candidates:                                                                         
Selected:                                   [──────────────────────────────]        
Overlap: 0
Fallback: false

## 2p1/2p1_v163_001335_x264.mp4 clip 0
Frames:     0──────────────────────────────────────────────────────────────146
Candidates:                                                                         
Selected:   [──────────────────────────────]                                        
Overlap: 0
Fallback: false

## 2p1/2p1_v175_005517_x264.mp4 clip 0
Frames:     0──────────────────────────────────────────────────────────────180
Candidates:                                                                         
Selected:                                     [────────────────────────]            
Overlap: 0
Fallback: false

## 3p0/3p0_v108_003858_x264.mp4 clip 0
Frames:     0──────────────────────────────────────────────────────────────146
Candidates: ████                                                                    
Selected:                                         [──────────────────────────────]  
Overlap: 0
Fallback: false

## 3p1/3p1_v120_004108_x264.mp4 clip 0
Frames:     0──────────────────────────────────────────────────────────────146
Candidates:                                                                         
Selected:                                     [──────────────────────────────]      
Overlap: 0
Fallback: false

## 3p1/3p1_v122_012723_x264.mp4 clip 0
Frames:     0──────────────────────────────────────────────────────────────146
Candidates:                                                                         
Selected:         [──────────────────────────────]                                  
Overlap: 0
Fallback: false

## ft0/ft0_v127_002040_x264.mp4 clip 0
Frames:     0──────────────────────────────────────────────────────────────180
Candidates: ███                                                             ████████
Selected:                         [─────────────────────────]                       
Overlap: 0
Fallback: false

## ft0/ft0_v159_011506_x264.mp4 clip 0
Frames:     0──────────────────────────────────────────────────────────────180
Candidates:             ███                       █████                             
Selected:                                                [────────────────────────] 
Overlap: 0
Fallback: false

## ft0/ft0_v166_002702_x264.mp4 clip 0
Frames:     0──────────────────────────────────────────────────────────────146
Candidates:                                ██████████                 ██████        
Selected:   [──────────────────────────────]                                        
Overlap: 1
Fallback: true

## ft0/ft0_v181_002018_x264.mp4 clip 0
Frames:     0──────────────────────────────────────────────────────────────145
Candidates:     ███████               ████████                            ██████    
Selected:                                   [──────────────────────────────]        
Overlap: 8
Fallback: true

## ft1/ft1_v111_005842_x264.mp4 clip 0
Frames:     0──────────────────────────────────────────────────────────────180
Candidates:                      ██               █████████                 ████████
Selected:           [─────────────────────────]                                     
Overlap: 5
Fallback: true

## ft1/ft1_v131_001017_x264.mp4 clip 0
Frames:     0──────────────────────────────────────────────────────────────180
Candidates:                                                                         
Selected:                      [────────────────────────]                           
Overlap: 0
Fallback: false

## ft1/ft1_v133_011206_x264.mp4 clip 0
Frames:     0──────────────────────────────────────────────────────────────146
Candidates:                             ███   ██████                       █████████
Selected:   [──────────────────────────────]                                        
Overlap: 5
Fallback: true

## ft1/ft1_v155_011026_x264.mp4 clip 0
Frames:     0──────────────────────────────────────────────────────────────146
Candidates:                                                                         
Selected:                             [──────────────────────────────]              
Overlap: 0
Fallback: false

## ft1/ft1_v169_004222_x264.mp4 clip 0
Frames:     0──────────────────────────────────────────────────────────────146
Candidates:         █████                        ████                               
Selected:                                           [──────────────────────────────]
Overlap: 1
Fallback: true

## mp0/mp0_v184_000641_x264.mp4 clip 0
Frames:     0──────────────────────────────────────────────────────────────146
Candidates:                                                                      ███
Selected:                                  [──────────────────────────────]         
Overlap: 0
Fallback: false
```

## Provenance 與邊界

- policy: `quality_aware_sampler_v1`; seed=20261009; epoch=0; clip length=64。
- sidecar run identity: `19372422e554779a09d54ca93e4927bf91d5665af5e6c127904727d4b5edc02a`。
- sidecar config hash: `ef73aea0b9cdd6656d8d017a90552b8737207fd7ee58a925b7d0dca45e5f9e20`。
- smoke manifest SHA-256: `68a349645f92d89c400ba0e49e9a10e375e96ec4b9e568de1c8d263963cc3c04`。
- validation/test quality mode disabled；沒有 augmentation、I3D/YOLO、訓練、評估或資料刪除。
