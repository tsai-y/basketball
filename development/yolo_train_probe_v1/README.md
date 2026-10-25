# YOLO 五張 small-overfit 工程驗證

這不是正式偵測效能實驗。只匯出原始 train 中已完成三類整張人工確認的影像，
排除 pending／uncertain／AI-only 草稿；持球角色不作第四類。

目前五張共 46 個 active_player、5 個 ball、5 個 rim，來源為4個 source_group。
`train` 與 `val` **刻意使用同一批影像**，只檢查能否記住訓練樣本。
任何 Ultralytics precision／recall／mAP 都是訓練影像重代入診斷，不能當泛化成績。
原始 validation/test 不讀、不改。這裡的 `val` 不是原始分類資料的 validation。

程式會驗證標註協定、整張人工確認事件、SHA、原始 train 身分、類別與框，
另存不可變標註快照，再產生標準 normalized xywh labels 與 data.yaml。
原圖保持320×240、位元相同；YOLO 訓練時 letterbox 到640。
原有人工JSON、AI草稿、MixNet-I3D、特徵快取與分類器都不改動。

流程：`probe --prepare` → `probe --check` → `probe --train --epochs 1 --smoke`
→ 確認 forward/backward、參數更新、checkpoint reload → `probe --train --epochs 150`。
請以 `python -B -m development.yolo_train_probe_v1.probe --help` 查看完整參數。

使用既有本機 yolo11m.pt 初始化三類模型，AdamW lr=0.001，batch=2/nbs=2，
停用增強、暖身、AMP與early stopping以隔離少量樣本記憶能力；150 epoch 固定上限。
smoke 與 overfit 使用獨立輸出，overfit 從原始COCO權重重新開始，不續接smoke。
每 epoch 記錄 loss、首尾參數更新及GPU峰值，訓練後驗證 last.pt 重新載入與推論。
只在明確 `--train` 時訓練；沒有CUDA即失敗，不會默默改跑CPU。

匯出五張並不代表可以建立可靠的偵測器。後續仍需更多完整標註、來源分離的
detector train/validation 與獨立評估，才可決定是否用於分類模型前處理。

## 首次實測結果

2026-10-25 已完成 1 epoch smoke 與獨立的150 epoch overfit，兩者皆使用GPU。
輸出位於本目錄 `artifacts/human5_overfit_dataset_f7opv7v8/`；
overfit 執行目錄為 `overfit_okmt1g7m/`，最終固定150 epoch權重在
`fit/weights/last.pt`，模型重新載入與同五張推論通過。

固定 conf=0.25、同類IoU≥0.5的一對一訓練圖比對：球員46/46、球5/5、
籃圈5/5，無額外框。這是重代入記憶測試，不是獨立驗證precision/recall，
也不是正式mAP成果。`training_image_visual_check__p_4vgsl/index.html`
保留逐張人工標註／預測並排圖與機器可讀診斷。

可使用 `check_predictions --dataset-dir ... --run-dir ...` 重做工程比對。
不可用同批影像的高分選定正式模型；仍需來源分離的偵測器驗證資料。

## 來源分離的完整影片推論診斷

`unseen_video_audit --dry-run/--run` 使用上述固定150 epoch的 `last.pt`，
從既有20支原始train診斷影片排除全部4個訓練來源群組，留下16支。
這些影片未參與五張圖的權重訓練，但曾用於開發檢查，**不是未接觸過的正式驗證集**。
執行前驗證資料、來源、標註快照和權重SHA；原片逐幀唯讀解碼，非train或
訓練來源在開檔前拒絕。可用 `python -B -m development.yolo_train_probe_v1.unseen_video_audit --help`
查看參數；只有 `--run` 會載入模型與輸出檔案，CUDA不可用時失敗。

每支影片輸出原片／預測並排影片、逐幀原座標JSON、固定五位置無損PNG、
可視化索引與尚未填寫的人工複查JSON。P/B/R為逐幀框號，並非追蹤ID。
不追加訓練、不追蹤、不裁切、不刪片、不更動分類特徵；原始validation/test未讀。

2026-10-25完成 `artifacts/unseen_video_audit_0mon4xya/`：16支、2,506幀，
其中2,264幀有人框、7幀有球框、1,598幀有籃圈框。
這些是偵測存在次數，**不是recall或偵測準確率**；沒有完整逐物件真值。
AI檢查四張概覽共48個固定幀及全部7個球候選的診斷截圖，並非完整影片人工複查：
可見球員漏框、轉場標誌誤認球員，且至少5個球候選框到場邊頭部。
因此這版尚不適合批次裁切／剔除分類片段；先擴充完整標註和來源分離的
detector train/validation。AI初審另存，不能冒充人工真值或自動匯入訓練。

## 下一批標註候選（未匯出訓練）

`prepare_expansion_v1 --dry-run/--prepare` 從上述診斷的原生無損PNG各取2張，共32張；
不重開原片、不跑YOLO、不訓練。固定seed的source-group雜湊先預留每種投籃類型
1個來源，共4個來源／8張，作偵測器的**開發驗證候選**；其餘12個來源／24張作
訓練候選。先前五張訓練圖的4個來源只能留在訓練。未來匯出必須遵守
`split_plan.json`，不可將保留來源的其他舊／新影像加進訓練。

全部影像仍來自原始分類train；分類split不變。來源曾被診斷檢視，並非正式未接觸
的test。8張也只是小型開發驗證候選，不足以作穩定泛化結論。驗證候選固定30/70%
位置；部分訓練候選選已知漏框／轉場案例，需披露選樣偏差。

初始輸出 `artifacts/annotation_expansion_v1_ngbs2tlk/`；人工JSON初始全pending，AI初篩與
E030框草稿在 `ai/` 另存。轉場與特寫依現有協定不能直接當空標籤負例。
球看不清須留uncertain，不可當absent；三類物件均完整確認後才可考慮匯出。
此程式沒有YOLO labels/data.yaml/訓練匯出功能；不能拿舊small-overfit程式直接接本批。

## 縮減標註後的來源分離小型訓練

`train_expansion_v1` 使用已完成9張擴充人工標註中的6張訓練／3張開發驗證，
再併入舊5張訓練圖，合計11 train／3 development-validation。E014等球不確定
與AI-only影像不匯出。兩組來源、影片及影像SHA必須分離，原始分類validation/test不讀。
核對整張人工確認、當時框草稿與顯示圖SHA後，才複製原生PNG並匯出標籤。

先 `--prepare --batch ... --previous-dataset ...`，再 `--check --dataset-dir ...`，
接著 `--train --smoke --epochs 1 --dataset-dir ...`；通過後用獨立run執行
`--train --epochs 100 --dataset-dir ...`。兩輪都從原始本機COCO `yolo11m.pt`
重新初始化，不續接舊5張overfit或smoke權重。

GPU、640輸入、batch/nbs=2、AdamW lr=1e-4、weight decay=5e-4，停用資料增強
與AMP；最多100 epochs、patience=20。bias warmup明確設為1e-4，避免套件較大的預設值。
記錄實際loader allowlist、有限loss、參數更新、best/last權重及重載結果。
最後最佳權重驗證的額外callback不計入訓練epoch。模型選擇沿用Ultralytics開發驗證fitness。

僅3張驗證影像、2個來源且經開發檢視，分數非常不穩定，不能作正式泛化結論；
仍缺部分投籃類型與來源。不自動套用全影片裁切、篩片或MixNet-I3D特徵重算。

## 新舊權重固定幀對照

`visual_check_expansion_v1 --dataset-dir ... --training-run ... --audit-run ... [--dry-run]`
比較新run的best.pt與原5張overfit的last.pt。開發驗證3張有人工標註；額外來源須
同時排除本輪train和development-validation來源，沿用既有原生無損PNG的
10/30/50/70/90%五位置，不重開原始影片。此批為10個額外來源、50張，加驗證共53張。

固定conf0.25/NMS IoU0.7/max_det100，輸出左原圖或人工標註／中舊權重／右新權重。
只有3張可作同類IoU≥0.5的貪婪一對一框匹配，不是AP；其餘沒有完整真值，
檢出幀數不能當recall。來源都曾用於開發診斷，不是未接觸的正式test。
`report.json`保留全部預測，`AI_review.json/html`另列AI實際看過的範圍與不確定性，
不冒充人工標註、不自動追加訓練。本次視覺診斷相關測試合計41項通過。

## 原始COCO球偵測／三類初始化診斷

`ball_pretraining_diagnostic_v1 --dataset-dir ... --training-run ... --prior-visual ... [--dry-run]`
沿用前項完全相同53張影像與固定參數；原始COCO權重只取依類別名稱動態查得的
`sports ball`，自訓模型只取`ball`，統一映射為球供比對。COCO類別不是籃球專用。
只有3張人工標註開發影像可算固定IoU匹配，其餘50張有框次數不是recall；
不重新訓練、不調閾值／解析度、不修改標註或MixNet特徵。

模型推論fuse前先核對原始／自訓state key與shape，另存`initialization_audit.json`。
只依形狀相容及本機Ultralytics載入程式解釋可轉移範圍，不冒稱重播過原始初始化；
分類輸出最後6個weight/bias不相容不等於整個偵測頭重置，也不證明唯一退化原因。
完整候選對照與AI實際檢查範圍另存，診斷相關測試合計46項通過。

## 640／1280配對輸入尺度診斷

`ball_input_scale_v1 --dataset-dir ... --training-run ... --reference ... [--dry-run]`
沿用前述53張／兩個模型，唯一操作因素為imgsz640或1280，固定conf0.25、NMS
IoU0.7、max_det100與FP32。640控制組須重現上輪球框；模型pre-hook記錄實際
送入網路的tensor尺寸，不能只依設定名稱假定已放大。共212次推論，沒有訓練。

每圖輸出COCO／自訓兩排，左原圖或人工球框、中640、右1280；全部球候選另產生
原生像素的最近鄰局部檢查圖，裁圖只用於顯示，並非模型輸入或全片ROI處理。
放大無法還原原片細節，50張未標註圖的檢出數仍不是recall。相關測試合計51項通過。

## 球專用、保留原始 checkpoint 結構的獨立分支

`ball_only_probe_v1` 沿用既有11 train／3 development影像及人工球框，僅將球標籤
從原三類class1映射成單類class0。沒有增加獨立資料、AI-only標籤或假造空標籤負例；
不更動原三類模型、人工標註、來源切分與MixNet特徵。這是小型診斷，不能部署全片篩選。

保留原始COCO checkpoint的實際模組結構，將sports ball分類輸出的6個weight/bias
切片映射到單類，其餘643個state tensors直接複製。初始化先比較NMS前原始box與
球分數，再驗證checkpoint重載。單類的類別競爭不同，raw score一致不保證NMS結果一致。

本機發現Ultralytics8.4.58從YAML重建SPPF時，cv1 activation為Identity，原checkpoint
則為SiLU；只核對state shape不足以保證前向等價。故本分支不從YAML重建，也不據此
宣稱先前所有漏球由此造成。初次準備被輸出一致性檢查攔下，失敗目錄保留稽核紀錄。

訓練只開啟最後Detect模組，其前所有層（含neck）參數與BN buffers均凍結。
自訂trainer在基底`_model_train()`之後強制其eval，避免epoch-start callback之後
被重新切回train。每batch檢查模式，結束時逐項核對凍結state完全相同。
試跑失敗不續接；通過獨立1 epoch smoke後，完整run仍從相同單類初始化重新開始。

用法：`--prepare --source-dataset ...`、`--check --dataset-dir ...`、
`--train --smoke --epochs 1 --dataset-dir ...`、`--train --epochs 100 --dataset-dir ...`，
最後`--evaluate --dataset-dir ... --training-run ... --reference ...`。訓練沿用640、
batch/nbs2、AdamW lr1e-4、無增強／無AMP、patience20；記錄原始碼快照及權重SHA。
評估固定相同53張、640/conf0.25/NMS IoU0.7，只有3張可作IoU0.5匹配。
其餘50張有框次數不是recall，AI初審不自動轉人工真值。相關單元測試合計58項通過。

### 2026-10-25 暫停／分支保存點

球專用分支已完成獨立1 epoch smoke與完整run；完整run第31 epoch早停、最佳第11 epoch。
固定53張、640/conf0.25的前後對照中，3個人工開發球框由0/3匹配變2/3，
仍有1個漏框及1個多餘框。額外50張有8個候選／7幀；AI看過全部候選及3張開發圖，
可見背景／手部誤報，不能據此估正式泛化precision/recall，尚不採用於全影片篩選。
同時改了單類輸出、保留checkpoint結構及凍結策略，不能將改善歸因於單一因素。

本機run：`artifacts/ball_only_probe_2z5i9tal/headtrain_swnpp53j/`，
最佳權重SHA-256：`983e8a8492eed9bbb13ce64849699be28cb34f136cec16cacec3b88ce55a4ac2`。
初審在該run的`before_after_2h8giyb9/AI_review.html/json`，全部預測在`report.json`。
Git保存程式、測試、文件，不包含被忽略的權重、標註快照、影片及大型artifacts；
push程式分支不等於已備份這些本機實驗資料。原MixNet特徵及三類YOLO均未覆寫。
