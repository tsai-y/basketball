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

## 既有完整標註衍生的負例裁圖候選（未訓練）

`ball_negative_crops_v1 --prepare --source-ball-dataset ...` 從已凍結球專用資料版本
的11張detector-train各產生1張候選。先核對來源三類完整人工確認事件、快照與SHA，
再要求所有球框連同原生12像素邊界均不與裁圖交疊；不能以模型沒有預測球推定負例。
固定四角160×120候選、上方優先、距球較遠者優先；無安全區則用128×96固定四角，
仍無安全區就拒絕，不降低安全邊界。裁圖PNG保持原生像素，不做影像修補或生成。

這是由既有完整球標註推導的augmentation候選，不是新增人工逐張確認，也不是
新增獨立來源。依賴既有人工標註沒有漏掉可見球，AI視覺初審不會消除這個限制。
禁止從detector-development或原始分類validation/test產生訓練負例；3張開發全圖
與11張訓練全圖正例清單及SHA保持不變。不把原本uncertain的場景改成無球。

輸出僅PNG、幾何排除證明、來源manifest、原圖／裁圖並排頁及原始碼快照，沒有
labels/data.yaml，也沒有訓練功能。`--check --candidate-dir ...` 會重算來源資格、
裁圖幾何與像素。球員／圈可出現在球專用負例中，並不改原三類物件的標註語意。
下一輪如要訓練，須另立匯出／實驗版本、保留全圖正例，並安排控制組與匹配的
optimizer更新預算和抽樣策略；增加影像後相同epoch不代表相同訓練量。
這些裁圖不會用於推論、影片篩除或MixNet前處理；相關單元測試合計67項通過。

## 固定更新與正例曝光的負例對照

`ball_negative_pair_v1` 是獨立訓練迴圈，使用Ultralytics原始偵測loss，不修改套件。
`--prepare --candidate-dir ...` 後先`--smoke --experiment-dir ...`（兩組各11次更新），
再以獨立run `--train --experiment-dir ...` 從同一個單類checkpoint重新初始化。
固定單一seed20261025、每循環11張完整正例、30循環，共各330次optimizer更新；
相同正例順序，每次更新兩組都完整計入1張全圖正例loss。控制組只有正例，實驗組
再對該圖的衍生負例獨立forward/backward，加入`0.25 * negative_loss`，最後才step。
負例多一次forward及計算量，但不增加optimizer更新、不替換／減少正例曝光。
這裡的0.25是負例loss係數，與先前品質降權alpha不是同一個參數。

保留checkpoint真實結構，只開Detect參數（DFL固定），兩組所有BN running buffers
均固定，避免額外負例forward改變BN統計。640平方letterbox、原生4:3內容變640×480、
上下各80像素padding，BGR→RGB與球框座標轉換一併測試。使用FP32、batch1、
AdamW lr1e-4／betas(0.9,0.999)、多維權重decay5e-4、bias與BN affine不decay，
gradient clip10，無增強、scheduler、EMA、early stopping或best-epoch選擇。
這不是原先Ultralytics trainer的精確重跑，主要比較本次兩個matched arms。

每11次更新只記錄固定3張全圖開發loss；主要endpoint固定最後330次更新，
不依結果改係數／提早停。結束逐項核對凍結參數與全部BN buffers完全相同、
有限loss/gradient、Detect更新與checkpoint重載。只保存final，沒有resume功能。
`--evaluate --experiment-dir ... --run-dir ... --reference ...` 沿用固定53張、
640/conf0.25/NMS IoU0.7，比較兩組final；3張可算IoU0.5匹配，其餘有框數不是
precision/recall。這是同11訓練圖／3開發圖、單一seed的小型診斷，不是泛化實驗。
不改三類YOLO、MixNet特徵或原始分類validation/test；相關單元測試合計74項通過。

本輪`artifacts/ball_negative_pair_492dq87v/paired_train_xbsz0bsp/`已完成兩組330更新，
固定final在3張開發球匹配均0/3；額外50張候選幀數控制組1、負例組3（非recall）。
`--training-check --experiment-dir ... --run-dir ...`核對11張完整訓練正例重代入：
控制組11/11、負例組10/11，無額外框；只用於診斷訓練記憶，不能當泛化成績。
本設定未觀察到開發改善，因此不採用、不掃更多係數、不覆寫先前模型。
控制組也0匹配使結果有地板限制，不能宣稱所有負例增強都無效；更不能與先前
不同BN／EMA／訓練預算下的2/3直接作單一因素因果比較。優先增加來源多樣性。

## 小批新來源、只標球的複查

`ball_source_review_v2 --prepare --batch ... --source-ball-dataset ...` 沿用原本固定的
source預留，排除既有14張資料所有train/development來源。原始分類train內，
2p0/2p1/3p0/3p1各選一個未使用train來源（source字典序），另取兩個未使用的
預留development來源。只讀既有無損PNG和診斷cache，不重開原片／分類validation/test。
每來源固定10/30/50/70/90%五位置，再排除已記錄的人類球不確定幀；本輪29張候選。

先AI檢查全池與局部／既有轉碼上下文；轉碼上下文只供辨識，不作框座標像素真值。
AI無法可靠定位的來源先defer，不湊滿名額、不轉成空負例。最多6張、每來源1張，
以`--stage --pool ... --proposals ...`另存帶SHA的原生球框草稿、原圖／局部並排圖，
原三類人工JSON不改。這是新球專用協定`visible_physical_ball_only_complete_image_v2`，
只確認所有可見實體球的框與全圖完整性，不包含player/rim/holding_ball角色。

本輪最終4張（B001/B006/B012 train、B029 development），6張暫定版本中B017/B021
已由AI撤回並保留稽核，不向人工追問。只有29張候選的開發用選樣，不是盲測；
AI按可辨識度選圖會有易例偏差，需披露。收到對當前明列圖片／框版本的直接人工
整批或逐張確認前，全部pending、training_ready=0；無YOLO標籤匯出或訓練功能。
`check_pending`核對原圖、顯示圖SHA、來源、球框與未確認狀態；相關測試80項通過。

局部圖更正可用`stage_location_correction`另存修正版：保留舊草稿、原始上傳圖SHA、
直接人工原文與舊／新框，其他圖片的球框和顯示圖SHA必須完全相同。
位置更正不等於新框邊界／整張完整性／整批確認，修正版仍全pending，不啟動訓練。
目前B006已依使用者局部圖移至右側球員前方，舊左側框撤回；相關測試82項通過。

後續兩次直接人工回覆已分別確認B001/B012/B029，以及B006修正版。
`human_review_progress.json`保存這4張球專用完整確認；每次事件包含實際回覆、
當時顯示圖SHA、接受球框與範圍。B029只算development確認，不列入training_ready。
原始AI草稿保持pending歷史，不覆寫成假裝原先就有的人工標註。

## 球專用來源擴充資料v2：14 train／4 development

`ball_dataset_v2`只提供`--prepare`、`--check`與`--preflight`，沒有訓練入口。
匯出前重播兩個人工確認事件、核對B006更正版本、舊11/3資料來源及所有影像SHA；
新4張僅有球專用完整標註，不冒稱具備player/rim/持球角色的完整確認。
舊14張中繼承11張train／3張development，新增B001/B006/B012為train、B029為development。
合計14/4張原生320×240 PNG、各14/4個ball=0框，train11個／development3個來源群組；
來源群組、影片與影像SHA皆不交疊。原始分類validation/test完全未納入。

本機完成資料版：`artifacts/ball_dataset_v2_p6hkr_me/`。
每個匯出檔案與initializer有SHA；沿用原COCO球語意初始化checkpoint，
不是舊擬合best或負例實驗final，不加負例裁圖、不覆寫舊資料／模型／MixNet特徵。
CPU預檢已用官方YOLODataset載入18張圖，全部為3×640×640 uint8及合法單球標籤；
初始化模型保留SPPF SiLU與ball=0語意，合成64×64輸入得到有限的[1,5,84]輸出。
沒有optimizer、backward或訓練。相關單元測試合計92項通過。

新增來源使舊53張診斷清單不能原封重用：v108/v111/v118進入train，v160_v166進入
development，因此從額外未標註清單移除這4群組的20張圖。
`diagnostic_plan.json`固定新版4張有真值development＋30張其他來源未標註全圖，
共34張；後續舊、新模型須在同一新版清單比較，不能直接拿舊2/3對新4張結果。
訓練label為class0，診斷計畫的GT class1只是相容既有比較器的canonical編碼。
未標註候選數不是recall；這批來源曾作開發檢查且偏向可辨識易例，不是盲測。

重驗此資料版：

```bash
python -B -m development.yolo_train_probe_v1.ball_dataset_v2 --check \
  --dataset-dir development/yolo_train_probe_v1/artifacts/ball_dataset_v2_p6hkr_me
python -B -m development.yolo_train_probe_v1.ball_dataset_v2 --preflight \
  --dataset-dir development/yolo_train_probe_v1/artifacts/ball_dataset_v2_p6hkr_me
```

下一步須建立並驗證14/4相容trainer，再跑獨立1 epoch smoke，通過後才能正式訓練。
舊ball-only trainer帶有11/3硬編碼檢查，不可直接套新資料路徑。
`training_plan.json`只記候選策略，`training_started=false`；4張development仍極少，
即使之後匹配增加也不代表可部署到全影片或已證明泛化。

## 14/4 trainer與獨立試跑

`ball_train_v2`使用新版dataset checker，不改舊11/3 trainer。初始化仍深複製實際
checkpoint模組（SPPF SiLU），只允許Detect參數更新，DFL固定；前23層的參數與
BN buffers完全凍結，Detect內BN仍可更新。FP32、640、batch/nbs2、AdamW lr1e-4、
seed20261025、無資料增強，沿用原ball-only策略；不是負例實驗的所有BN固定設定。
每epoch要求14張train各曝光1次與7次optimizer更新，4張development不進訓練batch。
每次更新前檢查梯度有限且至少一個Detect張量非零，訓練結束逐項比較凍結state；
best/last重載後都在4張development全圖驗證可推論。smoke不作正式效能判斷。

本輪獨立smoke `artifacts/ball_dataset_v2_p6hkr_me/smoke_v2_dty69tuq/`已完成：
1epoch、14次影像曝光、7次更新、1,411,779個可訓練參數，峰值allocated GPU約0.55GiB。
梯度、head更新、凍結層／DFL不變、兩份checkpoint重載與來源SHA檢查皆通過。
新6項trainer測試加入後合計98項單元測試通過。正式完整訓練尚未啟動；
dataset_manifest的`training_started=false`表示資料匯出時狀態，實際試跑狀態以
獨立run的`run_state.json`為準，不回寫／覆寫已凍結資料manifest。

執行獨立smoke：

```bash
python -B -m development.yolo_train_probe_v1.ball_train_v2 --smoke \
  --dataset-dir development/yolo_train_probe_v1/artifacts/ball_dataset_v2_p6hkr_me
```

後續完整run須以`--approved-smoke`指定同資料manifest、initializer、trainer來源碼
且全部檢查通過的1epoch smoke；重新從原initializer起跑，不接續試跑權重。
最多100epoch、patience20，所有輸出另建run，不覆寫舊模型：

```bash
python -B -m development.yolo_train_probe_v1.ball_train_v2 --train --epochs 100 \
  --dataset-dir development/yolo_train_probe_v1/artifacts/ball_dataset_v2_p6hkr_me \
  --approved-smoke development/yolo_train_probe_v1/artifacts/ball_dataset_v2_p6hkr_me/smoke_v2_dty69tuq
```

目前只完成smoke，尚未執行上述完整run，也未將任何YOLO用於全影片篩除／MixNet前處理。

後續完整run `artifacts/ball_dataset_v2_p6hkr_me/headtrain_v2_6_dbkorn/`已完成，
30epoch早停、best第10epoch、210次更新／420次train影像曝光，run紀錄約13秒。
所有梯度、凍結state、來源SHA、best/last在4張開發圖的重載推論檢查通過。
best SHA：`d9ca8c07e53f9a6affddb9c3d30b64f571a1dea71cef5a9b0087a27939ac833e`。

## 相同34張全圖：初始化／舊11張／新14張比較

`ball_compare_v2`沿用資料v2的已凍結`diagnostic_plan.json`，先核對兩run來源、
完成狀態、原始initializer相同與best權重SHA，再使用相同34張原生全圖推論。
固定640/conf0.25/NMS IoU0.7/max_det100/FP32，不重訓、不調門檻，球類training0
映到既有比較器canonical1；4張人工開發球以IoU0.5一對一匹配，30張未標註圖
只報候選框／有框影格數，不能據此計算precision/recall。

本次輸出在完整run的`fixed34_comparison_1totu7_0/`。
初始化0/4；舊11張best與新14張best都2/4、漏2、多餘1。兩版同樣匹配E017/E024、
漏E023/B029，E024多餘框為球員手部。額外30圖舊5框／5幀、新4框／4幀；
AI總覽看過34原圖、全部15個三模型候選局部、10張完整比較面板（4開發＋6額外候選）。
U002兩個候選有明顯看台背景誤報；U005_f000126僅列可能球，U016_f000073保留
不確定，不升格標註或編成真陽性數。沒有重做全部影格的完整球標註。

本輪不採用new14作為改善版或全影片篩除／ROI來源，舊新權重及RGB baseline皆保留。
這不等於YOLO無效，只是目前極小、易例偏選且曾反覆使用的4張開發圖沒有建立改善證據；
新舊訓練圖數、更新數與選中epoch不同，也不是單一因素受控消融。
下一步優先規劃更具代表性、來源分離的球專用訓練／開發資料，涵蓋持球、空中、
籃框附近與明確完整複查的無球幀，不再只靠幾張易例與反覆調門檻。AI觀察不是人工真值。
新增比較器5項測試後共103項通過；原始分類validation/test、標註、MixNet特徵不變。

## 14張訓練全圖重代入：best／last分開診斷

`ball_resubstitution_v2`只讀完整run已保存的best／last，對manifest內恰好14張train
全圖推論；排除4張development與原始分類validation/test，不訓練、不修改人工框。
固定640/conf0.25/NMS IoU0.7，以IoU0.5一對一匹配；另報繼承11張與新增3張。
default為best，`--checkpoint last`只作次要擬合診斷，不能根據訓練成績重挑best。

本次best輸出`headtrain_v2_6_dbkorn/training_resubstitution_v2_oenqnwin/`：
匹配8/14、漏6、多餘1；漏E015/E020/E021/E022/B001/B006，多餘在L022。
last輸出`headtrain_v2_6_dbkorn/training_resubstitution_v2_s_xhbjwl/`：
匹配14/14、漏0、多餘0。兩份各4頁總覽均已由AI視覺檢查，涵蓋14張的GT／預測，
原本人工標註與checkpoint選擇未更動；完整解讀在best輸出的`AI_review.html/json`。

因此不能宣稱此架構／凍結骨幹完全無法學習：相同訓練設定的last可以擬合全部14張。
best由開發集fitness選出，不必有最高訓練匹配；訓練14/14也不是泛化證據。
本次未重新推論development；先前best仍2/4開發球、1個額外框。下一優先方向是
來源分離且較具代表性的train/development資料、較可靠的開發選模，而不是自動
改用last、加epochs、降門檻或解凍骨幹。這是優先診斷方向，非唯一原因的因果證明。
6項測試加入後共109項通過，來源資料與權重SHA不變，不套用到全影片或MixNet。

## 新80張固定位置候選池（尚無完整標註／新訓練）

`ball_review_pool_v3`從canonical原始分類train allowlist挑新來源，排除既有全部20個
YOLO來源預留，而不只排除實際用過的14個。35個原始train來源中尚餘15個；
seed20261027先hash預留其中5個為detector-development，其餘10個為train，
舊20來源分配原封保留。8種原始類別各選1支train／1支development影片，train用
8個不同來源，development負載平衡分布5個來源；類別只作抽樣平衡，不是YOLO標籤。
每片固定10/30/50/70/90%五位置，16支影片共80張（40train／40development），
看像素／偵測球之前即固定抽樣；不以清晰度或模型成功替換難例、特寫、曝光或無球候選。

本機池：`artifacts/ball_review_pool_v3_muxe9h1n/`，原生320×240無損PNG，
16次guarded read-only原片開啟；原始分類validation/test皆0次。每個PNG與來源
身份前後一致，舊YOLO資料、權重及MixNet不改。本池所有新增development預留
都必須被未來匯出繼承，不能只用舊20來源清單檢查而漏掉新預留來源。
這是分類train內的detector開發資料；RGB分類器可能曾用過同原始train來源，不是
最終分類盲測或未觸及的正式test。每來源最多少數影片仍有來源與片內相關性限制。

`--propose --pool ... --run-dir ...`只用已完成new14 best作固定640/conf0.25的
弱提案，不訓練、不匯出labels。所有80張不論有無模型框都保留。11張產生13框、
69張空輸出不是無球、不是recall。提案原檔保持未複查狀態；另存`AI_proposals/AI_review`
記錄已看過80張總覽、11張有提案面板與8張放大圖的粗略分流，不冒稱80張完整標註。
N005大球漏框、N051/N071疑似無可見球特寫、N037曝光變化等都保留；無明確轉場
動畫確認，不為湊類型改標。AI無球候選不是已確認負例，空標籤尚未產生。

下一步依固定順序每批8–10張由AI補native球框草稿／uncertain或疑似無可見球狀態，
再以明列範圍整批人工確認；不能只接受可辨識易例或模型成功的development圖作成績。
目前training_ready=0，全數尚未人工確認，沒有新增球框真值、labels或訓練。
6項新測試後合計115項通過；抽樣、split與未確認狀態有fail-closed檢查。
