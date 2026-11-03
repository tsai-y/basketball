# 連續片段選擇原型v1

隔離的train診斷，不是投籃事件辨識器，也尚未接入MixNet-I3D／Transformer。
先檢查分鏡與球場外觀候選，不能把「有木地板」當成「目標投籃」。
無新訓練、不改原三窗口、分類標籤或特徵快取，不刪原片。

## 方法與安全界線

`selector.py`只接收原生320×240 uint8 BGR連續影格，不接Make/Miss／投籃類型、
品質sidecar、YOLO框或人工事件時間。中央區域低解析HSV直方圖與灰階差提出硬切鏡，
一幀突亮後近似回復時嘗試抑制閃光；寬鬆木地板色面積／橫向覆蓋作court-like外觀。
僅用片內5點局部多數平滑，不能跨切鏡平滑、拼接或補成同一段。

64幀只是與現有3D輸入相容的候選長度下限；每段preview保留原連續長度，尚未決定
新的64幀取樣／短片段策略。不強制為了湊64幀把轉場、後續其他事件接回來。
若多段court-like畫面（含短前段）或沒有足夠長區間，候選回退；不選最長的一段
直接冒稱原投籃。即使只有一段也`event_semantics_verified=false`，
`automatic_training_use_enabled=false`，原三窗口為實際訓練政策，尚無資料集整合。

這些是未校準的工程初始閾值，皮膚／圖卡／非木地板配色／快速移鏡會誤導，
漸變轉場未專門建模；toy測試通過不代表真影片有效。沒有出手與結果的完整時間真值。

## 執行與來源

```bash
python -B -m development.temporal_selection_v1.probe --dry-run
python -B -m development.temporal_selection_v1.probe
python -B -m unittest discover -s development/temporal_selection_v1/tests -p 'test_*.py' -v
```

使用既有seed20261024的20支train診斷樣本，一來源一影片，含已知轉場與特寫反例；
原樣本是有目的的平衡診斷，不是代表性隨機抽樣／盲測。用canonical SplitGuard在
任何影片open前核對split／ID／path；唯讀descriptor順序解碼20次、共3192幀，
原始分類validation/test開啟次數皆0。來源身份前後一致，baseline三窗口原始碼與
feature-cache run.pt的SHA保持不變，沒有對原特徵目錄寫入。

輸出：原片與連續候選mp4預覽、時間軸、帶frame_index的原生contact PNG、逐幀外觀
分數與切鏡候選、全未確認的human_event_review_draft.json。mp4是有損轉碼檢視檔，
不可拿來充當訓練像素；contact只有視覺抽幀，未提供完整事件真值。

## 首輪結果與不採用原因

本機：`artifacts/temporal_probe_v1__lwxmu8j/`。
20支中11支有單一長度合格候選、9支候選回退；11支中6支等同原全片、5支有縮短。
這不是11支剪對，更不是正確率；所有20支仍未允許自動訓練使用。

AI檢查10支的12頁抽幀／切鏡前後contact（不是20支全片連續播放）：
Y004/Y005/Y010/Y011/Y012/Y013/Y016/Y017/Y018/Y019。
Y012建議[36,142)，選到動畫後罰球；使用者於2026-10-28澄清這正是標籤目標，
原manifest亦為ft1，因此事件選取合理。先前AI誤把起始上籃當目標而拒絕的判定已撤回。
人工原文與範圍另存`human_target_event_clarification.json`，舊初審保存history；
本次只確認目標事件身分與選後段的合理性，不是精確邊界／去動畫／結果完整性通過。
Y019整段漸進動畫沒有硬切鏡候選（原圖約frame102/116可見圖卡），雖然因長度回退，
不能宣稱已排除動畫；Y005與Y013亮度／曝光突變被誤當切鏡，flash抑制不足。
Y011在frame44從特寫切回球場有合理分鏡候選，但出手與結果保留仍未確認。

首輪決定：保留Y012為事件選取合理、邊界待核對案例；不以此次澄清自動部署、重訓、
擴到全資料或掃分類準確率挑閾值。原逐幀候選與模型特徵未改。
下一版先分開解決硬切鏡、漸進圖卡與曝光閃光，再建立少量出手／到框／結果事件時間
參考，並處理「目標投籃短於64幀」；必須對準標籤事件，不預設第一個或最長事件才是目標，
也不能靠木地板代替事件定位。
11項單元測試包含半開區間、短片段回退、跨切鏡禁止串接、多事件不自選後段、
閃光toy案例與原始train來源guard；測試只證明這些契約，不證明實景成功率。

## v2：分鏡候選，不等於目標投籃選擇

`selector_v2.py`只接原生影格；取消木地板色排除規則。短曝光需前後外觀恢復，
僅抑制假切鏡、不刪閃光像素；硬切鏡與多幀外觀改變分開提出。滑動視窗遇曝光
不作漸變判定；單一硬切鏡的短視窗回應不膨脹成整段轉場。這也有漏檢風險：
曝光與真轉場同時發生可能漏標；漸變仍可能是快速移鏡，並非動畫分類器。
不接受類別、Make/Miss或人工時間作輸入，不因最早／最長／唯一長鏡頭自選目標。
不跨分鏡拼接。至少8幀的短場景保留在JSON；64幀只控制預覽，短於8幀的間隙
會省略，不能稱為精確逐幀轉場遮罩。尚未解決短投籃的3D取樣政策。

```bash
python -B -m development.temporal_selection_v1.probe_v2 --dry-run
python -B -m development.temporal_selection_v1.probe_v2
python -B -m unittest discover -s development/temporal_selection_v1/tests -p 'test_*.py' -v
```

2026-10-28：固定7支已看過的原train開發案例，非盲測／代表性評估。
初跑`artifacts/temporal_probe_v2_xcagijiw/`保留；發現閃光視窗假漸變與硬切鏡視窗
擴張後修正上述兩項處理，再跑`artifacts/temporal_probe_v2_xafv9o3e/`。
每次7個原train影片open，兩次合計14；分類validation/test皆0。沒掃參數找分類準確率。
來源保護雜湊、原生PNG round-trip及逐檔SHA皆記錄，原三窗口與特徵快取不變。

|案例|v2結果（零起算半開區間）|AI抽幀初審限制|
|---|---|---|
|Y005|保留[0,146)，57為曝光候選|原v1的57/58假切鏡不再出現|
|Y010|切鏡162；曝光88|161/162可見球場轉特寫，非事件完整性認證|
|Y011|切鏡44|43/44可見特寫轉罰球球場|
|Y012|場景[44,142)，漸變[2,44)|後段罰球身分已人工確認；38已見球場，起點偏保守|
|Y013|保留[0,146)，60–62為曝光候選區|v1的60–63假切鏡不再出現|
|Y018|保留[0,146)|已查的移鏡抽幀沒有被切斷，不代表全片無問題|
|Y019|漸變[84,105)、[110,126)；場景[0,84)、[126,146)|84–88仍是球場，120已回球場；105–109仍是圖卡，轉場遮罩不完整|

`AI_review.json/html`另存AI可見證據、實際看過的contact頁、剩餘問題；
`AI_event_reference.json`是獨立AI草稿，不覆寫全空的原事件表，也不是人類真值。
Y012提出[38,142)的檢視參考，精確出手／到框／結果幀無法可靠定位仍留空；
Y019前段罰球起初是AI待確認目標；使用者於2026-10-28回覆「是轉場前的罰球」，
已另存`human_Y019_target_confirmation.json`並同步事件表／初審頁，保留確認前snapshot。
只確認事件身分，不根據ft0或長度推論人工認可，也不擴張成[0,84)邊界／結果完整性通過。
目前20項單元測試通過，自動採用0、重訓0。下一步應先確認目標事件與保留完整性，
再固定政策測未看過的原train影片；不可直接擴全資料或覆蓋原特徵。

## 出手／結果保留檢查（兩支目標確認後）

```bash
python -B -m development.temporal_selection_v1.event_retention_v1 --dry-run
python -B -m development.temporal_selection_v1.event_retention_v1
```

2026-10-28輸出`artifacts/event_retention_v1_vqoqjui0/`：只讀Y012/Y019各一次、
共292幀，classification validation/test未開啟。保存每幀原生PNG與SHA、來源前後身分，
提供逐格拖拉與僅供檢視的nearest局部放大；不是YOLO裁切或新的訓練輸入。
AI逐格查看Y012[44,96)、Y019[0,52)的14頁局部圖與額外證據，未聲稱看過全部292幀。

Y012候選[44,142)保留約[47,56)的出手動作證據與後續動作；精確到框／結果不可靠仍留空。
Y019候選[0,84)沒有額外裁掉原片開頭；原片初始伸臂姿態疑似已出手，需區分來源左截斷
與選段造成的截斷，目前未確認，不能宣稱完整事件通過。原片89仍是球場，終點偏保守。
`AI_assessment.json`與`AI_review.html`單獨記錄AI證據、空值與唯一待確認問題，
不擴張使用者先前的「目標身分確認」為精確時間真值。
使用者接著回覆Y019開頭是否離手「不確定」，已另存`human_Y019_start_uncertainty.json`，
同步初審頁並保留更新前snapshot。開頭狀態維持null，不標成來源缺陷，不再反覆詢問；
目標仍是已確認的轉場前罰球，保留原片，事件完整性仍未知。

`bracket_retention`僅檢查所給事件證據區間是否落在候選內，null必須保持unresolved；
即使三項包含也不是獨立定位驗證或訓練批准。新增3項契約測試，合計23項通過。
沒有調整選段政策、掃模型準確率、重訓、刪影片或改原特徵快取。

## 固定v2的新影片診斷（不按結果調參）

`frozen_probe_v2.py`先核對`selector.py`／`selector_v2.py`與既有v2來源snapshot、
policy.json完全相同，再盤點YOLO audit／YOLO train probe／temporal artifacts中
manifest或sampling_plan內實際列出的video_id。不是完整人類觀看紀錄，沒有聲稱
未被品質pilot或RGB訓練使用。source_assignments中只預留但沒有實際影片的群組
不視為已看過；不開分類validation/test來湊獨立來源。

盤點21份檔案，36支train ID、33群組，僅剩v130/v156共2個未列出群組。
在看像素之前依seed20261028的SHA排序，每組6支，共12支，沒有按類別、球清楚度、
品質分數或模型準確率挑片。抽樣計畫先落盤；不因失敗／難例換片。
這12支只有2個獨立來源，不可當成12來源或正式分類盲測。

首次新批次（產生前可加`--dry-run`預覽，不寫檔／開影片）：

```bash
python -B -m development.temporal_selection_v1.frozen_probe_v2
```

本次已完成`artifacts/frozen_v2_new12_pst3tp7_/`，以其中index.html／AI_review.html
檢查即可。新manifest也會列入下次盤點，因此已用完未列出群組時重跑會拒絕，
不會悄悄放寬來源隔離或換到分類validation/test；要復查請使用既有輸出。

只讀12次原train、1752幀，來源前後身分／原生PNG round-trip／SHA記錄完整。
AI查看24頁固定16位置總覽＋15頁提出邊界前後圖，不是全部1752幀或全片連續播放。
12支都有長度合格場景，6支保持全片、6支縮短；不代表12支剪對。
T001有重播→圖卡→球場；T006尾部圖卡、T008/T009/T012尾部特寫有粗略分段，
仍有乾淨球場被提前移除、轉場類型不準與依短片段長度省略的問題。
T003/T008/T009/T011所查曝光未誤切，但T004的frame10曝光＋frame11恢復尾巴
被誤當硬切鏡，並產生[3,9)、[13,19)假漸變，長候選從19才開始。
像素測量顯示9→11尚未恢復（hist0.548），9→12已恢復（hist0.080），
程式在11太早做恢復檢查。`T004_flash_failure.json`保留原因，不在本輪修改v2。

28項契約測試通過，不等於實景剪段成功。未知事件不硬標通過，不再要求使用者
逐球複判此清楚的曝光反例；自動訓練使用0、未改原三窗口／特徵快取或重訓。
下一步另立版本修復曝光尾巴並測真切鏡反例，不能直接套用全片或用分類準確率調閾值。

## v3：曝光峰值後的恢復尾巴

`selector_v3.py`另立版本，v2檔案／既有數值保持不變。曝光峰值只作seed，
在原`flash_max_frames=4`上限內往後找恢復；中間影格需維持空間相關、不可大幅變暗。
新增工程預設：恢復需持續2幀、回到曝光前平均亮度差<=0.04。第一返回幀對比曝光前，
後續恢復幀的空間相關對比第一返回幀（hist／亮度仍對比曝光前），避免正常移鏡
被越來越舊的空間錨點誤判。沒有刪閃光像素、跨場景拼接或自動選目標事件。
這些仍是未校準啟發式：相似場景短切、過曝破壞結構、快速移鏡／片尾恢復可能混淆。

```bash
python -B -m development.temporal_selection_v1.compare_v3 --dry-run
python -B -m development.temporal_selection_v1.compare_v3
python -B -m unittest discover -s development/temporal_selection_v1/tests -p 'test_*.py' -v
```

使用原7案例＋新12案例，共19支已看過的train影片、9來源，屬開發回歸而非新盲測。
先重播v2並要求逐項結果等於原report，再跑v3；不按分類準確率調參或換片。
初稿`artifacts/exposure_tail_v3_8u82cpgz/`修好T004但讓Y013退步，完整輸出／當時
來源snapshot保留；修訂恢復持續的比較錨點後重跑，不降低原有相關度門檻。
初稿report內的工作區v3來源SHA代表當時版本，後續來源已修訂，查當時內容請看inputs/。

修訂結果：`artifacts/exposure_tail_v3_r7cwrftz/`。
T004由錯誤候選[19,146)改為[0,146)，辨認曝光[10,12)；其餘18支的切鏡、
漸變／曝光區間與場景區間不變。Y013多次曝光維持不誤切，Y010162、Y01144、
Y012142、T01210真切鏡仍保留。36項單元測試與8項指定影片回歸通過，
不等於19支投籃選段正確或已解決漸變邊界。

每輪原train19次open、2842幀，兩輪合計38次／5684幀；classification validation/test為0。
最終AI重看11頁重點抽幀（T004全總覽與邊界、Y013曝光、四例真切鏡），
其餘用結構化結果比較，沒有宣稱19支完整逐幀人工驗證。Y019開頭仍為使用者不確定。
沒有修改原三窗口、分類標籤、品質真值、特徵快取或訓練，尚不允許自動全資料採用。
下一步針對過寬的場景變化範圍與不確定回退另作處理，不把此修正冒稱完整事件定位器。

## v4：寬搜尋區與細邊界分開

`selector_v4.py`不修改v2/v3，保留v3曝光與原先數值。只在v3提出的寬搜尋區內，
以相鄰幀的局部強峰判別單次突變（hist>=.40、gray>=.14、峰值優勢>=2.5），
或以hist>=.15／gray>=.07的活動範圍縮回邊緣。間隔<=2×原視窗半徑且至少50%
影格仍在變化時才合併兩塊搜尋區，不因場景短就串接；未有局部證據時保留原寬區。
上述是未校準工程預設，不是論文閾值或已驗證圖卡語意。

活動edge i指i-1→i；多幀變化[first,last)把last當返回幀候選，不把返回幀多刪一幀。
若片尾還在變化，明確標right_censored，不能聲稱見到返回。
`all_segments_including_short`完整記錄每幀所屬區間，避免藉小於8幀的省略遮掩問題。
不足64幀仍只是不產生模型長度預覽，不代表不是目標；不選最早／最長場景訓練。

```bash
python -B -m development.temporal_selection_v1.compare_v4 --dry-run
python -B -m development.temporal_selection_v1.compare_v4
```

本機`artifacts/boundary_refine_v4_se8f2bo6/`：19筆已知v3像素分數全部細修，7筆有變更。
僅7筆重新原生解碼（7次train open、1022幀），要求v3結果完全重現、v4原生組合
等於快取細修；12筆不變沿用舊分數與有損預覽，分類validation/test為0。
AI看完7筆29頁總覽／邊界抽幀，不是完整逐幀真值或新盲測。

|案例|v3長候選 → v4候選|可見效果與限制|
|---|---|---|
|Y012|[44,142) → [37,142)|多7幀罰球準備；變化起點4仍誤包上籃，非精確圖卡遮罩|
|Y019|[0,84) → [0,89)，後段[126,146) → [120,146)|變化提案[89,120)，原間隙105–109所見圖卡接續；原片開頭仍不確定|
|T001|[28,146) → [21,146)|圖卡提案[13,21)，另記短重播[0,13)，不自選目標事件|
|T006|[0,137) → [0,143)|尾部圖卡起點縮回；片尾右截斷、精確首像素未認證|
|T008|[0,133) → [0,140)|139→140為所見球場切特寫，6幀特寫也完整留紀錄|
|T009|[0,131) → [0,138)|137→138切特寫，後8幀另列短場景|
|T012|[10,127) → [10,132)|131→132切特寫，前後特寫都另列|

43項契約測試、5項整批回歸通過：曝光提案全19不變、T004修正保留、四個既查真切鏡保留、
所有短／長／變化區間可完整核算、無目標自選或訓練批准。不代表7例均完全正確，尤其
Y012[4,37)仍含真正上籃動作，必須保留語意不確定與原三窗口回退，不能自動刪除。
沒有變動原標籤、品質真值、特徵快取或訓練，不將局部邊界改善冒稱完整投籃定位。

## 固定 v4 → 三窗口輸入：小規模串接測試

`window_adapter_v1.py`以 composition 包住原 `BasketballThreeWindowDataset`，不改原載入器。
這是 train-only 診斷介面，非全資料選段政策：只接受明示的候選測試請求，不自動選最長場景。
採用區間以半開原片座標 `[a,b)` 表示；長度66–192幀才可用三個連續64幀窗口完整覆蓋，
前／中／後起點為 `a, a+(b-a-64)//2, b-64`。不確定、未啟用、過短、越界或無法覆蓋皆
呼叫原流程回退，保留其所有原欄位與同一個 tensor，只附加 adaptation 診斷 metadata。
若原影片本身不符合原政策，仍由原載入器拒絕，不補幀、不拼接或放寬原契約。

固定5個已檢視案例，不增加逐球標註或調整v4閾值：

|案例|用途|結果|
|---|---|---|
|T004|正常完整原片對照|identity；原三窗口不變|
|Y012|後段罰球 `[37,142)` 的候選輸入分支|`[37,101)`、`[57,121)`、`[78,142)`|
|Y019|轉場前罰球，但原片開頭仍不確定|回退原政策|
|T001|重播與後段球場的目標事件未確認|回退，不自選最長|
|T008|既有6幀特寫 `[140,146)`，刻意提供的短片段負例|回退；不是把特寫當投籃|

Y012只有目標事件身分獲人確認，完整事件與精確端點並未變成人工真值；此次明示允許的
是候選座標／輸入機械測試，所有案例 `training_use_approved=false`，不可直接轉作訓練清單。

```bash
python -B -m development.temporal_selection_v1.smoke_windows_v1 --dry-run
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -B -m development.temporal_selection_v1.smoke_windows_v1
python -B -m unittest discover -s development/temporal_selection_v1/tests -p 'test_*.py' -v
```

本次輸出：`artifacts/adapter_smoke_v1_yngsk1h0/index.html`（每次執行另建目錄）。
50項 temporal 契約測試與14項 MixNet-I3D 既有單元測試通過；5筆共10次GPU前向皆通過。
逐筆獨立原生BGR解碼後明示轉RGB、縮放，核對每個窗口全部64幀／全部像素；stem hook
另核對只做一次mean/std。輸入 `[3,3,64,192,256]`、特徵 `[1,3,8,1536]`、logits `[1]`，
皆有限。3個回退與1個identity的原欄位／輸入逐值相同，前向特徵SHA與logit也完全相同。
AI另查看Y012新舊與T008回退共3張抽幀拼圖，未宣稱完整逐幀語意審核。

原train合計38次open（含獨立重播／像素檢查），classification validation/test為0。
模型全eval／凍結，無optimizer；參數與buffers不變。原快取8743檔的檔名／大小／mtime清單
雜湊前後一致，沒有讀特徵內容做比較或覆寫特徵；未訓練、不計準確率、未碰TANet。
保留38次原片讀取與模型／來源SHA在report，後續是否重建特徵與受控訓練比較另作決定。

## 固定 v4 的100支 train 試跑（不是來源盲測）

`batch_probe_v1.py`不改v4與轉接器，僅列出場景、窗口幾何與回退計畫，不擷取tensor／特徵、
不載入模型或計準確率。看像素前以seed20261028固定雜湊排序，輪流從全部35個train來源
各取2–3支，排除既有YOLO／temporal manifest與sampling plan列出的48個ID；不按標籤、
長度、品質或模型輸出挑片。新ID仍可能曾用於RGB訓練／其他未盤點觀看，來源也不是新群組。

初次執行可用 `python -B -m development.temporal_selection_v1.batch_probe_v1 --dry-run` 預覽；
計畫落盤後預設拒絕重抽，重播必須明示原計畫（不因已看過像素再挑100支）：

```bash
python -B -m development.temporal_selection_v1.batch_probe_v1 --dry-run --plan development/temporal_selection_v1/artifacts/batch100_v4_hdz2h_f4/sampling_plan.json
# 若需重播同一批，移除 --dry-run；另建新輸出目錄，不覆寫本次結果。
```

本次輸出 `artifacts/batch100_v4_hdz2h_f4/index.html` 與 `AI_review.html`：

- 100/100解碼完成、35來源、15379幀，來源開啟train100次、classification validation/test各0。
- 原長度145幀3支、146幀74支、180幀23支；76支全片identity，24支多場景保守回退。
- 必須數進所有短場景（含不足8幀），不因另有唯一長段就自選目標；24支中22支恰有唯一
  可供三窗口覆蓋的長段，但短段也可能包含出手。沒有新增裁段採用或實際刪除幀。
- 已生成124個有損影片預覽、157張原生接觸圖及13張小總覽。所有預覽只供檢查，不當特徵輸入。

AI初審查看28張圖：100支各8位置的小總覽，30支各6個原生位置的重點圖，另7支邊界圖。
不是100支連續播放、逐幀真值或分類評分。原生重點圖由既有16位置PNG無插值取tile，
沒有再開原片。AI觀察到至少6個「全片仍含換鏡／圖卡」反例（S001/S016/S027/S033/S048/S090）；
S012/S021/S026/S036/S051/S069/S094的部分變化提案仍包含球場動作／曝光，不能當動畫刪掉。
S030/S032的短早段可見罰球準備或抬手，說明單純取後段長場景可能丟失事件證據。
15項具體反例及逐頁SHA見獨立`AI_review.json`；初始manifest保持原樣，不改人工標註或標籤。

這輪結論是「介面與回退可用，但v4尚不能安全自動裁全資料」，不是76%選段準確或24%失敗。
此輪固定閾值，不再調參；沒有重建特徵、開新訓練或改TANet。原快取8743檔的檔名／大小／mtime
清單與受保護來源／歷史SHA前後一致。`run.pt`等既有快取provenance僅可讀作SHA核對，
沒有載入或寫入特徵tensor。後續若修正，另立版本並保留本批作開發回歸，不冒稱新盲測。

61項temporal單元測試通過；另核對原抽樣可重現、100支來源身分、117個受保護SHA、
28張已查圖SHA、530個HTML連結與片內錨點、原快取清單與原／有效窗口。結果存於
`verification.json`。manifest的AI pending是初始分析時點；獨立`AI_review.json`記錄後續
AI抽幀檢查，不覆寫初始分析或冒充人類確認。

## v5：局部曝光脈衝，不降低原全畫面門檻

`selector_v5.py`另立版本，不改v1–v4。固定100支的反例顯示，部分掃描式／局部曝光
全畫面亮度峰值約0.06–0.09，未達原0.10；原漸變視窗因而在曝光前後產生假變化區。
v5在原96×54灰階descriptor分成3×4區塊，各自對前後5幀的tile亮度中位數計峰值，
至少兩個四鄰接區塊超過原 `luma_peak=0.10` 才補提曝光種子。
全畫面曝光規則及既有數值全部保持，新增格線／連通數是未校準工程預設，非論文參數。

局部種子仍需原4幀上限內恢復、2幀持續、hist<=0.30、相關>=0.75、平均亮度差<=0.04；
中間影格也需保留原外觀結構且不可突然變暗。相鄰脈衝依舊gap=1合成episode，
可能含中間不亮影格，不能當逐幀曝光真值。曝光只抑制假切鏡／漸變提案，不刪像素。
相似短鏡頭、移動亮物與局部照明仍可混淆，不將本版冒稱投籃或圖卡分類器。

```bash
python -B -m development.temporal_selection_v1.compare_v5 --dry-run
python -B -m development.temporal_selection_v1.compare_v5
python -B -m unittest discover -s development/temporal_selection_v1/tests -p 'test_*.py' -v
```

本機 `artifacts/local_exposure_v5_of7zq9p2/`：固定100支＋舊19支共119個已見train開發案例，
先完整原生解碼重現v4全部欄位，再跑固定v5，不按結果換片、做參數搜尋或分類評分。
主回歸119次train open／18221幀；另有前置成因量測7次train open；validation/test均0。

69項temporal單元測試與8項整批回歸通過：

- S012/S021/S036/S051/S069/S094六個反例的14段曝光周邊變化提案撤除。
- 新增7支共10個局部曝光episode；除上述六支外，Y011新增曝光但場景／換鏡不變。
- 既有v4硬切候選全保留、原v3全畫面曝光紀錄逐欄不變；四個既查真正切鏡與
  T004／Y013完整場景回歸保留。所有場景／短區間仍完整核算，無拼接或自動選目標。
- 固定100支整片identity由76→81、多場景回退24→19，不是分類或選段準確率。

S026仍有[70,84)運鏡／動作變化提案。雖f77是亮度脈衝，f78對曝光前影格相關0.7137
低於原0.75，因此不放寬門檻硬判恢復，保留回退與已知限制。前輪6個特寫／圖卡漏切
也未在本版處理。新裁段採用0、未改原三窗口、標籤、特徵或訓練，TANet保持不動。

AI看過14張本輪原生抽幀圖（含全部10個新增曝光的前／起／返回／返回後影格），另看
1張舊S026邊界圖；未重新連續播放119片，沒有宣稱完整事件真值。原快取清單及242個
來源／模型provenance／歷史／預覽SHA維持。結果、AI範圍與限制在report／AI_review／verification。

## 固定 v5 → 三窗口：獨立轉接測試

`window_adapter_v5.py`獨立包住原載入器，不修改原資料流程、v4轉接器或v5閾值。
沿用已測過的原片座標解碼／reference RGB空間轉換機制，額外要求裁段請求必須精確對應
凍結v5紀錄中的乾淨場景提案；僅有長度合格、手動指定跨場景或完整事件未確認都不構成
訓練採用。保存v5完整report SHA、逐筆analysis SHA、原請求、有效區間、原片窗口座標及原因。
不確定／短片段仍直接返回原tensor與所有原欄位，不補幀、不跨鏡頭拼接、不新增選事件模組。

```bash
python -B -m development.temporal_selection_v1.smoke_windows_v5 --dry-run
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -B -m development.temporal_selection_v1.smoke_windows_v5
```

本機 `artifacts/adapter_smoke_v5_zisd_ekx/index.html`，8個既有train案例：

|案例|分支與理由|
|---|---|
|T004、S021|整片identity對照；S021包含已修正曝光提案，但不需要裁片|
|Y012|只測非identity輸入分支；[37,142)→[37,101)、[57,121)、[78,142)|
|Y019、S026|原片開頭不確定／曝光恢復未確認，回退|
|T001、S030|多事件目標未確認／短早段可能含出手，回退，不選最長|
|T008|既有6幀特寫負例，回退，不當成投籃|

Y012先前只有後段罰球的事件身分獲人確認；此次是明示的工程候選測試，不表示完整出手、
結果或端點已成為真值。所有8筆 `production_crop_adopted=false`，新訓練裁段採用仍0。
這也不改寫先前119筆全部使用原窗口的歷史結果。

75項temporal單元測試、8筆實際串接、16次GPU前向與10項輸入契約檢查通過。
每個原／新窗口全部64幀／全部像素，另由原生BGR→RGB→reference縮放核對；標籤維持。
stem hook確認只做一次mean/std；輸入[3,3,64,192,256]、特徵[1,3,8,1536]、logits[1]均有限。
5個回退與2個identity的所有原欄位／tensor逐值相同，特徵SHA與logit亦完全一致。
AI看過Y012原／新與S030回退共3張輸入拼圖，只核對工程呈現，不宣稱完整事件認證。

原train共59次open（含獨立原流程重播與逐像素檢查），classification validation/test原片皆0。
既有權重離線載入、全凍結eval/inference_mode、無optimizer，參數與buffers不變。
原快取8743檔的檔名／大小／mtime清單與249個受保護SHA不變；27個64幀有損預覽只供檢視，
不存特徵tensor、不訓練、不計準確率、不改TANet。通過的是串接契約，不是裁剪效果。
