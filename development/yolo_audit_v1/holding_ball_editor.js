/* Role-only local editor: no object geometry or detection readiness mutations. */
(function () {
  'use strict';
  const core = globalThis.LabelPilotCore;
  const data = JSON.parse(document.getElementById('pilot-data').textContent);
  const $ = id => document.getElementById(id);
  const key = `holding-ball:${data.pilot_id}:${data.storage_namespace}`;
  let review = core.clone(data.draft), blocked = false;
  try {
    const cached = localStorage.getItem(key);
    if (cached) review = core.checkImport(data.draft,JSON.parse(cached),data.images);
  } catch (e) { blocked = true; $('message').textContent = `未載入暫存，保留舊資料：${e.message}`; }
  let imageId = null;
  function imageRows() { return review.images.filter(r => r.annotations.some(a => a.class_id === 0)); }
  function persist() {
    try { if (blocked) throw new Error('Preserve unreadable storage'); localStorage.setItem(key,JSON.stringify(review)); }
    catch (_) { $('message').textContent = '本機暫存不可用，請匯出進度JSON。'; }
  }
  function draw(meta,row) {
    const img = new Image(), expectedId = row.image_id;
    img.onload = () => {
      if (expectedId !== imageId) return;
      const ctx = $('canvas').getContext('2d');
      ctx.setTransform(3,0,0,3,0,0); ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0,0,320,240); ctx.drawImage(img,0,0,320,240);
      for (const a of row.annotations) {
        const color = a.class_id === 0 ? (a.holding_ball === 'yes' ? '#ff9f39' : '#50e277') : a.class_id === 1 ? '#ffe35a' : '#56dfef';
        ctx.strokeStyle = color; ctx.lineWidth = .65;
        const [x1,y1,x2,y2] = a.xyxy;
        ctx.strokeRect(x1,y1,x2-x1,y2-y1); ctx.font = '4px sans-serif';
        const y = Math.max(5,y1), label = `${a.id}${a.class_id === 0 ? ':'+(a.holding_ball ?? '?') : ''}`;
        ctx.fillStyle = '#14202b'; ctx.fillRect(x1,y-5,ctx.measureText(label).width+2,5);
        ctx.fillStyle = color; ctx.fillText(label,x1+1,y-1);
      }
    };
    img.onerror = () => { if (imageId === expectedId) $('message').textContent = '影像載入失敗，請保留原始images路徑。'; };
    img.src = meta.image_path;
  }
  function render() {
    const rows = imageRows();
    if (!rows.some(r => r.image_id === imageId)) imageId = (rows.find(r => r.status === 'reviewed') || rows[0] || {}).image_id;
    $('image-picker').replaceChildren();
    for (const r of rows) {
      const o = document.createElement('option'); o.value = r.image_id;
      o.textContent = `${r.image_id} / 物件${r.status === 'reviewed' ? '已複查' : '待審'}`; $('image-picker').append(o);
    }
    $('image-picker').value = imageId || '';
    $('roles').replaceChildren();
    const counts = {unreviewed:0,yes:0,no:0,uncertain:0};
    for (const r of review.images) for (const a of r.annotations) if (a.class_id === 0) counts[a.holding_ball == null ? 'unreviewed' : a.holding_ball]++;
    $('progress').textContent = `物件複查完成 ${review.images.filter(r => r.status === 'reviewed').length}/${review.images.length}；角色：未判斷 ${counts.unreviewed}，yes ${counts.yes}，no ${counts.no}，uncertain ${counts.uncertain}`;
    const row = rows.find(r => r.image_id === imageId);
    if (!row) { $('message').textContent = '目前尚無球員標註框，沒有可填的角色；不從官方person提案自動推定。'; return; }
    const meta = data.images.find(r => r.image_id === imageId);
    for (const a of row.annotations.filter(a => a.class_id === 0)) {
      const div = document.createElement('div'); div.className = 'row';
      const label = document.createElement('label'); label.textContent = `${a.id} [${a.xyxy.map(x => Number(x.toFixed(1))).join(', ')}] 手持球 `;
      const select = document.createElement('select'); select.setAttribute('aria-label',`${row.image_id} ${a.id} 手持球角色`);
      for (const [value,text] of [['','未判斷'],['yes','yes：可見手持球'],['no','no：可確認未持球'],['uncertain','uncertain：無法確認']]) {
        const o = document.createElement('option'); o.value = value; o.textContent = text; select.append(o);
      }
      select.value = a.holding_ball == null ? '' : a.holding_ball;
      select.addEventListener('change', () => {
        core.setHoldingBall(review,row.image_id,a.id,select.value || null); persist(); render();
      });
      label.append(select); div.append(label); $('roles').append(div);
    }
    draw(meta,row);
  }
  $('image-picker').addEventListener('change', () => { imageId = $('image-picker').value; render(); });
  $('export').addEventListener('click', () => {
    try { review = core.checkImport(data.draft,review,data.images); }
    catch (e) { $('message').textContent = `匯出檢查失敗：${e.message}`; return; }
    const url = URL.createObjectURL(new Blob([JSON.stringify(review,null,2)],{type:'application/json'}));
    const a = document.createElement('a'); a.href = url; a.download = `holding_ball_${data.pilot_id.slice(0,12)}_review.json`;
    document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url),1000);
    $('message').textContent = '已匯出進度JSON；未輸出YOLO標籤、未訓練。請將此JSON交回同步。';
  });
  $('import').addEventListener('change', async event => {
    const f = event.target.files[0]; if (!f) return;
    try {
      const imported = core.checkImport(data.draft,JSON.parse(await f.text()),data.images);
      review = imported; blocked = false; imageId = null; persist(); render();
      $('message').textContent = '已匯入同批進度；本頁仍只能修改角色。';
    } catch (e) { $('message').textContent = `未匯入：${e.message}；原進度不變。`; }
    event.target.value = '';
  });
  render();
})();
