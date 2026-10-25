/* Local-only draft editor. Downloads JSON; never writes training labels. */
(function () {
  'use strict';
  const core = globalThis.LabelPilotCore;
  const data = JSON.parse(document.getElementById('pilot-data').textContent);
  const $ = id => document.getElementById(id);
  const names = ['場上球員', '球', '籃圈'], colors = ['#50e277', '#ffe35a', '#56dfef'];
  const key = `yolo-target-label:${data.pilot_id}`;
  let review = core.clone(data.draft), index = 0, selected = null, drag = null, img = new Image();
  let autoSave = true, storageBlocked = false;
  try {
    const saved = localStorage.getItem(key);
    if (saved) review = core.checkImport(data.draft, JSON.parse(saved), data.images);
  } catch (error) { autoSave = false; storageBlocked = true; $('message').textContent = `未載入本機暫存：${error.message}。舊暫存不覆寫，請使用匯出／匯入JSON。`; }
  function current() { return review.images.find(r => r.image_id === data.images[index].image_id); }
  function say(text) { $('message').textContent = text; }
  function save() {
    try { if (storageBlocked) throw new Error('Preserve unreadable existing storage'); localStorage.setItem(key, JSON.stringify(review)); autoSave = true; }
    catch (_) { autoSave = false; }
    const meta = data.images[index];
    $('image-info').textContent = `${meta.image_id} / ${meta.audit_sample_id} f=${meta.frame_index}；狀態 ${current().status}`;
    $('progress').textContent = `已完成 ${review.images.filter(r => r.status === 'reviewed').length}/${data.images.length}；略過 ${review.images.filter(r => r.status === 'excluded').length}；${autoSave ? '本機已暫存' : '暫存不可用，請匯出JSON'}`;
  }
  function changed() { current().status = 'pending'; current().annotation_complete = false; $('complete').checked = false; save(); }
  function button(text, action) { const b = document.createElement('button'); b.type = 'button'; b.textContent = text; b.addEventListener('click', action); return b; }
  function element(tag, text) { const e = document.createElement(tag); if (text != null) e.textContent = text; return e; }
  function nextId() { let n = 1; const ids = new Set(current().annotations.map(a => a.id)); while (ids.has(`A${n}`)) n++; return `A${n}`; }
  function boundsText(box) { return box.map(n => Number(n.toFixed(1))).join(', '); }
  function drawBox(ctx, box, color, label, dashed) {
    ctx.strokeStyle = color; ctx.lineWidth = selected === label ? 1.2 : .65;
    ctx.setLineDash(dashed ? [2, 1] : []);
    ctx.strokeRect(box[0], box[1], box[2] - box[0], box[3] - box[1]);
    ctx.setLineDash([]); ctx.font = '4px sans-serif';
    const width = ctx.measureText(label).width + 2, y = Math.max(5, box[1]);
    ctx.fillStyle = '#14202b'; ctx.fillRect(box[0], y - 5, width, 5);
    ctx.fillStyle = color; ctx.fillText(label, box[0] + 1, y - 1);
  }
  function draw() {
    const ctx = $('canvas').getContext('2d'); ctx.setTransform(3, 0, 0, 3, 0, 0);
    ctx.fillStyle = '#14202b'; ctx.fillRect(0, 0, 320, 240); ctx.imageSmoothingEnabled = false;
    if (img.complete && img.naturalWidth) ctx.drawImage(img, 0, 0, 320, 240);
    if ($('show-proposals').checked) for (const p of data.images[index].proposals) {
      if (['pending', 'uncertain'].includes(current().proposal_states[p.id])) drawBox(ctx, p.xyxy, '#ffaf53', p.id, true);
    }
    for (const a of current().annotations) drawBox(ctx, a.xyxy, colors[a.class_id], a.id, false);
    if (drag) drawBox(ctx, [Math.min(drag.start[0], drag.end[0]), Math.min(drag.start[1], drag.end[1]),
      Math.max(drag.start[0], drag.end[0]), Math.max(drag.start[1], drag.end[1])], colors[Number($('new-class').value)], 'new', false);
  }
  function showSelected() {
    const a = current().annotations.find(a => a.id === selected);
    $('coordinate-editor').hidden = !a;
    if (a) { $('selected-id').textContent = `${a.id} ${names[a.class_id]}`; ['x1','y1','x2','y2'].forEach((key, i) => { $(key).value = a.xyxy[i]; }); }
  }
  function renderLists() {
    const pList = $('proposals'), aList = $('annotations'); pList.replaceChildren(); aList.replaceChildren();
    for (const proposal of data.images[index].proposals) {
      const row = element('div'); row.className = 'row';
      row.append(element('div', `${proposal.id} conf=${proposal.confidence.toFixed(2)} [${boundsText(proposal.xyxy)}] — ${current().proposal_states[proposal.id]}`));
      for (const [label, state] of [['球員','accepted'], ['拒絕','rejected'], ['不確定','uncertain']]) row.append(button(label, () => {
        current().annotations = current().annotations.filter(a => a.source_proposal_id !== proposal.id);
        current().proposal_states[proposal.id] = state;
        if (state === 'accepted') current().annotations.push({id:nextId(), class_id:0, xyxy:[...proposal.xyxy], source_proposal_id:proposal.id, holding_ball:null});
        selected = null; changed(); renderLists(); draw();
      }));
      pList.append(row);
    }
    if (!data.images[index].proposals.length) pList.append(element('p','此張沒有person提案；仍需檢查漏掉的球員。'));
    for (const a of current().annotations) {
      const row = element('div'); row.className = 'row' + (a.id === selected ? ' selected' : '');
      row.append(button(`${a.id} ${names[a.class_id]} [${boundsText(a.xyxy)}]`, () => { selected = a.id; showSelected(); renderLists(); draw(); }));
      if (a.class_id === 0) {
        const label = element('label', '手持球（非投籃者身分） '), select = element('select');
        for (const [value,text] of [['','未判斷'],['yes','yes：可見手持球'],['no','no：可見未持球'],['uncertain','uncertain：無法確認']]) {
          const option = element('option',text); option.value = value; select.append(option);
        }
        select.value = a.holding_ball == null ? '' : a.holding_ball;
        select.addEventListener('change', () => {
          core.setHoldingBall(review,current().image_id,a.id,select.value || null);
          save(); say('只更新角色；已接受的物件框與完整性不變。角色不會自動成為YOLO三類訓練目標。');
        });
        label.append(select); row.append(label);
      }
      aList.append(row);
    }
    if (!current().annotations.length) aList.append(element('p','尚無人工標註框。橘色虛線不是已完成標註。'));
    showSelected();
  }
  function renderVisibility() {
    $('visibility').replaceChildren();
    core.names.forEach((name, id) => {
      const label = element('label', names[id] + ' '), select = element('select');
      for (const [value, text] of [['unreviewed','未判斷'],['present','有，可框定'],['absent','沒有可見物件'],['uncertain','不確定／看不清']]) { const option = element('option',text); option.value = value; select.append(option); }
      select.value = current().class_visibility[name];
      select.addEventListener('change', () => { current().class_visibility[name] = select.value; changed(); });
      label.append(select); $('visibility').append(label);
    });
  }
  function show() {
    selected = null; drag = null; const meta = data.images[index], row = current();
    $('image-picker').value = meta.image_id;
    $('image-info').textContent = `${meta.image_id} / ${meta.audit_sample_id} f=${meta.frame_index}；狀態 ${row.status}`;
    $('scene').value = row.scene; $('notes').value = row.notes; $('complete').checked = row.annotation_complete;
    renderVisibility(); renderLists();
    img = new Image(); img.onload = draw; img.onerror = () => say('影像無法載入，請確認images資料夾與此頁放在一起。'); img.src = meta.image_path;
    draw(); save();
  }
  function point(event) { const r = $('canvas').getBoundingClientRect(); return [Math.max(0,Math.min(320,(event.clientX-r.left)*320/r.width)), Math.max(0,Math.min(240,(event.clientY-r.top)*240/r.height))]; }
  $('canvas').addEventListener('pointerdown', e => { if (e.button !== 0) return; const p = point(e); drag = {start:p,end:p}; $('canvas').setPointerCapture(e.pointerId); draw(); });
  $('canvas').addEventListener('pointermove', e => { if (drag) { drag.end = point(e); draw(); } });
  $('canvas').addEventListener('pointercancel', () => { drag = null; draw(); });
  $('canvas').addEventListener('pointerup', e => {
    if (!drag) return;
    drag.end = point(e); const [s,t] = [drag.start,drag.end]; drag = null;
    const box = [Math.min(s[0],t[0]),Math.min(s[1],t[1]),Math.max(s[0],t[0]),Math.max(s[1],t[1])].map(n => Math.round(n*100)/100);
    if (box[2]-box[0] >= 1 && box[3]-box[1] >= 1) {
      const a = {id:nextId(),class_id:Number($('new-class').value),xyxy:box,source_proposal_id:null};
      if (a.class_id === 0) a.holding_ball = null;
      current().annotations.push(a); selected = a.id; changed();
    } else {
      const found = [...current().annotations].reverse().find(a => s[0]>=a.xyxy[0] && s[0]<=a.xyxy[2] && s[1]>=a.xyxy[1] && s[1]<=a.xyxy[3]); selected = found ? found.id : null;
    }
    renderLists(); draw();
  });
  $('apply-coordinates').addEventListener('click', () => {
    const a = current().annotations.find(a => a.id === selected); if (!a) return;
    const box = ['x1','y1','x2','y2'].map(key => Number($(key).value));
    if (!core.boxValid(box)) return say('座標需在320×240內，且寬高均大於0。');
    a.xyxy = box; changed(); renderLists(); draw();
  });
  $('delete-box').addEventListener('click', () => {
    const a = current().annotations.find(a => a.id === selected); if (!a) return;
    if (a.source_proposal_id) current().proposal_states[a.source_proposal_id] = 'pending';
    current().annotations = current().annotations.filter(a => a.id !== selected); selected = null; changed(); renderLists(); draw();
  });
  $('show-proposals').addEventListener('change', draw);
  $('scene').addEventListener('change', () => { current().scene = $('scene').value; changed(); });
  $('notes').addEventListener('input', () => { current().notes = $('notes').value; changed(); });
  $('complete').addEventListener('change', () => { current().status = 'pending'; current().annotation_complete = $('complete').checked; save(); });
  $('reviewed').addEventListener('click', () => {
    const problems = core.readyProblems(current()); if (problems.length) return say(problems.join('\n'));
    current().status = 'reviewed'; save(); say('此張標註完成，尚未輸出訓練標籤。離開前請匯出進度JSON。'); show();
  });
  $('exclude').addEventListener('click', () => {
    if (!current().notes.trim()) return say('請先寫略過原因，例如特寫、轉場或球太小無法框定。');
    current().status = 'excluded'; current().annotation_complete = false; save(); say('此張已略過，不會自動作為負樣本。'); show();
  });
  data.images.forEach(meta => { const option = element('option',meta.image_id); option.value = meta.image_id; $('image-picker').append(option); });
  $('image-picker').addEventListener('change', () => { index = data.images.findIndex(m => m.image_id === $('image-picker').value); show(); });
  $('previous').addEventListener('click', () => { index = Math.max(0,index-1); show(); });
  $('next').addEventListener('click', () => { index = Math.min(data.images.length-1,index+1); show(); });
  $('export').addEventListener('click', () => {
    try { review = core.checkImport(data.draft,review,data.images); }
    catch (e) { return say(`匯出前檢查失敗：${e.message}`); }
    const blob = new Blob([JSON.stringify(review,null,2)],{type:'application/json'}), url = URL.createObjectURL(blob);
    const link = element('a'); link.href = url; link.download = `labels_${data.pilot_id.slice(0,12)}_review.json`; document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url),1000);
    say('已下載進度JSON；內容仍保留待確認狀態，不代表全部是真值。');
  });
  $('import').addEventListener('change', async event => {
    const file = event.target.files[0]; if (!file) return;
    try { const imported = core.checkImport(data.draft,JSON.parse(await file.text()),data.images); review = imported; storageBlocked = false; index = 0; show(); say('已匯入同批進度。'); }
    catch (e) { say(`未匯入：${e.message}。原進度保持不變。`); }
    event.target.value = '';
  });
  show();
})();
