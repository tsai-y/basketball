/* Pure annotation validation. No UI, network, training or file access. */
(function (root) {
  'use strict';
  const names = ['active_player', 'ball', 'rim'];
  const clone = data => JSON.parse(JSON.stringify(data));
  const need = (condition, message) => { if (!condition) throw new Error(message); };
  const roleProtocol = 'visible-hand-held-ball-v1';
  const roleValues = [null, 'yes', 'no', 'uncertain'];
  function checkRole(annotation) {
    if (annotation.class_id !== 0) {
      need(!Object.prototype.hasOwnProperty.call(annotation, 'holding_ball'), 'holding_ball 只可附在場上球員');
    } else {
      need(roleValues.includes(annotation.holding_ball === undefined ? null : annotation.holding_ball), '非法 holding_ball 值');
    }
  }
  function setHoldingBall(review, imageId, annotationId, value, timestamp) {
    need(roleValues.includes(value), '非法 holding_ball 值');
    need(review.role_metadata_protocol === undefined || review.role_metadata_protocol === roleProtocol, '角色規則版本不符');
    const row = review.images.find(r => r.image_id === imageId);
    need(row, '影像不存在');
    const annotation = row.annotations.find(a => a.id === annotationId);
    need(annotation && annotation.class_id === 0, '持球屬性只可附在場上球員');
    need(review.role_annotation_events === undefined || Array.isArray(review.role_annotation_events), '角色事件紀錄格式不符');
    const previous = annotation.holding_ball === undefined ? null : annotation.holding_ball;
    if (previous === value) return;
    review.role_metadata_protocol = roleProtocol;
    annotation.holding_ball = value;
    (review.role_annotation_events ||= []).push({image_id:imageId, annotation_id:annotationId,
      previous, value, source:'human_role_editor', recorded_at_utc:timestamp || new Date().toISOString(),
      confirmation_scope:'visible_hand_holding_ball_only_not_shooter_or_dribble_possession'});
    // Roles are independent from accepted detection boxes, scene and completeness.
  }
  function boxValid(box) {
    return Array.isArray(box) && box.length === 4 && box.every(v => typeof v === 'number' && Number.isFinite(v)) &&
      0 <= box[0] && box[0] < box[2] && box[2] <= 320 && 0 <= box[1] && box[1] < box[3] && box[3] <= 240;
  }
  function readyProblems(row) {
    const problems = [];
    if (row.scene !== 'court_play') problems.push('尚未確認是球場比賽畫面；特寫／轉場可填原因後略過');
    if (!row.annotation_complete) problems.push('尚未勾選已檢查漏標與重複框');
    if (Object.values(row.proposal_states).some(s => !['accepted', 'rejected'].includes(s))) problems.push('仍有人物提案未決定或不確定');
    names.forEach((name, id) => {
      const state = row.class_visibility[name], count = row.annotations.filter(a => a.class_id === id).length;
      if (!['present', 'absent'].includes(state)) problems.push(`${name} 尚未確認有／無`);
      else if ((state === 'present') !== (count > 0)) problems.push(`${name} 的有／無與框數不一致`);
    });
    return problems;
  }
  function checkImport(base, review, metadata) {
    need(review && review.schema === base.schema && review.pilot_id === base.pilot_id, '不是本批標註檔');
    need(JSON.stringify(review.classes) === JSON.stringify(names) && review.protocol === base.protocol, '類別對照或規則版本不符');
    need(review.role_metadata_protocol === undefined || review.role_metadata_protocol === roleProtocol, '角色規則版本不符');
    need(Array.isArray(review.images) && review.images.length === base.images.length, '影像列數不符');
    const allowed = new Map(metadata.map(row => [row.image_id, row]));
    const seen = new Set();
    for (const row of review.images) {
      const source = allowed.get(row.image_id);
      need(source && !seen.has(row.image_id) && row.image_sha256 === source.image_sha256, '影像ID重複或來源不符');
      seen.add(row.image_id);
      need(['pending', 'reviewed', 'excluded'].includes(row.status), '非法複查狀態');
      need(['unreviewed', 'court_play', 'closeup', 'transition', 'other', 'uncertain'].includes(row.scene), '非法畫面類型');
      need(typeof row.annotation_complete === 'boolean' && typeof row.notes === 'string', '非法確認／備註欄位');
      need(Array.isArray(row.annotations), '非法框清單');
      const ids = new Set();
      for (const a of row.annotations) {
        need(typeof a.id === 'string' && !ids.has(a.id) && Number.isInteger(a.class_id) && a.class_id >= 0 && a.class_id < 3 && boxValid(a.xyxy), '非法或重複框');
        ids.add(a.id);
        checkRole(a);
      }
      need(row.proposal_states && typeof row.proposal_states === 'object', '缺少提案狀態');
      const proposalIds = source.proposals.map(p => p.id);
      need(Object.keys(row.proposal_states).length === proposalIds.length && proposalIds.every(id => ['pending', 'accepted', 'rejected', 'uncertain'].includes(row.proposal_states[id])), '提案狀態不符');
      for (const pid of proposalIds) {
        const linked = row.annotations.filter(a => a.source_proposal_id === pid);
        need(linked.length === (row.proposal_states[pid] === 'accepted' ? 1 : 0), '提案與標註框不一致');
        need(!linked.length || linked[0].class_id === 0, '人物提案只能確認為球員');
      }
      need(row.annotations.every(a => a.source_proposal_id == null || proposalIds.includes(a.source_proposal_id)), '未知提案來源');
      need(row.class_visibility && Object.keys(row.class_visibility).length === 3 && names.every(name => ['unreviewed', 'present', 'absent', 'uncertain'].includes(row.class_visibility[name])), '類別可見性不符');
      if (row.status === 'reviewed') need(readyProblems(row).length === 0, '標註仍有缺項，不可宣告完成');
      if (row.status === 'excluded') need(row.notes.trim(), '略過必須填原因');
    }
    return clone(review);
  }
  const api = Object.freeze({names, clone, boxValid, readyProblems, checkImport, roleProtocol, setHoldingBall});
  root.LabelPilotCore = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(globalThis);
