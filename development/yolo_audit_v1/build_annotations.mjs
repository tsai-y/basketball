// Use the bundled Artifact Tool. The workspace contains a node_modules symlink
// to the dependency loader's node_modules; no project packages are installed.
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

const [runArg, workspaceArg] = process.argv.slice(2);
if (!runArg || !workspaceArg) throw new Error('Usage: build_annotations.mjs AUDIT_RUN BUNDLED_WORKSPACE');
const run = path.resolve(runArg);
const require = createRequire(path.join(path.resolve(workspaceArg), 'artifact-loader.cjs'));
const { Workbook, SpreadsheetFile } = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const payload = JSON.parse(await fs.readFile(path.join(run, 'annotation_payload.json'), 'utf8'));
if (payload.videos.length !== 20 || payload.frames.length !== 100) throw new Error('Unexpected annotation population');
const outputDir = path.join(run, 'outputs', 'yolo_audit_v1');
await fs.mkdir(outputDir, { recursive: true });
for (const name of ['annotation_book.xlsx', 'video_annotations.csv', 'frame_annotations.csv']) {
  try { await fs.access(path.join(outputDir, name)); throw new Error(`Refusing to overwrite ${name}`); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
}

const workbook = Workbook.create();
const yn = ['yes', 'no', 'uncertain'];
const yna = [...yn, 'n_a'];
const choices = {
  shot_event_visible: yn, pre_release_shooter_missed: yna,
  shooter_dropout: ['none', 'brief', 'repeated', 'uncertain', 'n_a'],
  non_player_false_boxes: yn,
  id_stability: ['stable', 'switch', 'lost', 'repeated_switch', 'uncertain', 'n_a'],
  cut_or_replay: ['none', 'cut', 'replay', 'both', 'uncertain'], outcome_visible: yn,
  roi_would_lose_task_info: yn,
  overall_suitability: ['usable', 'partially_usable', 'unusable', 'uncertain'],
  shooter_visible: yn, shooter_detected: yna, ball_visible: yn, ball_in_roi: yna,
  rim_visible: yn, rim_in_roi: yna,
};
const countFields = new Set(['matched_player_boxes', 'non_player_or_duplicate_boxes',
  'uncertain_boxes', 'missed_visible_players', 'first_issue_frame']);
function column(n) {
  let result = '';
  for (n++; n > 0; n = Math.floor((n - 1) / 26)) result = String.fromCharCode(65 + (n - 1) % 26) + result;
  return result;
}
function csvValue(value) {
  const text = value === null || value === undefined ? '' : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

for (const spec of [
  { name: 'Video Audit', rows: payload.videos, human: payload.video_human_fields,
    table: 'VideoAuditRecords', csv: 'video_annotations.csv' },
  { name: 'Frame Audit', rows: payload.frames, human: payload.frame_human_fields,
    table: 'FrameAuditRecords', csv: 'frame_annotations.csv' },
]) {
  const sheet = workbook.worksheets.add(spec.name);
  sheet.showGridLines = false;
  const headers = Object.keys(spec.rows[0]);
  const values = [headers, ...spec.rows.map(row => headers.map(key => row[key] ?? null))];
  for (const row of spec.rows) for (const key of spec.human) {
    if (row[key] !== null) throw new Error(`Human input unexpectedly filled: ${key}`);
  }
  const end = `${column(headers.length - 1)}${values.length}`;
  const region = sheet.getRange(`A1:${end}`);
  region.values = values;
  region.format.font = { name: 'Arial', size: 10, color: '#202936' };
  region.format.verticalAlignment = 'center';
  region.format.rowHeight = 25;
  const header = sheet.getRange(`A1:${column(headers.length - 1)}1`);
  header.format.fill = '#29445F';
  header.format.font = { name: 'Arial', size: 10, bold: true, color: '#FFFFFF' };
  header.format.horizontalAlignment = 'center';
  header.format.wrapText = true;
  header.format.rowHeight = 44;
  sheet.tables.add(`A1:${end}`, true, spec.table);
  for (let i = 0; i < headers.length; i++) {
    const key = headers[i], letter = column(i);
    const body = sheet.getRange(`${letter}2:${letter}${values.length}`);
    const whole = sheet.getRange(`${letter}1:${letter}${values.length}`);
    whole.format.columnWidth = key === 'video_id' ? 45 : key === 'notes' ? 45 :
      key === 'sample_id' ? 11 : key === 'frame_sample_id' ? 23 :
      ['image_path', 'video_path'].includes(key) ? 36 : 29;
    body.format.horizontalAlignment = 'left';
    if (spec.human.includes(key)) body.format.fill = '#FFF2CC';
    if (choices[key]) body.dataValidation = { rule: { type: 'list', values: choices[key] } };
    if (countFields.has(key)) {
      sheet.dataValidations.add({ range: `${letter}2:${letter}${values.length}`,
        rule: { type: 'whole', operator: 'between', formula1: 0, formula2: 10000 } });
      body.setNumberFormat('0');
      body.format.horizontalAlignment = 'right';
    }
    if (['frame_index', 'predicted_person_boxes'].includes(key)) {
      body.setNumberFormat('0'); body.format.horizontalAlignment = 'right';
    }
    if (key === 'time_seconds') { body.setNumberFormat('0.000'); body.format.horizontalAlignment = 'right'; }
  }
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(spec.name === 'Frame Audit' ? 2 : 1);
  const noteCol = column(headers.length + 1);
  const notes = sheet.getRange(`${noteCol}1:${noteCol}5`);
  notes.values = [
    ['黃色欄位供人工填寫；空白不是0。'],
    ['影片與影像路徑以 Audit 資料夾為基準。'],
    ['球與籃框目前由人工觀察，非 YOLO 偵測。'],
    ['詳細定義見專案 YOLO Audit README。'],
    ['來源：固定 train manifest 與本次逐幀偵測。'],
  ];
  notes.format.columnWidth = 65;
  notes.format.font = { name: 'Arial', size: 10, color: '#596678' };
  notes.format.wrapText = false;
  // CSV is exported from the actual workbook data region, excluding notes.
  const exported = region.values;
  if (exported.length !== values.length || exported[0].length !== headers.length)
    throw new Error('Workbook region shape mismatch');
  const csv = '\uFEFF' + exported.map(row => row.map(csvValue).join(',')).join('\n') + '\n';
  await fs.writeFile(path.join(outputDir, spec.csv), csv, { flag: 'wx' });
}
workbook.recalculate();
for (const name of ['Video Audit', 'Frame Audit']) {
  const result = await workbook.inspect({ kind: 'table', range: `'${name}'!A1:F4`,
    include: 'values,formulas', tableMaxRows: 4, tableMaxCols: 6, maxChars: 1800 });
  console.log(result.ndjson);
  const ranges = name === 'Video Audit' ? ['A1:H5', 'I1:P5'] : ['A1:H5', 'I1:Q5'];
  for (let i = 0; i < ranges.length; i++) {
    const preview = await workbook.render({ sheetName: name, range: ranges[i], scale: 1, format: 'png' });
    await fs.writeFile(path.join(outputDir, `${name.replaceAll(' ', '_')}_${i}.png`),
      new Uint8Array(await preview.arrayBuffer()));
  }
}
const errors = await workbook.inspect({ kind: 'match',
  searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',
  options: { useRegex: true, maxResults: 20 }, maxChars: 1800 });
console.log(errors.ndjson);
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(path.join(outputDir, 'annotation_book.xlsx'));
console.log(JSON.stringify({ output: outputDir, videoRows: 20, frameRows: 100,
  humanJudgmentsPrefilled: false, sheets: ['Video Audit', 'Frame Audit'] }));
