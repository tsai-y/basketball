// Read existing CSV via Artifact Tool; output JSON/JSONL only, no workbook edits.
import fs from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { Workbook } from '@oai/artifact-tool';

const FOLDER = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(FOLDER, '../..');
const PYTHON = '/home/tsai/miniconda3/envs/yolo11/bin/python';
export const VARIANTS = Object.freeze({
  aca_equal: { dense: .5, dw3d: .5, mainline: 0 },
  dense_mainline_equal: { dense: .5, dw3d: 0, mainline: .5 },
  dw3d_mainline_equal: { dense: 0, dw3d: .5, mainline: .5 },
  three_equal: { dense: 1/3, dw3d: 1/3, mainline: 1/3 },
  mainline_half_aca_half: { dense: .25, dw3d: .25, mainline: .5 },
});
export function requireValue(condition, message) { if (!condition) throw new Error(message); }
async function digest(path) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
}
async function verify(files) {
  for (const [path, expected] of Object.entries(files)) requireValue(await digest(path) === expected, `Input changed: ${path}`);
}
const readJson = async p => JSON.parse(await fs.readFile(p, 'utf8'));
const readRows = async p => (await fs.readFile(p, 'utf8')).trim().split('\n').map(JSON.parse);
async function writeNew(path, value) { await fs.writeFile(path, JSON.stringify(value, null, 2)+'\n', {flag:'wx'}); }
async function writeRows(path, rows) { await fs.writeFile(path, rows.map(x=>JSON.stringify(x)).join('\n')+'\n', {flag:'wx'}); }

export function records(matrix) {
  requireValue(matrix.length > 1 && new Set(matrix[0]).size === matrix[0].length, 'Invalid CSV header');
  return matrix.slice(1).map(row => Object.fromEntries(matrix[0].map((key, i) => [key, row[i]])));
}
async function readCsv(path) {
  const workbook = await Workbook.fromCSV(await fs.readFile(path, 'utf8'), {sheetName:'Source'});
  return records(workbook.worksheets.getItemAt(0).getUsedRange().values);
}
function metricMatrix(cm) {
  const count = cm.flat().reduce((a,b)=>a+b,0);
  const recall = i => cm[i][i]/(cm[i][0]+cm[i][1] || 1);
  const f1 = i => 2*cm[i][i]/(2*cm[i][i]+cm[i][1-i]+cm[1-i][i] || 1);
  return {count, accuracy:(cm[0][0]+cm[1][1])/count, macro_f1:(f1(0)+f1(1))/2,
    miss_recall:recall(0), make_recall:recall(1), confusion_rows_true_columns_predicted:cm};
}
export function summarize(rows) {
  requireValue(rows.length>0, 'Empty rows');
  const cm=[[0,0],[0,0]], shots={}; let loss=0;
  for (const row of rows) {
    requireValue([0,1].includes(row.label) && [0,1].includes(row.prediction) && Number.isFinite(row.ce_loss) && row.ce_loss>=-1e-12 &&
      Number.isFinite(row.p_make) && row.p_make>=0 && row.p_make<=1, 'Invalid predictions');
    cm[row.label][row.prediction]++;
    shots[row.shot_type] ??= [[0,0],[0,0]];
    shots[row.shot_type][row.label][row.prediction]++; loss+=row.ce_loss;
  }
  return {...metricMatrix(cm), loss:loss/rows.length,
    by_shot_type:Object.fromEntries(Object.entries(shots).map(([k,v])=>[k,metricMatrix(v)]))};
}
export function align(rows, expected) {
  requireValue(rows.length===expected.length, 'Incomplete prediction set');
  const index=new Map(rows.map(row=>[row.video_id,row]));
  requireValue(index.size===expected.length, 'Duplicate prediction IDs');
  return expected.map(source=>{
    const row=index.get(source.video_id);
    requireValue(row && row.split==='validation' && row.label===Number(source.binary_label) &&
      row.original_class===source.original_class && row.source_group===source.source_group &&
      row.shot_type===source.original_class.slice(0,-1), 'Foreign/test/misaligned metadata');
    return row;
  });
}
function equalMetrics(actual, expected, legacy=false) {
  requireValue(Math.abs(actual.accuracy-expected.accuracy)<1e-12 && Math.abs(actual.macro_f1-expected.macro_f1)<1e-12 &&
    Math.abs(actual.loss-expected.loss)<1e-6, 'Source metrics not reproduced');
  requireValue(JSON.stringify(actual.confusion_rows_true_columns_predicted) ===
    JSON.stringify(legacy ? expected.confusion_matrix : expected.confusion_rows_true_columns_predicted), 'Source confusion not reproduced');
}
export function verifyCsv(csv, rows) {
  requireValue(csv.length===rows.length, 'CSV missing rows');
  const map=new Map(csv.map(row=>[row.video_id,row]));
  requireValue(map.size===rows.length, 'CSV duplicate IDs');
  for (const row of rows) {
    const item=map.get(row.video_id);
    requireValue(item && item.true_label!=='' && item.p_make!=='' && item.predicted_label!=='' &&
      Number(item.true_label)===row.label && Number(item.predicted_label)===row.prediction &&
      Number.isFinite(Number(item.p_make)) && Number(item.p_make)===row.p_make,
      'CSV differs from saved best-checkpoint logits');
  }
}
export function verifyHistory(history, summary) {
  requireValue(history.length===summary.epochs_completed+1, 'Incomplete mainline history');
  history.forEach((row,i)=>requireValue(Number(row.epoch)===i && Number.isFinite(Number(row.val_loss)), 'Invalid epoch history'));
  const best=history.reduce((a,b)=>Number(b.val_loss)<Number(a.val_loss)?b:a);
  requireValue(Number(best.epoch)===summary.best_epoch && Math.abs(Number(best.val_loss)-summary.validation.loss)<1e-12,
    'Mainline min-loss selection mismatch');
}
export function fuse(models, weights) {
  const names=Object.keys(models).filter(n=>weights[n]>0);
  requireValue(names.length>1 && Object.values(weights).every(w=>Number.isFinite(w)&&w>=0) &&
    Math.abs(Object.values(weights).reduce((a,b)=>a+b,0)-1)<1e-12, 'Invalid fusion weights');
  return models[names[0]].map((base,i)=>{
    requireValue(names.every(n=>models[n][i].video_id===base.video_id && models[n][i].label===base.label), 'Unaligned fusion');
    let p=0; const logs=[];
    for (const n of names) {
      const row=models[n][i];
      requireValue(Array.isArray(row.log_probabilities) && row.log_probabilities.length===2 &&
        row.log_probabilities.every(Number.isFinite), 'Missing stable log probabilities');
      p+=weights[n]*row.p_make;
      logs.push(Math.log(weights[n])+row.log_probabilities[base.label]);
    }
    const high=Math.max(...logs), loss=-(high+Math.log(logs.reduce((s,x)=>s+Math.exp(x-high),0)));
    const {video_id,split,label,original_class,shot_type,source_group}=base;
    return {video_id,split,label,original_class,shot_type,source_group,p_make:p,prediction:Number(p>.5),ce_loss:loss};
  });
}
function changes(reference, candidate) {
  let rescued=0,harmed=0;
  reference.forEach((row,i)=>{
    const old=row.prediction===row.label, now=candidate[i].prediction===row.label;
    rescued+=Number(!old&&now); harmed+=Number(old&&!now);
  });
  return {rescued,harmed,net_correct:rescued-harmed};
}
export function diagnostics(models) {
  const names=Object.keys(models), patterns={}, bothAcaWrong={count:0,mainline_rescues:0,still_wrong:0,by_shot_type:{}};
  for (let i=0;i<models.dense.length;i++) {
    const bits=names.map(n=>Number(models[n][i].prediction===models[n][i].label)).join('');
    patterns[bits]=(patterns[bits]??0)+1;
    if (bits.startsWith('00')) {
      const shot=models.dense[i].shot_type; bothAcaWrong.count++;
      bothAcaWrong.by_shot_type[shot]??={count:0,mainline_rescues:0}; bothAcaWrong.by_shot_type[shot].count++;
      if (bits[2]==='1') {bothAcaWrong.mainline_rescues++;bothAcaWrong.by_shot_type[shot].mainline_rescues++;}
      else bothAcaWrong.still_wrong++;
    }
  }
  return {pattern_order:names,correctness_patterns:patterns,aca_both_wrong:bothAcaWrong,
    label_oracle_diagnostic:{accuracy:1-(patterns['000']??0)/models.dense.length,deployable:false,
      meaning:'Ground-truth chooses an already-correct class prediction; not an achieved score or general threshold/model upper bound'}};
}

export async function run() {
  // All variants are fixed before inspecting their outcomes; no seed/threshold/weight search.
  const captured=JSON.parse(execFileSync(PYTHON,['-B','-m','development.aca_mainline_fusion_v1.checkpoint'],
    {cwd:ROOT,env:{...process.env,CUDA_VISIBLE_DEVICES:''},maxBuffer:8*1024*1024,encoding:'utf8'}));
  const protectedFiles={...captured.protected_files_sha256};
  for (const name of ['checkpoint.py','test_checkpoint.py','fuse.mjs','test_fuse.mjs','__init__.py']) protectedFiles[join(FOLDER,name)]=await digest(join(FOLDER,name));
  await verify(protectedFiles);
  const original=await readJson(join(captured.aca_comparison,'report.json'));
  const expected=captured.expected_validation_rows;
  const mainline=align(captured.rows,expected);
  verifyCsv(await readCsv(join(captured.mainline_run,'validation_predictions.csv')),mainline);
  verifyHistory(await readCsv(join(captured.mainline_run,'history.csv')),captured.summary);
  const models={dense:align(await readRows(join(captured.aca_comparison,'dense_predictions.jsonl')),expected),
    dw3d:align(await readRows(join(captured.aca_comparison,'dw3d_predictions.jsonl')),expected),mainline};
  const metrics=Object.fromEntries(Object.entries(models).map(([n,rows])=>[n,summarize(rows)]));
  equalMetrics(metrics.dense,original.dense); equalMetrics(metrics.dw3d,original.dw3d);
  equalMetrics(metrics.mainline,captured.summary.validation,true);
  const artifacts=join(FOLDER,'artifacts'); await fs.mkdir(artifacts,{recursive:true});
  const output=await fs.mkdtemp(join(artifacts,'mainline_fusion_'));
  await writeNew(join(output,'provenance.json'),{protected_files_sha256:protectedFiles,mainline_run:captured.mainline_run,
    aca_comparison:captured.aca_comparison,mainline_checkpoint_best_epoch:captured.summary.best_epoch,
    mainline_is_pre_finetuning_epoch0:true,mainline_seed_fixed_not_best_seed:20261024,
    variants_declared_before_fusion:VARIANTS,threshold:'p_make > 0.5; ties Miss',
    mainline_csv_matches_saved_checkpoint:true,mainline_history_min_ce_verified:true,
    videos_decoded:0,optimizer_updates:0,test_videos_opened:0});
  try {
    await writeRows(join(output,'mainline_verified_predictions.jsonl'),mainline);
    const results={}, allRows={};
    for (const [name,weights] of Object.entries(VARIANTS)) {
      const rows=fuse(models,weights); allRows[name]=rows;
      results[name]={weights,...summarize(rows)};
      await writeRows(join(output,`${name}_predictions.jsonl`),rows);
    }
    equalMetrics(results.aca_equal,original.equal_weight_fusion);
    for (const name of Object.keys(results)) results[name].changes_vs_aca_equal=changes(allRows.aca_equal,allRows[name]);
    const candidates={...metrics,...results};
    const selected=Object.keys(candidates).reduce((a,b)=>candidates[b].loss<candidates[a].loss?b:a);
    await verify(protectedFiles);
    const report={status:'completed',scope:'validation_only_not_test',count:expected.length,
      single_models:metrics,fusions:results,diagnostics:diagnostics(models),
      exploratory_selection:{rule:'minimum_validation_CE; exact ties keep earlier listed candidate',
        selected,accuracy:candidates[selected].accuracy,loss:candidates[selected].loss,
        candidate_order:Object.keys(candidates),post_hoc_development_comparison:true},
      protected_files_unchanged:true,mainline_predictions_reproduced_from_checkpoint:true,
      weights_seeds_thresholds_searched:false,videos_decoded:0,optimizer_updates:0,test_videos_opened:0,
      caveat:'Repeated validation development, not independent test. Epoch0 mainline is not improved fine-tuned weights. No guarantee of test90.'};
    await writeNew(join(output,'report.json'),report);
    console.log(JSON.stringify({output,...report},null,2));
    return {output,report};
  } catch(error) {
    await writeNew(join(output,'failure.json'),{status:'failed',error:String(error),no_automatic_retry:true});
    throw error;
  }
}

if (process.argv[1] && import.meta.url===pathToFileURL(resolve(process.argv[1])).href) {
  requireValue(process.argv.slice(2).every(x=>x==='--execute'),'Only --execute is supported');
  if (process.argv.includes('--execute')) await run();
  else console.log(JSON.stringify({mode:'plan_only',variants:VARIANTS,no_training:true,no_test:true},null,2));
}
