// Freeze an already verified analysis candidate; no inference/test entry point.
import fs from 'node:fs/promises';
import {createReadStream} from 'node:fs';
import {createHash} from 'node:crypto';
import {dirname,join,resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const folder=dirname(fileURLToPath(import.meta.url));
const root=resolve(folder,'../..');
const output=join(folder,'artifacts/mainline_fusion_4Lbbzt');
const read=async path=>JSON.parse(await fs.readFile(path,'utf8'));
async function sha(path) {
  const h=createHash('sha256');for await(const chunk of createReadStream(path))h.update(chunk);
  return h.digest('hex');
}
const report=await read(join(output,'report.json')), provenance=await read(join(output,'provenance.json'));
if(report.status!=='completed'||report.exploratory_selection.selected!=='three_equal'||!report.protected_files_unchanged)
  throw Error('Expected completed, verified three-equal comparison');
for(const [path,hash] of Object.entries(provenance.protected_files_sha256))
  if(await sha(path)!==hash)throw Error(`Protected input changed: ${path}`);
const runs={
  dense:join(root,'development/aca_extension_v1/artifacts/aca_extend15_seed20261025_wuaps2g9'),
  dw3d:join(root,'development/aca_dw3d_v1/artifacts/aca_dw3d_train_seed20261025_zmjmo2o1'),
  mainline:provenance.mainline_run,
};
const components={};
for(const [name,run]of Object.entries(runs)) {
  const config=await read(join(run,'config.json')), summary=await read(join(run,'summary.json'));
  components[name]={checkpoint:join(run,'best.pt'),checkpoint_sha256:await sha(join(run,'best.pt')),
    config:join(run,'config.json'),config_sha256:await sha(join(run,'config.json')),
    best_epoch:summary.best_epoch,weight:report.fusions.three_equal.weights[name],
    probability:name==='mainline'?'sigmoid(binary_logit)':'softmax(two_logits)[class_1_Make]',
    preprocessing:name==='mainline'?config.input:config.policy,
    label_mapping:{Miss:0,Make:1}};
}
const candidate={schema:'basketball_prediction_fusion_candidate_v1',selected_method:'three_equal',
  readiness:'validated_on_saved_validation_predictions; full_video_fused_inference_and_test_not_implemented',
  components,aggregation:'arithmetic weighted mean of Make probabilities; not logits',
  prediction:'Make iff fused_p_make > 0.5; exact tie Miss',source_root:'/home/tsai/projects/basketball',
  validation:report.fusions.three_equal,selection:'minimum_validation_CE_among_predeclared_candidates',
  post_hoc_repeated_validation_development:true,mainline_epoch0_not_improved_finetuning:true,
  report_sha256:await sha(join(output,'report.json')),provenance_sha256:await sha(join(output,'provenance.json')),
  selected_predictions:join(output,'three_equal_predictions.jsonl'),
  selected_predictions_sha256:await sha(join(output,'three_equal_predictions.jsonl')),
  freeze_script_sha256:await sha(fileURLToPath(import.meta.url)),
  test_evaluated:false,guarantees_test90:false,model_parameters_modified:false};
await fs.writeFile(join(output,'candidate.json'),JSON.stringify(candidate,null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify({candidate:join(output,'candidate.json'),accuracy:candidate.validation.accuracy,
  weights:Object.fromEntries(Object.entries(components).map(([n,x])=>[n,x.weight])),test_evaluated:false},null,2));
