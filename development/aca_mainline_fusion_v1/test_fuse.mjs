import test from 'node:test';
import assert from 'node:assert/strict';
import {align,diagnostics,fuse,records,summarize,verifyCsv,verifyHistory} from './fuse.mjs';

const expected=[0,1,0,1].map((label,i)=>({video_id:String(i),binary_label:String(label),original_class:`ft${label}`,source_group:String(i)}));
const rows=ps=>ps.map((p,i)=>({...expected[i],split:'validation',label:Number(expected[i].binary_label),
  shot_type:'ft',p_make:p,prediction:Number(p>.5),ce_loss:-(Number(expected[i].binary_label)?Math.log(p):Math.log1p(-p)),
  log_probabilities:[Math.log1p(-p),Math.log(p)]}));

test('ID keyed alignment, reordered rows',()=>{
  const a=rows([.1,.9,.1,.9]); assert.deepEqual(align([...a].reverse(),expected),a);
});
test('duplicate/missing/test IDs and labels refused',()=>{
  const a=rows([.1,.9,.1,.9]);
  for (const bad of [a.slice(1),[a[0],a[0],a[2],a[3]],a.map(x=>({...x,split:'test'})),a.map(x=>({...x,label:1-x.label}))])
    assert.throws(()=>align(bad,expected));
});
test('CSV agrees with checkpoint, duplicate and changed values refused',()=>{
  const a=rows([.1,.9,.1,.9]), csv=a.map(x=>({video_id:x.video_id,true_label:String(x.label),p_make:String(x.p_make),predicted_label:String(x.prediction)}));
  verifyCsv([...csv].reverse(),a);
  assert.throws(()=>verifyCsv(csv.map(x=>({...x,p_make:'.5'})),a));
  assert.throws(()=>verifyCsv(csv.map(x=>({...x,p_make:String(Number(x.p_make)+1e-9)})),a));
  assert.throws(()=>verifyCsv([csv[0],csv[0],csv[2],csv[3]],a));
});
test('epoch0 and strict min CE, not maximum accuracy',()=>{
  const h=[{epoch:'0',val_loss:'.4'},{epoch:'1',val_loss:'.5'}];
  verifyHistory(h,{epochs_completed:1,best_epoch:0,validation:{loss:.4}});
  assert.throws(()=>verifyHistory(h,{epochs_completed:1,best_epoch:1,validation:{loss:.5}}));
});
test('equal probability average, ties choose Miss, no logit averaging',()=>{
  const a=rows([.2,.2,.2,.2]),b=rows([.8,.8,.8,.8]);
  const out=fuse({a,b},{a:.5,b:.5}); assert.ok(out.every(x=>x.p_make===.5&&x.prediction===0));
  assert.ok(Math.abs(summarize(out).loss-Math.log(2))<1e-12);
});
test('stable CE with saturated probabilities',()=>{
  const a=rows([.2,.2,.2,.2]).map(x=>({...x,p_make:1,log_probabilities:[-1000,0]}));
  assert.ok(Math.abs(fuse({a,b:a},{a:.5,b:.5})[0].ce_loss-1000)<1e-9);
});
test('weights must sum to one and require multiple models',()=>{
  const a=rows([.1,.9,.1,.9]);
  assert.throws(()=>fuse({a,b:a},{a:.3,b:.3})); assert.throws(()=>fuse({a,b:a},{a:1,b:0}));
});
test('nonfinite predictions refused',()=>assert.throws(()=>summarize(rows([.1,.9,.1,NaN]))));
test('correctness patterns and mainline rescue of common ACA error',()=>{
  const a=rows([.1,.9,.9,.1]),b=rows([.1,.1,.1,.1]),m=rows([.1,.9,.1,.9]);
  const d=diagnostics({dense:a,dw3d:b,mainline:m});
  assert.equal(d.aca_both_wrong.count,1); assert.equal(d.aca_both_wrong.mainline_rescues,1);
  assert.equal(d.label_oracle_diagnostic.accuracy,1);assert.equal(d.label_oracle_diagnostic.deployable,false);
});
test('CSV header and string records are handled explicitly',()=>{
  assert.deepEqual(records([['id','p'],['x','.5']]),[{id:'x',p:'.5'}]);
  assert.throws(()=>records([['id','id'],['a','b']]));
});
