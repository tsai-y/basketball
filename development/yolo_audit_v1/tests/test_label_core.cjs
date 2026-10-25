// Pure logic tests only. Does not open or automate a browser.
const assert = require('node:assert/strict');
const core = require('../label_core.js');
const meta = [{image_id:'L001',image_sha256:'pixels',proposals:[{id:'D1',xyxy:[10,10,30,100]}]}];
const base = {schema:'review-v1',pilot_id:'pilot',protocol:'rules',classes:core.names,
 images:[{image_id:'L001',image_sha256:'pixels',status:'pending',scene:'unreviewed',
  annotation_complete:false,notes:'',annotations:[],proposal_states:{D1:'pending'},
  class_visibility:{active_player:'unreviewed',ball:'unreviewed',rim:'unreviewed'}}]};
assert(core.boxValid([0,0,320,240]));
assert(!core.boxValid([0,0,0,1]));
assert(!core.boxValid([0,0,NaN,1]));
assert.deepEqual(core.checkImport(base,base,meta),base);
assert(core.readyProblems(base.images[0]).length>0);
const reviewed = core.clone(base), row = reviewed.images[0];
Object.assign(row,{status:'reviewed',scene:'court_play',annotation_complete:true,
 annotations:[{id:'A1',class_id:0,xyxy:[10,10,30,100],source_proposal_id:'D1'}],
 proposal_states:{D1:'accepted'},class_visibility:{active_player:'present',ball:'absent',rim:'absent'}});
assert.equal(core.readyProblems(row).length,0);
assert.deepEqual(core.checkImport(base,reviewed,meta),reviewed);
const uncertain = core.clone(reviewed); uncertain.images[0].class_visibility.ball='uncertain';
assert.throws(()=>core.checkImport(base,uncertain,meta));
const unlinked = core.clone(reviewed); unlinked.images[0].annotations=[];
assert.throws(()=>core.checkImport(base,unlinked,meta));
const badMapping = core.clone(base); badMapping.classes=['person'];
assert.throws(()=>core.checkImport(base,badMapping,meta));
const excluded = core.clone(base); excluded.images[0].status='excluded';
assert.throws(()=>core.checkImport(base,excluded,meta));
excluded.images[0].notes='Cannot reliably localize the tiny ball';
assert.deepEqual(core.checkImport(base,excluded,meta),excluded);
const before = JSON.stringify(reviewed); core.checkImport(base,reviewed,meta);
assert.equal(JSON.stringify(reviewed),before);
console.log('Box geometry, pending-vs-reviewed, proposals, uncertainty, exclusions, class mapping, immutable import: PASS');

// Optional player attributes are not detection classes or readiness requirements.
for (const value of [null,'yes','no','uncertain']) {
  const r=core.clone(reviewed); r.role_metadata_protocol=core.roleProtocol;
  r.images[0].annotations[0].holding_ball=value;
  assert.deepEqual(core.checkImport(base,r,meta),r);
  assert.equal(core.readyProblems(r.images[0]).length,0);
}
for (const value of ['',true,false,0,1,[],{},'holder']) {
  const r=core.clone(reviewed); r.images[0].annotations[0].holding_ball=value;
  assert.throws(()=>core.checkImport(base,r,meta));
}
for(const classId of [1,2]) {
  const r=core.clone(reviewed);
  r.images[0].annotations.push({id:'B1',class_id:classId,xyxy:[0,0,2,2],source_proposal_id:null,holding_ball:null});
  assert.throws(()=>core.checkImport(base,r,meta));
}
const roles=core.clone(reviewed), objectBefore=core.clone(reviewed);
const timestamp='2026-10-25T00:00:00.000Z';
core.setHoldingBall(roles,'L001','A1','yes',timestamp);
assert.equal(roles.images[0].annotations[0].holding_ball,'yes');
assert.equal(roles.images[0].status,'reviewed');
assert.equal(roles.images[0].annotation_complete,true);
assert.equal(roles.role_annotation_events[0].previous,null);
assert.equal(roles.role_annotation_events[0].source,'human_role_editor');
assert.equal(roles.role_annotation_events[0].recorded_at_utc,timestamp);
core.setHoldingBall(roles,'L001','A1','uncertain',timestamp);
assert.equal(roles.images[0].status,'reviewed');
core.setHoldingBall(roles,'L001','A1',null,timestamp);
assert.equal(roles.images[0].annotations[0].holding_ball,null);
assert.equal(roles.role_annotation_events.length,3);
const roleBefore=JSON.stringify(roles);
assert.throws(()=>core.setHoldingBall(roles,'L001','A1','invalid',timestamp));
assert.throws(()=>core.setHoldingBall(roles,'wrong','A1','yes',timestamp));
assert.equal(JSON.stringify(roles),roleBefore);
const stripped=core.clone(roles); delete stripped.role_metadata_protocol; delete stripped.role_annotation_events;
delete stripped.images[0].annotations[0].holding_ball;
assert.deepEqual(stripped,objectBefore);
assert.deepEqual(core.checkImport(base,roles,meta),roles);
const badProtocol=core.clone(roles); badProtocol.role_metadata_protocol='other';
assert.throws(()=>core.checkImport(base,badProtocol,meta));
console.log('Optional holding_ball enum, legacy compatibility, human-edit provenance, independent detection completeness, no geometry changes: PASS');
