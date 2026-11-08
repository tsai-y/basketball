"""Read-only G2 metadata/contract audit; standard library only.

No video path is opened. No model, checkpoint, training, inference or test run.
"""
import ast
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load(relative):
    return json.loads((ROOT/relative).read_text())


def audit():
    report = load('docs/research/results/aca_eight_class_data_audit_v1.json')
    contract = load('docs/research/specs/aca_eight_class_contract_draft_v1.json')
    require(report['status']=='metadata_audit_completed_training_release_on_hold', 'Audit status changed')
    require(contract['status']=='draft_not_released_for_training' and
            not contract['training_authorized'] and not contract['executable_training_config'], 'Contract scope changed')
    for name, expected in report['source_sha256'].items():
        require(sha(ROOT/name)==expected, 'Pinned source changed: '+name)
    with (ROOT/'data/processed/dataset_manifest.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    summary = load('data/processed/split_summary.json')
    cached = load('docs/dataset_summary.json')
    require(len(rows)==len({r['video_id'] for r in rows})==len({r['video_path'] for r in rows})==10311,
            'Manifest ID/path uniqueness changed')
    expected = {k: report['counts'][k] for k in ('train','validation','test','excluded')}
    require(Counter(r['split'] for r in rows)==expected, 'Manifest split totals differ')
    require(Counter(r['original_class'] for r in rows)==cached['classes'], 'Cached raw class counts differ')
    classes = ['2p0','2p1','3p0','3p1','ft0','ft1','mp0','mp1']
    outcomes = [int(c[-1]) for c in classes]
    require(contract['class_order']==classes and contract['class_to_binary']==outcomes, 'Label mapping differs')
    groups = defaultdict(set)
    for row in rows:
        require(row['original_class'] in classes and int(row['binary_label'])==int(row['original_class'][-1]),
                'Invalid eight-to-binary metadata mapping')
        if row['split']!='excluded':
            require(summary['group_assignment'][row['source_group']]==row['split'], 'Source assignment differs')
            groups[row['source_group']].add(row['split'])
    require(len(groups)==50 and all(len(v)==1 for v in groups.values()), 'Source group overlaps')
    for index, cls in enumerate(classes):
        declared = report['classes'][index]
        require(declared['id']==index and declared['label']==cls and declared['outcome']==outcomes[index],
                'Class schema changed')
        for split in expected:
            count = sum(r['original_class']==cls and r['split']==split for r in rows)
            require(count==declared[split], f'Class/split count differs: {cls}/{split}')
        require(sum(declared[s] for s in expected)==declared['raw'], 'Class total differs')
    lookup = {r['video_id']:r for r in rows}
    conflicts = []; retained = []
    for group in cached['exact_duplicate_groups']:
        members = [lookup[i] for i in group]
        kept = [r for r in members if r['split']!='excluded']
        require(len(kept)<=1, 'Known byte duplicate has multiple retained copies')
        if len({r['original_class'] for r in members})>1:
            conflicts.append(members)
            retained.extend(kept)
    require(len(conflicts)==len(report['eight_class_conflict_groups'])==5, 'Conflict group count differs')
    require(len(retained)==len(report['unadjudicated_retained_groups'])==2, 'Unadjudicated retained count differs')
    require({r['video_id'] for r in retained}=={r['retained_id'] for r in report['unadjudicated_retained_groups']},
            'Unadjudicated IDs differ')
    with (ROOT/'docs/research/results/aca_eight_class_label_review_v1.csv').open(newline='') as stream:
        queue = list(csv.DictReader(stream))
    require(queue==report['unadjudicated_retained_groups'], 'Review CSV differs')
    # An exhaustive label-pair implication, not a measured classification score.
    for truth in range(8):
        for prediction in range(8):
            require(not truth==prediction or outcomes[truth]==outcomes[prediction], 'Mapping implication failed')
    data_tree = ast.parse((ROOT/'development/aca_net_v1/data.py').read_text())
    class_constant = next(ast.literal_eval(n.value) for n in data_tree.body if isinstance(n,ast.Assign)
                          and any(isinstance(t,ast.Name) and t.id=='CLASSES' for t in n.targets))
    require(list(class_constant)==classes, 'Existing data class order differs')
    require(report['this_turn']['raw_videos_opened']==report['this_turn']['test_runs']==0, 'Scope differs')
    return dict(schema='aca_eight_class_contract_metadata_QA_v1',status='passed_for_audit_only',
                training_release='hold',pinned_sources_verified=len(report['source_sha256']),
                manifest_rows_checked=len(rows),class_split_cells_checked=8*4,
                group_count=len(groups),known_duplicate_groups_checked=len(cached['exact_duplicate_groups']),
                eight_class_conflict_groups=5,retained_unadjudicated_sources=2,
                review_queue_matches_report=True,label_pair_implications_checked=64,
                existing_class_order_checked_by_AST=True,raw_video_access=0,models_created=0,
                new_inference_or_test=0,new_optimizer_updates=0,
                caveat='QA verifies the audit against saved metadata; does not adjudicate labels, re-decode, or release training')


if __name__ == '__main__':
    print(json.dumps(audit(),ensure_ascii=False,indent=2))
