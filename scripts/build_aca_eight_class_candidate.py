"""Prepare or verify an isolated eight-class METADATA candidate, never training.

All source CSV/JSON files are read-only. No raw video path is opened. Candidate
CSV preserves old labels/splits in separate columns and is not an active input.
"""
import argparse
import csv
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY = 'docs/research/specs/aca_eight_class_split_policy_v1.json'
FIELDS = ['video_id','video_path','original_class','class_id','binary_label','source_group',
          'duplicate_group','original_split','original_excluded','original_exclusion_reason',
          'cohort_status','cohort_exclusion_reason','candidate_split']


def require(value, message):
    if not value:
        raise ValueError(message)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()


def load_inputs():
    policy = json.loads((ROOT/POLICY).read_text())
    require(policy['schema']=='aca_eight_class_split_policy_v1' and
            policy['training_authorized'] is False and policy['activate_existing_trainer'] is False,
            'Unexpected policy or activation request')
    for name, expected in policy['input_sha256'].items():
        require(sha_bytes((ROOT/name).read_bytes())==expected, 'Pinned input changed: '+name)
    with (ROOT/'data/processed/dataset_manifest.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    cached = json.loads((ROOT/'docs/dataset_summary.json').read_text())
    require(len(rows)==10311 and len({r['video_id'] for r in rows})==len(rows), 'Unexpected raw manifest')
    return policy, rows, cached


def cohort(rows, cached, classes):
    lookup = {r['video_id']:r for r in rows}
    reasons = {}
    for members in cached['exact_duplicate_groups']:
        require(len(members)==len(set(members)) and all(i in lookup for i in members), 'Invalid duplicate IDs')
        require(not set(members).intersection(reasons), 'Overlapping duplicate groups')
        if len({lookup[i]['original_class'] for i in members})>1:
            reasons.update({i:'ambiguous_eight_class_label' for i in members})
        else:
            reasons.update({i:'duplicate_copy' for i in sorted(members)[1:]})
    prepared=[]
    for row in sorted(rows,key=lambda r:r['video_id']):
        cls = row['original_class']
        require(cls in classes and row['binary_label']==cls[-1], 'Label mapping differs')
        reason = reasons.get(row['video_id'],'')
        prepared.append(dict(video_id=row['video_id'],video_path=row['video_path'],original_class=cls,
                             class_id=str(classes.index(cls)),binary_label=row['binary_label'],
                             source_group=row['source_group'],duplicate_group=row['duplicate_group'],
                             original_split=row['split'],original_excluded=row['excluded'],
                             original_exclusion_reason=row['exclusion_reason'],
                             cohort_status='excluded' if reason else 'included',
                             cohort_exclusion_reason=reason,candidate_split='excluded' if reason else ''))
    return prepared


def choose_split(prepared, policy):
    cfg = policy['split']
    classes = policy['class_order']
    splits = ('train','validation','test')
    groups = defaultdict(lambda:[0]*len(classes))
    for row in prepared:
        if row['cohort_status']=='included':groups[row['source_group']][int(row['class_id'])]+=1
    require(len(groups)==sum(cfg['group_quotas'].values())==50, 'Unexpected included group count')
    totals = [sum(g[c] for g in groups.values()) for c in range(len(classes))]
    total = sum(totals)
    best = None
    valid_trials=0
    for trial in range(cfg['metadata_candidates']):
        ranked = sorted(groups,key=lambda g:(hashlib.sha256(
            f"{cfg['ranking_namespace']}:{cfg['seed']}:{trial}:{g}".encode()).digest(),g))
        assignment = {}
        offset = 0
        for split in splits:
            for group in ranked[offset:offset+cfg['group_quotas'][split]]:assignment[group]=split
            offset += cfg['group_quotas'][split]
        counts = {s:[0]*len(classes) for s in splits}
        for group,vector in groups.items():
            bucket=counts[assignment[group]]
            for c,value in enumerate(vector):bucket[c]+=value
        if any(value==0 for vector in counts.values() for value in vector):continue
        valid_trials+=1
        # Equal treatment of eight classes; only metadata, no predictions/scores.
        score = Fraction(0)
        for split in splits:
            ratio = Fraction(*cfg['ratios'][split])
            score += abs(Fraction(sum(counts[split]),total)-ratio)
            score += sum(abs(Fraction(counts[split][c],totals[c])-ratio)
                         for c in range(len(classes)))/len(classes)
        tie = sha_bytes(canonical(assignment))
        key = (score,tie,trial)
        if best is None or key<best[0]:best=(key,assignment,counts,trial)
    require(best is not None, 'No class-covered metadata assignment')
    (score,_,_),assignment,counts,trial=best
    for row in prepared:
        if row['cohort_status']=='included':row['candidate_split']=assignment[row['source_group']]
    return dict(group_assignment=dict(sorted(assignment.items())),class_counts=counts,selected_trial=trial,
                valid_metadata_candidates=valid_trials,score_fraction=f'{score.numerator}/{score.denominator}',
                score_decimal=float(score))


def derive(policy, rows, cached):
    prepared = cohort(rows,cached,policy['class_order'])
    result = choose_split(prepared,policy)
    require(Counter(r['cohort_exclusion_reason'] for r in prepared if r['cohort_status']=='excluded')==
            {'ambiguous_eight_class_label':10,'duplicate_copy':24}, 'Unexpected exclusions')
    require(sum(r['cohort_status']=='included' for r in prepared)==10277,'Unexpected cohort count')
    by_id = {r['video_id']:r for r in prepared}
    for group in cached['exact_duplicate_groups']:
        kept = [by_id[i] for i in group if by_id[i]['cohort_status']=='included']
        require(len(kept)<=1,'Known exact duplicates retained more than once')
        if len({by_id[i]['original_class'] for i in group})>1:
            require(not kept,'Ambiguous eight-class group retained')
    group_splits = defaultdict(set)
    for row in prepared:
        if row['cohort_status']=='included':group_splits[row['source_group']].add(row['candidate_split'])
    require(all(len(v)==1 for v in group_splits.values()),'Source crosses candidate splits')
    table={old:{new:sum(r['original_split']==old and r['candidate_split']==new for r in prepared)
                for new in ('train','validation','test','excluded')}
           for old in ('train','validation','test','excluded')}
    result.update(counts=dict(Counter(r['candidate_split'] for r in prepared)),
                  exclusions={'ambiguous_eight_class_label':10,'duplicate_copy':24},
                  included=10277,raw=10311,group_counts=dict(Counter(result['group_assignment'].values())),
                  old_to_candidate_split_counts=table,
                  newly_quarantined=[r['video_id'] for r in prepared if r['original_excluded']=='0' and r['cohort_status']=='excluded'],
                  original_excluded_reintroduced=[r['video_id'] for r in prepared if r['original_excluded']=='1' and r['cohort_status']=='included'])
    require(len(result['newly_quarantined'])==2 and not result['original_excluded_reintroduced'],
            'Unexpected cohort differences')
    return prepared,result


def encode_csv(rows):
    text=io.StringIO(newline='')
    writer=csv.DictWriter(text,fieldnames=FIELDS,lineterminator='\n')
    writer.writeheader();writer.writerows(rows)
    return text.getvalue().encode()


def prepare(output):
    policy,raw,cached=load_inputs()
    output=output.absolute()
    allowed=(ROOT/'development/aca_eight_class_data_v1/artifacts').resolve()
    require(output.parent.resolve()==allowed,'Output must be one direct child of the isolated artifact directory')
    require(not output.exists(),'Output exists; refusing to overwrite')
    rows,result=derive(policy,raw,cached)
    payload=encode_csv(rows)
    plan=canonical(policy)
    report=dict(schema='aca_eight_class_split_candidate_v1',status='metadata_candidate_not_active',
                policy_sha256=sha_bytes(plan),builder_sha256=sha_bytes(Path(__file__).read_bytes()),
                manifest_sha256=sha_bytes(payload),exact_author_split=False,training_authorized=False,
                globally_unseen_test=False,original_manifest_overwritten=False,
                raw_video_access=0,predictions_or_model_scores_used=0,
                artifact_directory=output.relative_to(ROOT).as_posix(),**result)
    output.mkdir(parents=True,exist_ok=False)
    (output/'policy.json').write_bytes(plan+b'\n')
    (output/'candidate_manifest.csv').write_bytes(payload)
    (output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


def verify(output):
    policy,raw,cached=load_inputs()
    saved=json.loads((output/'summary.json').read_text())
    require(json.loads((output/'policy.json').read_text())==policy,'Saved policy differs')
    expected,result=derive(policy,raw,cached)
    require((output/'candidate_manifest.csv').read_bytes()==encode_csv(expected),
            'Candidate CSV differs from deterministic derivation')
    require(saved['manifest_sha256']==sha_bytes(encode_csv(expected)) and
            saved['policy_sha256']==sha_bytes(canonical(policy)) and
            saved['builder_sha256']==sha_bytes(Path(__file__).read_bytes()),'Candidate fingerprints differ')
    require(all(saved[k]==v for k,v in result.items()),'Candidate summary differs')
    require(saved['training_authorized'] is False and saved['status']=='metadata_candidate_not_active'
            and saved['globally_unseen_test'] is False,'Candidate scope differs')
    return dict(status='verified_metadata_candidate_only',rows_verified=len(expected),
                manifest_sha256=saved['manifest_sha256'],source_group_overlap=0,
                ambiguous_groups_retained=0,raw_video_access=0,training_jobs=0,test_runs=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--output',type=Path)
    mode.add_argument('--verify',type=Path)
    args=parser.parse_args()
    result=prepare(args.output) if args.output else verify(args.verify)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
