"""Separate train-only diagnostic adapter. Never approves candidate use in training."""
from dataclasses import dataclass, asdict
from fractions import Fraction

import torch

from development.data_pipeline.pipeline import GateError
from development.data_pipeline.multiwindow_v1 import (
    BasketballThreeWindowDataset, select_covering_windows, candidate_overlap_counts,
)
from development.data_quality.transforms.video_clip_transforms import ClipTransform, derived_seed, normalize_tensor


def need(ok, message):
    if not ok:
        raise ValueError(message)


@dataclass(frozen=True)
class IntervalRequest:
    interval: tuple[int, int] | None = None
    review_scope: str = 'unreviewed'
    target_identity_confirmed: bool = False
    uncertainty_reasons: tuple[str, ...] = ()
    event_completeness_verified: bool = False
    provenance: str = ''


def resolve_interval(frame_count, request=None):
    """Plan only; original dataset still owns fallback decoding and failures.

    A diagnostic crop needs explicit opt-in and reviewed target/boundaries. This
    does not turn AI boundary review into human event truth or a training policy.
    """
    need(type(frame_count) is int and frame_count > 0, 'Invalid source frame count')
    request = request or IntervalRequest()
    need(isinstance(request, IntervalRequest), 'Expected IntervalRequest')
    decision = dict(mode='fallback_original', reason='no_interval', requested_interval=request.interval,
                    effective_interval=None, relative_windows=None, absolute_windows=None,
                    request=asdict(request), training_use_approved=False)
    interval=request.interval
    if interval is None:
        if request.uncertainty_reasons:
            decision['reason']='uncertain:'+','.join(request.uncertainty_reasons)
        return decision
    if (not isinstance(interval,(tuple,list)) or len(interval)!=2
            or any(type(x) is not int for x in interval)
            or not 0<=interval[0]<interval[1]<=frame_count):
        decision['reason']='invalid_or_out_of_source_interval'; return decision
    a,b=interval;length=b-a
    if length<66:
        decision['reason']='interval_shorter_than_66';return decision
    try:
        relative=select_covering_windows(length)
    except ValueError:
        decision['reason']='three_windows_cannot_cover_interval';return decision
    absolute=tuple((a+start,a+end) for start,end in relative)
    need(all(a<=start<end<=b and end-start==64 for start,end in absolute),'Invalid translated windows')
    need(absolute[0][0]==a and absolute[-1][1]==b and
         all(left[1]>=right[0] for left,right in zip(absolute,absolute[1:])), 'Incomplete coverage')
    if request.uncertainty_reasons:
        decision['reason']='uncertain:'+','.join(request.uncertainty_reasons);return decision
    # A full-source identity path makes no temporal or semantic selection at all.
    if (a,b)==(0,frame_count):
        mode='identity_original';reason='full_source_interval_no_temporal_change'
    elif request.review_scope!='explicit_input_smoke_only' or not request.target_identity_confirmed:
        decision['reason']='candidate_not_enabled_for_diagnostic_crop';return decision
    else:
        mode='diagnostic_candidate';reason='explicit_candidate_mechanics_test_not_training_approval'
    decision.update(mode=mode,reason=reason,effective_interval=(a,b),
                    relative_windows=relative,absolute_windows=absolute)
    return decision


def apply_request(original, index, item, request):
    """Fallback returns all original fields/tensor untouched, plus audit metadata."""
    d=resolve_interval(item['frame_count'],request)
    d['original_windows']=item['windows']
    d['source_frame_count']=item['frame_count']
    d['temporal_resampling_used']=False;d['frames_concatenated_across_scenes']=False
    if d['mode']!='diagnostic_candidate':
        d['effective_interval']=(0,item['frame_count'])
        d['absolute_windows']=item['windows']
        d['relative_windows']=item['windows']
        return {**item,'adaptation':d}
    base=original.base;row=base.rows[index];n=item['frame_count']
    need(row['video_id']==item['video_id'] and int(row['binary_label'])==item['binary_label'],'Original row mismatch')
    need(base.split=='train','Diagnostic adapter is train-only')
    q=base.quality_index;need(q is not None,'Missing original train quality metadata')
    summary=q.summaries[row['video_id']]
    need(int(summary['frame_count'])==n,'Source/sidecar count mismatch')
    need(abs(float(Fraction(summary['fps_rational']))-item['fps'])<1e-4,'Source/sidecar FPS mismatch')
    transform=ClipTransform({k:v for k,v in base.config.raw['transform'].items() if k!='policy'},'reference')
    tensors=[];transform_records=[];decoder_records=[]
    for i,(start,end) in enumerate(d['absolute_windows']):
        clip,metadata,selected,actual_n,fps=base.decoder(row,base.guard,
            selection={'start_frame':start,'end_frame_exclusive':end},expected_frame_count=n,
            expected_fps_rational=summary['fps_rational'])
        need(actual_n==n and abs(fps-item['fps'])<1e-4,'Decoder source changed')
        need((selected['start_frame'],selected['end_frame_exclusive'])==(start,end),'Decoder selected wrong coordinates')
        array,record=transform(clip,seed=derived_seed(base.config.raw['seed'],row['video_id'],base.epoch,i,'reference'),
            split=base.split,sample_metadata={'start_frame':start,'end_frame_exclusive':end})
        array,norm=normalize_tensor(array,mode='identity')
        tensors.append(torch.from_numpy(array));transform_records.append({'transform':record,'normalization':norm})
        decoder_records.append(metadata)
    d['decoder_records']=decoder_records;d['transform_records']=transform_records
    return {**item,'tensor':torch.stack(tensors),'windows':d['absolute_windows'],
            'policy_version':'fixed_v4_candidate_three_windows_input_smoke_only_v1',
            'candidate_overlap_frames':torch.tensor(candidate_overlap_counts(d['absolute_windows'],q.segments.get(row['video_id'],[])),dtype=torch.int64),
            'adaptation':d}


class DiagnosticThreeWindowAdapter(torch.utils.data.Dataset):
    """Composition wrapper: no original-loader changes and no auto interval choice."""
    def __init__(self,config,*,split,video_ids,requests=None,epoch=0):
        if split!='train':
            raise GateError('This input smoke adapter accepts original train only; reject before config/path access')
        self.original=BasketballThreeWindowDataset(config,split=split,video_ids=video_ids,epoch=epoch)
        self.requests=dict(requests or {})
        need(set(self.requests).issubset({r['video_id'] for r in self.original.base.rows}),'Foreign interval request')

    def __len__(self):
        return len(self.original)

    def __getitem__(self,index):
        item=self.original[index]
        return apply_request(self.original,index,item,self.requests.get(item['video_id']))
