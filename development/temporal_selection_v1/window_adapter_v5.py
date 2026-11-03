"""Frozen-v5 train-only input diagnostic, not an automatically adopted crop policy.

Compose the original loader and reuse the tested coordinate/decoder/transform
mechanics without modifying the v4 adapter. A cropped request must name an exact
clean span in its frozen v5 record, and still needs explicit input-smoke opt-in.
Event completeness and production adoption are never inferred from that span.
"""
from dataclasses import asdict,replace
import hashlib
import json

import torch

from development.data_pipeline.pipeline import GateError
from development.data_pipeline.multiwindow_v1 import BasketballThreeWindowDataset
from .selector_v5 import ExposurePolicyV5
from .window_adapter_v1 import IntervalRequest,apply_request,need


def validate_record(record):
    need(isinstance(record,dict) and record.get('original_split')=='train','Need a frozen original-train record')
    need(record['v5']['policy']==asdict(ExposurePolicyV5()),'Frozen v5 policy mismatch')
    need(not record['v5'].get('training_use_approved',True),'Record unexpectedly approves training')
    need(type(record['frame_count']) is int and record['frame_count']>0,'Invalid frozen source length')


def apply_v5_request(original,index,item,request,record,source_report_sha256):
    validate_record(record);request=request or IntervalRequest()
    need(isinstance(request,IntervalRequest),'Expected explicit IntervalRequest')
    need(record['video_id']==item['video_id'] and record['frame_count']==item['frame_count'],
         'Original item differs from frozen v5 source')
    requested=request.interval
    clean={(s['start_frame'],s['end_frame_exclusive']) for s in record['v5']['all_segments_including_short']
           if not s['inside_proposed_change_span']}
    pair=(tuple(requested) if isinstance(requested,(tuple,list)) and len(requested)==2
          and all(type(v) is int for v in requested) else None)
    is_clean=pair in clean if pair is not None else False
    identity_control=pair==(0,item['frame_count'])
    effective=request
    if pair is not None and not identity_control and not is_clean:
        effective=replace(request,uncertainty_reasons=(*request.uncertainty_reasons,
                          'requested_interval_is_not_exact_frozen_v5_clean_span'))
    result=apply_request(original,index,item,effective)
    d=result['adaptation']
    d.update(source_policy_version=record['v5']['policy']['version'],
             source_report_sha256=source_report_sha256,
             frozen_v5_record_sha256=hashlib.sha256(json.dumps(record['v5'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),
             originating_request=asdict(request),requested_interval_matches_frozen_clean_span=is_clean,
             full_source_identity_control=identity_control,
             coordinate_system='zero_based_original_source_half_open_[start,end)',
             event_completeness_verified=request.event_completeness_verified,
             production_crop_adopted=False,
             fallback_same_tensor_object=result['tensor'] is item['tensor'])
    if d['mode']=='diagnostic_candidate':
        need(is_clean,'A diagnostic crop cannot bypass frozen v5 scene evidence')
        result['policy_version']='fixed_v5_candidate_three_windows_input_smoke_only_v1'
    return result


class V5DiagnosticThreeWindowAdapter(torch.utils.data.Dataset):
    """Train-only composition; no source selection or event classification."""
    def __init__(self,config,*,split,video_ids,v5_records,source_report_sha256,requests=None,epoch=0):
        if split!='train':
            raise GateError('v5 input smoke is original-train-only; reject before config/path access')
        ids=list(video_ids)
        need(len(ids)==len(set(ids)) and set(v5_records)==set(ids),'Exact selected-v5-record identity required')
        for v,record in v5_records.items():
            validate_record(record);need(v==record['video_id'],'Frozen v5 record key mismatch')
        self.requests=dict(requests or {})
        need(set(self.requests)<=set(ids) and all(isinstance(r,IntervalRequest) for r in self.requests.values()),
             'Foreign/invalid v5 request')
        need(isinstance(source_report_sha256,str) and len(source_report_sha256)==64 and
             all(c in '0123456789abcdef' for c in source_report_sha256),'Expected source report SHA-256')
        self.original=BasketballThreeWindowDataset(config,split=split,video_ids=ids,epoch=epoch)
        self.v5_records=dict(v5_records);self.source_report_sha256=source_report_sha256
        for row in self.original.base.rows:
            record=self.v5_records[row['video_id']]
            need(self.original.base.guard.identity(row['video_id'])==record['source_identity']['before'],
                 'Frozen v5 source identity changed')

    def __len__(self):return len(self.original)

    def __getitem__(self,index):
        item=self.original[index];record=self.v5_records[item['video_id']]
        need(self.original.base.guard.identity(item['video_id'])==record['source_identity']['before'],
             'Source changed since frozen v5 analysis')
        return apply_v5_request(self.original,index,item,self.requests.get(item['video_id']),record,self.source_report_sha256)
