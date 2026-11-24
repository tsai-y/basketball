"""New train adapter; unchanged temporal decoder and fixed validation delegate."""
from development.aca_temporal_v1 import data as parent
from development.aca_binary_v1.runtime import EpochStream as BaseStream
from . import augment

class SpatialDataset(parent.TemporalDataset):
    def __getitem__(self,index):
        if self.split=='validation':
            return super().__getitem__(index)
        parent.need(self.split=='train' and self.epoch is not None,'Explicit train epoch required')
        vid=self._ids[index];ref=self.revision.record(vid,'train')
        source_sha=parent.labels.f.baseline.guarded_source_sha256(self._base.scope,vid)
        parent.need(source_sha==ref['source_sha256'],'Source SHA changed')
        rgb,m,row=parent.decode_stratified50(self._base.scope,vid,parent.SEED,self.epoch)
        parent.need(row['original_class']==ref['original_class'] and row['source_group']==ref['source_group'],'Source provenance mismatch')
        tensor,spatial=augment.transform(rgb,'train',parent.SEED,self.epoch,vid)
        m.update(data_version=parent.labels.VERSION,label_revision_sha256=parent.labels.MANIFEST_SHA,
            label_patch_sha256=parent.labels.PATCH_SHA,source_sha256=source_sha,tensor_sha256=parent.tensor_sha(tensor),
            temporal_sampling_version=parent.sampling.VERSION,temporal_seed=parent.SEED,epoch_index=self.epoch,
            pixel_contract_parent_sha256=parent.labels.f.CONTRACT_SHA256,
            fixed_uniform_reference_tensor_sha256=ref['tensor_sha256'],input_contract_sha256=None,
            fixed_uniform_tensor_match_claimed=False,times_are_nominal_index_over_fps_not_verified_pts=True,
            spatial_augmentation=spatial)
        return dict(tensor=tensor,video_id=vid,split='train',source_group=ref['source_group'],class_id=ref['class_id'],
            binary_label=ref['binary_label'],original_class=ref['original_class'],original_class_id=ref['original_class_id'],
            effective_class=ref['effective_class'],metadata=m)

class SpatialCollator(parent.TemporalCollator):
    def __call__(self,items):
        parent.need(bool(items),'Empty batch')
        if items[0]['split']=='train':
            for item in items:
                m=item['metadata'];s=m.get('spatial_augmentation')
                expected=dict(version=augment.VERSION,policy_sha256=augment.POLICY_SHA,seed=parent.SEED,
                    epoch_index=m['epoch_index'],video_id=item['video_id'],
                    parameters=augment.parameters(parent.SEED,m['epoch_index'],item['video_id']))
                parent.need(s==expected,'Stale/incorrect spatial identity')
        else:
            parent.need(all('spatial_augmentation' not in x['metadata'] for x in items),'Validation spatial metadata forbidden')
        batch=super().__call__(items)
        if items[0]['split']=='train':batch['spatial_policy_sha256']=augment.POLICY_SHA
        return batch

class EpochStream(parent.EpochStream):
    def __init__(self,datasets,plan,state_getter,*,workers=2):
        parent.need(plan['seed']==parent.SEED and plan['temporal_sampling']==parent.sampling.POLICY and
                    plan['spatial_augmentation']==augment.POLICY,'Sampling policy mismatch')
        parent.labels.require_revision_identity(plan)
        BaseStream.__init__(self,datasets,plan,state_getter,collate=SpatialCollator(),workers=workers)
