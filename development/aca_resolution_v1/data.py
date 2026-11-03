"""Paired direct-from-original resize with frozen scopes and uniform50 indices.

Do not upsample old 160x120 inputs. No monkey-patching frozen loader globals.
"""
import math
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from development.aca_net_v1.data import VideoScope, uniform_indices, MEAN, STD, require
from development.data_pipeline.pipeline import gate1a

RESOLUTIONS = ((120,160),(192,256))


def check_resolution(size):
    require(type(size) in (list,tuple) and len(size)==2 and all(type(v) is int for v in size), 'Expected integer H,W')
    require(tuple(size) in RESOLUTIONS, 'Only fixed control/candidate resolutions accepted')
    return tuple(size)


def normalize(rgb):
    require(isinstance(rgb,np.ndarray) and rgb.dtype==np.uint8 and rgb.ndim==4 and rgb.shape[0]==50 and rgb.shape[-1]==3,
            'Expected 50 uint8 RGB frames')
    check_resolution(rgb.shape[1:3])
    x=torch.from_numpy(np.ascontiguousarray(rgb)).permute(3,0,1,2).float()/255.
    return ((x-x.new_tensor(MEAN)[:,None,None,None])/x.new_tensor(STD)[:,None,None,None]).contiguous()


def identity(stat):
    return stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns


def decode_pair(scope,video_id):
    row,path=scope.authorize(video_id)  # authorization precedes filesystem access
    require(scope.split=='train' or scope.split=='validation','Forbidden split')
    fd=gate1a.open_guarded(path);cap=None
    try:
        before=identity(os.fstat(fd));cap=cv2.VideoCapture(f'/proc/self/fd/{fd}',cv2.CAP_FFMPEG)
        require(cap.isOpened(),'Cannot decode authorized video')
        count,fps=float(cap.get(cv2.CAP_PROP_FRAME_COUNT)),float(cap.get(cv2.CAP_PROP_FPS))
        require(math.isfinite(count) and count>=50 and abs(count-round(count))<1e-6 and math.isfinite(fps) and fps>0,
                'Invalid/short source metadata')
        n=round(count);indices=uniform_indices(n);wanted=set(indices);retained={size:[] for size in RESOLUTIONS}
        for index in range(n):
            ok,bgr=cap.read()
            require(ok and isinstance(bgr,np.ndarray) and bgr.dtype==np.uint8 and bgr.shape==(240,320,3),
                    f'Invalid original frame {index}')
            if index in wanted:
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                for h,w in RESOLUTIONS:retained[(h,w)].append(cv2.resize(rgb,(w,h),interpolation=cv2.INTER_AREA))
        require(not cap.read()[0] and identity(os.fstat(fd))==before,'Frame count/source identity changed')
        require(all(len(x)==50 for x in retained.values()),'Incomplete uniform50')
        metadata={'frame_count':n,'fps':fps,'selected_indices':list(indices),'selected_times_seconds':[i/fps for i in indices],
                  'video_open_count':1,'decoded_frames':n,'both_resolutions_from_same_original_decode':True,
                  'crop':'none','color':'RGB','interpolation':'INTER_AREA','source_unchanged_during_decode':True}
        return {size:np.stack(values) for size,values in retained.items()},metadata,row
    finally:
        if cap is not None:cap.release()
        os.close(fd)


class ResolutionDataset(Dataset):
    """For future integration only; no training CLI is provided by this module."""
    def __init__(self,*,source_root,split,resolution,video_ids=None):
        # Reject test before resolution or source-root access.
        require(split in ('train','validation'),'Only train/validation accepted')
        self.resolution=check_resolution(resolution)
        self.scope=VideoScope(source_root=source_root,split=split,video_ids=video_ids)

    def __len__(self):return len(self.scope.rows)

    def __getitem__(self,index):
        video_id=self.scope.rows[index]['video_id'];pair,metadata,row=decode_pair(self.scope,video_id)
        return {'tensor':normalize(pair[self.resolution]),'binary_label':int(row['binary_label']),
                'video_id':video_id,'original_class':row['original_class'],'source_group':row['source_group'],
                'split':row['split'],'metadata':{**metadata,'resolution_height_width':list(self.resolution)}}
