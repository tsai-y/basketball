"""Fixed eight-class reconstruction preprocessing; no file/video/model access.

The caller must authorize and decode sources, bind display-order timestamps to
frames, and pass ONLY selected BGR arrays. This module never opens a video.
"""
from bisect import bisect_left
from fractions import Fraction

FRAMES = 50
TARGET_FPS = Fraction(25, 1)
SOURCE_HW = (240, 320)
OUTPUT_HW = (120, 160)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def require(value, message):
    if not value:
        raise ValueError(message)


def exact_time(value):
    require(isinstance(value, (int, str, Fraction)) and not isinstance(value, bool),
            'Timestamp must be an exact rational, integer, or rational/decimal string; no floats or missing PTS')
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError('Invalid timestamp') from error


def nearest_index(times, target):
    """Nearest display-time source index; exact ties choose the earlier frame."""
    pos = bisect_left(times, target)
    if pos == 0:
        return 0
    if pos == len(times):
        return len(times)-1
    return pos-1 if target-times[pos-1] <= times[pos]-target else pos


def temporal_plan(timestamps):
    """Map 50 uniform positions on a virtual 25Hz grid to native display PTS.

    Fractions are seconds. They must come from per-frame timestamps and stream
    time_base, not from average FPS. No encoding, interpolation or temporal pad.
    """
    times = tuple(exact_time(v) for v in timestamps)
    require(len(times) >= FRAMES, 'Fewer than 50 source frames')
    require(all(a < b for a, b in zip(times, times[1:])), 'Display timestamps must be strictly increasing')
    relative = tuple(t-times[0] for t in times)
    duration = relative[-1]
    grid_count = int(duration*TARGET_FPS)+1
    require(grid_count >= FRAMES, 'Virtual 25Hz grid has fewer than 50 positions')
    grid_indices = tuple(i*(grid_count-1)//(FRAMES-1) for i in range(FRAMES))
    targets = tuple(Fraction(i, 1)/TARGET_FPS for i in grid_indices)
    source_indices = tuple(nearest_index(relative,t) for t in targets)
    require(len(set(source_indices)) == FRAMES, 'Temporal mapping repeats a source frame; reject rather than pad')
    require(all(a < b for a,b in zip(source_indices,source_indices[1:])), 'Source indices must increase')
    errors = tuple(abs(relative[i]-target) for i,target in zip(source_indices,targets))
    return dict(source_frame_count=len(times),source_duration_seconds=str(duration),
                virtual_fps='25',virtual_frame_count=grid_count,
                virtual_indices=list(grid_indices),source_indices=list(source_indices),
                target_seconds=[str(t) for t in targets],
                selected_source_seconds=[str(relative[i]) for i in source_indices],
                absolute_time_errors_seconds=[str(e) for e in errors],
                max_absolute_time_error_seconds=str(max(errors)),
                full_clip_temporal_span=True,encoded_source_rewritten=False)


def spatial_normalize(selected_bgr):
    """50 native-size uint8 BGR frames -> contiguous float32 [3,50,120,160].

    Full-frame identity crop, 2:1 INTER_AREA resize, RGB, ImageNet V1 mean/std.
    Identical deterministic transform for train/validation; no augmentation.
    """
    import cv2
    import numpy as np
    require(isinstance(selected_bgr,np.ndarray) and selected_bgr.dtype==np.uint8 and
            selected_bgr.shape==(FRAMES,*SOURCE_HW,3), 'Expected 50 uint8 BGR frames of size 240x320')
    rgb = np.stack([cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),
                              (OUTPUT_HW[1],OUTPUT_HW[0]),interpolation=cv2.INTER_AREA)
                    for frame in selected_bgr])
    value = rgb.astype(np.float32)/np.float32(255.0)
    value = (value-np.asarray(MEAN,dtype=np.float32))/np.asarray(STD,dtype=np.float32)
    value = np.ascontiguousarray(value.transpose(3,0,1,2))
    require(value.shape==(3,FRAMES,*OUTPUT_HW) and value.dtype==np.float32 and
            np.isfinite(value).all(), 'Invalid normalized tensor')
    return value
