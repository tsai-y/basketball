"""Project-only endpoint-preserving stratified50; not author-exact sampling."""
import hashlib
import json

VERSION = 'endpoint_stratified50_train_v1'
POLICY = dict(version=VERSION, frames=50, endpoints='always_0_and_N_minus_1',
    interior='48_disjoint_bins_[1+floor(j*(N-2)/48),1+floor((j+1)*(N-2)/48))',
    draw='one_uniform_integer_per_bin_SHA256_rejection_sampling',
    identity='JSON_[version,seed,zero_based_epoch,video_id,bin,counter]',
    train_only=True, validation='unchanged_floor(i*(N-1)/49)',
    short_video='reject_N_below_50', spatial='unchanged_full_frame_INTER_AREA_160x120',
    color='unchanged_RGB', normalization='unchanged_ImageNet_V1',
    author_exact_sampling_verified=False)

def indices(n, seed, epoch, video_id):
    if not (type(n) is int and n >= 50 and type(seed) is int and
            type(epoch) is int and 0 <= epoch < 30 and isinstance(video_id, str) and video_id):
        raise ValueError('Invalid stratified50 identity/count/epoch')
    result = [0]
    for j in range(48):
        lo, hi = 1+j*(n-2)//48, 1+(j+1)*(n-2)//48
        width = hi-lo
        limit = (1 << 256) - ((1 << 256) % width)
        counter = 0
        while True:
            key = json.dumps([VERSION,seed,epoch,video_id,j,counter],separators=(',',':')).encode()
            value = int.from_bytes(hashlib.sha256(key).digest(), 'big')
            if value < limit:
                result.append(lo+value%width)
                break
            counter += 1
    return tuple(result+[n-1])
