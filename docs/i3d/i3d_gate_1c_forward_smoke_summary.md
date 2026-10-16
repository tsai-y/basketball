# I3D Gate 1C Forward Smoke Summary

Status: **PASS** — 2026-10-17 (Asia/Taipei).

| Path | Model input | `Mixed_5c` | GAP | Stride T / H×W | Finite / repeated deterministic |
| --- | --- | --- | --- | --- | --- |
| official synthetic | `[1,3,64,224,224]` | `[1,1024,8,7,7]` | `[1,1024]` | 8 / 32×32 | yes / yes |
| project synthetic | `[1,3,64,192,256]` | `[1,1024,8,6,8]` | `[1,1024]` | 8 / 32×32 | yes / yes |
| train real | `[1,3,64,192,256]` | `[1,1024,8,6,8]` | `[1,1024]` | 8 / 32×32 | yes / n/a |
| validation real | `[1,3,64,192,256]` | `[1,1024,8,6,8]` | `[1,1024]` | 8 / 32×32 | yes / n/a |

Strict load used 287 source tensors for 287 model slots with missing/unexpected/unused/shape mismatch counts all zero. The project adapter is the out-of-place float32 affine `x*2-1`; Data Pipeline Gate v1 remains identity `[0,1]`, so normalization happens exactly once at the I3D boundary.

Real-data access: train open/decode 1/1, validation 1/1, test **0/0**. Runtime activity: eval + inference mode, dropout disabled, BN statistics unchanged, optimizer 0, optimizer steps 0, backward 0, training 0.

Cross-framework endpoint parity is `NOT_RUN` because an exact pinned Sonnet 1.x runtime is unavailable. Exact conversion tensor mapping passed. See the machine-readable summary for full statistics and deterministic digests.
