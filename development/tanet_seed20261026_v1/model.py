"""Thin binary adapter around the author's unchanged ResNet50/TAM injection.

No automatic downloads; no ACA/Triplet/gating; no temporal interpolation.
This is an author-code adaptation to this project, not their complete recipe.
"""
from contextlib import redirect_stdout
import io
from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnet50

from .config import InputConfig, need, sha256, verify_upstream
from .upstream.temporal_module import make_temporal_modeling


class TANetBinary(nn.Module):
    def __init__(self, *, frames=50, dropout=0.5, checkpoint=None,
                 checkpoint_sha256=None, checkpoint_size=None):
        super().__init__()
        InputConfig(frames=frames)
        need(type(dropout) in (float, int) and 0 <= dropout < 1, "Invalid dropout")
        self.frames = frames
        self.upstream_identity = verify_upstream()
        base = resnet50(weights=None)  # Never make an implicit network request.
        self.initialization = {"kind": "random_for_engineering_only", "checkpoint": None}
        if checkpoint is not None:
            path = Path(checkpoint)
            need(checkpoint_sha256 is not None and checkpoint_size is not None, "Checkpoint identity required")
            need(path.is_file() and not path.is_symlink(), "Expected local regular ImageNet checkpoint")
            need(path.stat().st_size == checkpoint_size and sha256(path) == checkpoint_sha256,
                 "ImageNet checkpoint identity mismatch")
            state = torch.load(path, map_location="cpu", weights_only=True)
            # Load the complete 1000-class ImageNet ResNet BEFORE injection/head replacement.
            # Torchvision initializes legacy missing BN counters through its standard loader.
            base.load_state_dict(state, strict=True)
            self.initialization = {"kind": "ImageNet1K_V1_backbone_fresh_TAM_and_binary_head",
                                   "checkpoint": str(path.resolve()), "sha256": checkpoint_sha256,
                                   "source_tensor_count": len(state), "load_strict": True,
                                   "video_pretrained_checkpoint_loaded": False}
        else:
            need(checkpoint_sha256 is None and checkpoint_size is None, "Identity without checkpoint")
        # Suppress only verbose construction messages; the source/algorithm is unchanged.
        with redirect_stdout(io.StringIO()):
            make_temporal_modeling(base, n_segment=frames, t_kernel_size=3, t_stride=1, t_padding=1)
        feature_dim = base.fc.in_features
        base.fc = nn.Dropout(float(dropout)) if dropout else nn.Identity()
        self.backbone = base
        self.classifier = nn.Linear(feature_dim, 2)
        # Same new-head distribution used in upstream TSN._prepare_tsn.
        nn.init.normal_(self.classifier.weight, mean=0, std=0.001)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, clips):
        need(clips.ndim == 5 and clips.shape[0] > 0 and
             tuple(clips.shape[1:3]) == (3, self.frames) and min(clips.shape[-2:]) >= 2,
             "Expected nonempty [B,3,configured_T,H,W]; no guessing/interpolation")
        need(clips.dtype == torch.float32, "Expected ImageNet-normalized float32 RGB")
        n, _, t, h, w = clips.shape
        frames = clips.permute(0, 2, 1, 3, 4).reshape(n * t, 3, h, w)
        frame_logits = self.classifier(self.backbone(frames))
        # Average LOGITS, then external softmax/CE. Never average labels/probabilities here.
        return frame_logits.reshape(n, t, 2).mean(dim=1)
