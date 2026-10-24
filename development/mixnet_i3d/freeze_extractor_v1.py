import hashlib
from pathlib import Path

import timm
import torch

from development.mixnet_i3d.backbone import MixNetI3DBackbone
from development.mixnet_i3d.temporal_features import ThreeWindowFeatureExtractor


def main():
    folder = Path(__file__).resolve().parent
    output = folder / "artifacts/frozen_extractor_v1.pt"

    if output.exists():
        raise FileExistsError(f"Already exists: {output}")

    torch.manual_seed(20261023)

    old = timm.create_model("mixnet_s.ft_in1k", pretrained=True)
    mean = tuple(old.pretrained_cfg["mean"])
    std = tuple(old.pretrained_cfg["std"])

    extractor = ThreeWindowFeatureExtractor(
        MixNetI3DBackbone(old), mean=mean, std=std
    ).cpu().eval()

    # 記錄產生這份權重所用的實作版本。
    source_hashes = {}
    for name in (
        "inflation.py",
        "blocks.py",
        "backbone.py",
        "temporal_features.py",
    ):
        source_hashes[name] = hashlib.sha256(
            (folder / name).read_bytes()
        ).hexdigest()

    payload = {
        "schema_version": "frozen_extractor_v1",
        "model_name": "mixnet_s.ft_in1k",
        "timm_version": timm.__version__,
        "torch_version": str(torch.__version__),
        "mean": mean,
        "std": std,
        "source_sha256": source_hashes,
        "state_dict": extractor.state_dict(),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        torch.save(payload, handle)

    restored = torch.load(output, map_location="cpu", weights_only=True)
    for name, tensor in extractor.state_dict().items():
        assert torch.equal(tensor, restored["state_dict"][name])

    assert not extractor.backbone.training
    assert all(not p.requires_grad for p in extractor.parameters())

    print("weight round-trip: PASS")
    print("backbone frozen: True")
    print("Saved:", output)
    print("sha256:", hashlib.sha256(output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
