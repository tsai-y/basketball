from pathlib import Path

import timm
import torch

from development.data_pipeline.pipeline import load_gate_config
from development.data_pipeline.multiwindow_v1 import BasketballThreeWindowDataset
from development.mixnet_i3d.backbone import MixNetI3DBackbone
from development.mixnet_i3d.temporal_features import ThreeWindowFeatureExtractor


def main():
    root = Path(__file__).resolve().parents[2]
    output = root / "development/mixnet_i3d/artifacts/pilot_two_v1.pt"

    # 避免重跑時覆寫先前結果。
    if output.exists():
        raise FileExistsError(f"Already exists: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check the yolo11 environment")

    torch.manual_seed(20261023)

    # 載入 ImageNet 權重，再膨脹成 3D。
    old = timm.create_model("mixnet_s.ft_in1k", pretrained=True)
    mean = tuple(old.pretrained_cfg["mean"])
    std = tuple(old.pretrained_cfg["std"])

    extractor = ThreeWindowFeatureExtractor(
        MixNetI3DBackbone(old), mean=mean, std=std
    ).cuda().eval()

    dataset = BasketballThreeWindowDataset(
        load_gate_config(),
        split="train",
        video_ids=[
            "2p0/2p0_v108_000437_x264.mp4",
            "2p1/2p1_v108_000340_x264.mp4",
        ],
    )

    records = []
    for index in range(len(dataset)):
        item = dataset[index]

        # [3,3,64,192,256] → 加入 batch 維度。
        clips = item["tensor"].unsqueeze(0).cuda()
        features = extractor(clips)[0].cpu().contiguous()

        assert tuple(features.shape) == (3, 8, 1536)
        assert torch.isfinite(features).all().item()
        assert not features.requires_grad

        records.append({
            "video_id": item["video_id"],
            "binary_label": item["binary_label"],
            "split": item["split"],
            "features": features,
            "windows": item["windows"],
            "frame_count": int(item["frame_count"]),
            "fps": float(item["fps"]),
            "candidate_overlap_frames": item["candidate_overlap_frames"],
            "quality_metadata_available": item["quality_metadata_available"],
        })

    payload = {
        "schema_version": "feature_cache_pilot_v1",
        "model_name": "mixnet_s.ft_in1k",
        "timm_version": timm.__version__,
        "torch_version": str(torch.__version__),
        "mean": mean,
        "std": std,
        "backbone_frozen": True,
        "window_policy": "three_window_coverage_v1",
        "quality_aware_selection_used": False,
        "quality_weighting_used": False,
        "samples": records,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        torch.save(payload, handle)

    # 讀回自己剛存的檔案，確認內容沒有改變。
    restored = torch.load(output, map_location="cpu", weights_only=True)
    assert len(restored["samples"]) == len(records)

    for original, loaded in zip(records, restored["samples"]):
        for key in ("video_id", "binary_label", "split", "windows"):
            assert original[key] == loaded[key]
        assert torch.equal(original["features"], loaded["features"])

        print(
            loaded["video_id"],
            "label:", loaded["binary_label"],
            "shape:", tuple(loaded["features"].shape),
            "round-trip: PASS",
        )

    print("Saved:", output)


if __name__ == "__main__":
    main()
