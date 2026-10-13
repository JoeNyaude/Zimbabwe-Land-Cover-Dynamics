"""Inference adapter for the separately trained Agric U-Net checkpoint."""
from functools import lru_cache
from pathlib import Path
import importlib.util

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGRIC_ROOT = PROJECT_ROOT / "Agric"
CHECKPOINT_PATH = AGRIC_ROOT / "checkpoints" / "best_unet.pt"
MEDIA_ROOT = PROJECT_ROOT / "core" / "media"
CLASS_NAMES = ["Urban", "Agriculture", "Rangeland", "Forest", "Water", "Barren", "Unknown"]
CLASS_COLORS = np.asarray([
    (0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 255, 0),
    (0, 0, 255), (255, 255, 255), (0, 0, 0),
], dtype=np.uint8)


@lru_cache(maxsize=1)
def _load_model():
    import torch

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Agric checkpoint not found: {CHECKPOINT_PATH}")
    spec = importlib.util.spec_from_file_location("zimlcd_agric_model", AGRIC_ROOT / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    args = checkpoint.get("args", {})
    image_size = int(args.get("image_size", 384))
    model = module.UNet(num_classes=len(CLASS_NAMES), base=int(args.get("base_channels", 32))).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return torch, model, device, image_size


def _tile_starts(length: int, tile_size: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(tile_size // 2, 1)
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def predict_image(source, output_dir: Path):
    """Segment overlapping model-sized windows at source resolution."""
    torch, model, device, image_size = _load_model()
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        original = image.convert("RGB")
        original_width, original_height = original.size
        image_array = np.asarray(original, dtype=np.float32) / 255.0
        padded_height = max(original_height, image_size)
        padded_width = max(original_width, image_size)
        array = np.pad(
            image_array,
            ((0, padded_height - original_height), (0, padded_width - original_width), (0, 0)),
            mode="edge",
        )
        mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
        std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
        array = (array - mean) / std
        y_starts = _tile_starts(padded_height, image_size)
        x_starts = _tile_starts(padded_width, image_size)
        tile_positions = [
            (top, left) for top in y_starts for left in x_starts
        ]
        logits_sum = None
        prediction_count = np.zeros((padded_height, padded_width), dtype=np.uint16)
        batch_size = 4

        with torch.inference_mode():
            for offset in range(0, len(tile_positions), batch_size):
                positions = tile_positions[offset:offset + batch_size]
                tiles = np.stack([
                    array[top:top + image_size, left:left + image_size].transpose(2, 0, 1)
                    for top, left in positions
                ])
                tensor = torch.from_numpy(tiles).to(device)
                logits = model(tensor).float().cpu().numpy()
                if logits_sum is None:
                    logits_sum = np.zeros(
                        (logits.shape[1], padded_height, padded_width), dtype=np.float32
                    )
                for (top, left), tile_logits in zip(positions, logits):
                    logits_sum[
                        :, top:top + image_size, left:left + image_size
                    ] += tile_logits
                    prediction_count[
                        top:top + image_size, left:left + image_size
                    ] += 1

        averaged_logits = logits_sum / prediction_count[None]
        prediction = averaged_logits.argmax(axis=0)[:original_height, :original_width].astype(np.uint8)
        mask = Image.fromarray(CLASS_COLORS[prediction])

    mask_path = output_dir / "prediction.png"
    mask.save(mask_path)
    counts = np.bincount(prediction.ravel(), minlength=len(CLASS_NAMES))
    total = max(int(counts.sum()), 1)
    composition = [{"name": name, "color": "#%02x%02x%02x" % tuple(CLASS_COLORS[i]),
                    "percentage": round(100 * int(counts[i]) / total, 2)}
                   for i, name in enumerate(CLASS_NAMES) if counts[i]]
    return mask_path, composition
