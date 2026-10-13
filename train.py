"""Train a U-Net CNN on the DeepGlobe-style agricultural land-cover dataset.

The official valid/ and test/ folders have satellite images only (no masks),
so labeled train/ images are split into train and validation sets.

Examples:
  python train.py --epochs 15 --batch-size 4
  python train.py --smoke-test
  python train.py --predict checkpoints/best_unet.pt --split valid
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from PIL import ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CLASS_NAMES, LandCoverDataset, class_mask_to_rgb, list_labeled_ids
from model import UNet


ROOT = Path(__file__).resolve().parent
NUM_CLASSES = len(CLASS_NAMES)
UNKNOWN_INDEX = 6


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_ids(ids: list[str], val_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    shuffled = ids.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]


def confusion_from_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    preds = logits.argmax(dim=1)
    valid = target != UNKNOWN_INDEX
    preds = preds[valid]
    target = target[valid]
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64, device=logits.device)
    if preds.numel() == 0:
        return cm
    idx = target * NUM_CLASSES + preds
    counts = torch.bincount(idx, minlength=NUM_CLASSES ** 2)
    return counts.reshape(NUM_CLASSES, NUM_CLASSES)


def estimate_class_weights(image_dir: Path, ids: list[str]) -> torch.Tensor:
    """Estimate pixel-frequency weights from training masks only."""
    counts = torch.zeros(NUM_CLASSES - 1, dtype=torch.float64)
    for image_id in ids:
        mask = np.asarray(Image.open(image_dir / f"{image_id}_mask.png").convert("RGB"))
        # The RGB palette is exact, so count each class without resizing masks.
        for class_id, color in enumerate(((0, 255, 255), (255, 255, 0), (255, 0, 255),
                                          (0, 255, 0), (0, 0, 255), (255, 255, 255))):
            counts[class_id] += int(np.all(mask == color, axis=-1).sum())

    frequencies = counts / counts.sum().clamp_min(1)
    # Inverse square-root weights are less extreme than inverse frequencies.
    weights = frequencies.clamp_min(1e-6).rsqrt()
    weights = weights / weights.mean()
    return weights.float()


class WeightedCEDiceLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor, ignore_index: int) -> None:
        super().__init__()
        # Cross-entropy requires one weight per output channel. Unknown is ignored.
        self.register_buffer("class_weights", torch.cat((class_weights, torch.ones(1))))
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(
            logits, target, weight=self.class_weights, ignore_index=self.ignore_index
        )

        valid = target != self.ignore_index
        safe_target = target.masked_fill(~valid, 0)
        probs = logits.softmax(dim=1)[:, :NUM_CLASSES - 1]
        one_hot = nn.functional.one_hot(safe_target, NUM_CLASSES).permute(0, 3, 1, 2)
        one_hot = one_hot[:, :NUM_CLASSES - 1].to(probs.dtype)
        valid_channel = valid.unsqueeze(1).to(probs.dtype)
        probs = probs * valid_channel
        one_hot = one_hot * valid_channel

        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dims)
        denominator = probs.sum(dims) + one_hot.sum(dims)
        present = one_hot.sum(dims) > 0
        dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
        dice_loss = (1 - dice[present]).mean() if present.any() else logits.sum() * 0
        return ce + dice_loss


def scores_from_confusion(cm: torch.Tensor) -> dict[str, float]:
    # Unknown target pixels are excluded. Unknown predictions still count as errors.
    cm = cm.cpu().float()
    n = UNKNOWN_INDEX
    tp = torch.diag(cm)[:n]
    fp = cm[:n, :n].sum(dim=0) - tp
    fn = cm[:n, :].sum(dim=1) - tp
    iou = tp / (tp + fp + fn + 1e-7)
    acc = tp.sum() / (cm[:n, :].sum() + 1e-7)
    return {
        "pixel_acc": acc.item(),
        "miou": iou.mean().item(),
        **{f"iou_{name}": iou[i].item() for i, name in enumerate(CLASS_NAMES[:-1])},
    }


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> dict[str, float]:
    model.train(train)
    total_loss = 0.0
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64, device=device)
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for images, masks, _ids in tqdm(loader, leave=False, desc="train" if train else "val"):
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            cm += confusion_from_logits(logits.detach(), masks)

    metrics = scores_from_confusion(cm)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics


def format_metrics(metrics: dict[str, float]) -> str:
    main = f"loss={metrics['loss']:.4f} acc={metrics['pixel_acc']:.3f} mIoU={metrics['miou']:.3f}"
    per_class = " ".join(f"{name}={metrics[f'iou_{name}']:.2f}" for name in CLASS_NAMES[:-1])
    return f"{main} | {per_class}"


def save_validation_preview(model, dataset, ids: list[str], device, output_path: Path, limit: int = 8) -> None:
    """Save input, truth, and prediction panels for a quick visual error check."""
    tile_size = dataset.image_size
    label_height = 24
    canvas = Image.new("RGB", (tile_size * 3, (tile_size + label_height) * min(limit, len(ids))), "white")
    draw = ImageDraw.Draw(canvas)
    model.eval()

    with torch.no_grad():
        for row, image_id in enumerate(ids[:limit]):
            image_tensor, target, _ = dataset[row]
            logits = model(image_tensor.unsqueeze(0).to(device))
            pred = logits.argmax(dim=1)[0].cpu().numpy()
            target_rgb = class_mask_to_rgb(target.numpy())
            pred_rgb = class_mask_to_rgb(pred)
            source = Image.open(dataset.image_dir / f"{image_id}_sat.jpg").convert("RGB")
            source = source.resize((tile_size, tile_size), Image.Resampling.BILINEAR)

            y = row * (tile_size + label_height)
            for col, (label, tile) in enumerate((
                ("Image", source), ("Ground truth", Image.fromarray(target_rgb)),
                ("Prediction", Image.fromarray(pred_rgb)),
            )):
                x = col * tile_size
                draw.text((x + 4, y + 4), f"{image_id} | {label}", fill="black")
                canvas.paste(tile.resize((tile_size, tile_size)), (x, y + label_height))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"Saved validation preview: {output_path}")


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    labeled_ids = list_labeled_ids(ROOT / "train")
    if args.smoke_test:
        labeled_ids = labeled_ids[:16]
        args.epochs = 1
        args.batch_size = min(args.batch_size, 2)
        print("Smoke test: 16 images, 1 epoch")

    train_ids, val_ids = split_ids(labeled_ids, args.val_ratio, args.seed)
    print(f"Train images: {len(train_ids)}  Val images: {len(val_ids)}")

    train_ds = LandCoverDataset(ROOT / "train", train_ids, args.image_size, train=True)
    val_ds = LandCoverDataset(ROOT / "train", val_ids, args.image_size, train=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = UNet(num_classes=NUM_CLASSES, base=args.base_channels).to(device)
    class_weights = estimate_class_weights(ROOT / "train", train_ids).to(device)
    print("Train pixel weights:", " ".join(
        f"{name}={weight:.2f}" for name, weight in zip(CLASS_NAMES[:-1], class_weights.tolist())
    ))
    criterion = WeightedCEDiceLoss(class_weights, ignore_index=UNKNOWN_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    best_miou = -1.0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_metrics["miou"])

        print(f"Epoch {epoch:02d}/{args.epochs}  train {format_metrics(train_metrics)}")
        print(f"            val   {format_metrics(val_metrics)}")

        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "val_miou": val_metrics["miou"],
            "args": vars(args),
        }
        torch.save(payload, ckpt_dir / "last_unet.pt")
        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            torch.save(payload, ckpt_dir / "best_unet.pt")
            print(f"            saved best_unet.pt  mIoU={best_miou:.3f}")

    print(f"Done. Best val mIoU: {best_miou:.3f}")
    best = torch.load(ckpt_dir / "best_unet.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    save_validation_preview(
        model, val_ds, val_ids, device, ROOT / "diagnostics" / "val_preview.png"
    )


def predict(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ckpt = torch.load(args.predict, map_location=device, weights_only=False)
    image_size = ckpt.get("args", {}).get("image_size", args.image_size)
    base = ckpt.get("args", {}).get("base_channels", args.base_channels)

    model = UNet(num_classes=NUM_CLASSES, base=base).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    split_dir = ROOT / args.split
    ids = sorted(p.name.replace("_sat.jpg", "") for p in split_dir.glob("*_sat.jpg"))
    loader = DataLoader(
        LandCoverDataset(split_dir, ids, image_size, train=False, labeled=False),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    out_dir = ROOT / "predictions" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for images, image_ids in tqdm(loader, desc=f"predict {args.split}"):
            logits = model(images.to(device))
            pred = logits.argmax(dim=1)[0].cpu().numpy()
            rgb = class_mask_to_rgb(pred)
            Image.fromarray(rgb).save(out_dir / f"{image_ids[0]}_pred.png")
    print(f"Wrote {len(ids)} masks to {out_dir}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train U-Net on land-cover satellite imagery")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1 short epoch to verify the pipeline")
    parser.add_argument("--predict", type=str, default="", help="Path to checkpoint for inference")
    parser.add_argument("--split", type=str, default="valid", choices=["valid", "test", "train"])
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.predict:
        predict(arguments)
    else:
        train(arguments)
