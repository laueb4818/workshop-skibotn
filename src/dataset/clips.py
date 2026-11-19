"""Video clips -> tensors for ViViT. Lives in src/dataset/clips.py.

Assumes the common Kinetics layout ``<root>/.../<class name>/<video>.mp4``: the class of a video
is the name of its parent folder. If the dataset is flat (all videos in one folder, labels in a
CSV), ``build_manifest`` raises with a message saying so; adapt it to your annotation file then.
"""

from __future__ import annotations

import random
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".webm", ".mov"}


# --------------------------------------------------------------------------- manifest


def build_manifest(root: str | Path) -> list[tuple[Path, str]]:
    """``(video path, class name)`` for every video under ``root``; class = parent folder name."""
    root = Path(root)
    items = [
        (p, p.parent.name)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ]
    if not items:
        raise FileNotFoundError(f"no video files ({sorted(VIDEO_EXTS)}) found under {root}")
    classes = {c for _, c in items}
    if len(classes) < 2:
        raise ValueError(
            f"all {len(items)} videos sit in one folder ({next(iter(classes))!r}), so the folder "
            "name cannot be the label. The labels are probably in an annotation CSV: "
            "adapt build_manifest to read it."
        )
    return items


def _norm(name: str) -> str:
    return name.lower().replace("_", " ").strip()


def make_label_map(classes, pretrained_label2id: dict[str, int]) -> tuple[dict[str, int], int]:
    """``({class name: label id}, num_labels)``.

    If every class is a Kinetics-400 class of the pretrained checkpoint, keep the checkpoint's own
    ids so its 400-way head stays valid. Otherwise number the classes 0..K-1 for a new head.
    """
    known = {_norm(k): v for k, v in pretrained_label2id.items()}
    classes = sorted(classes)
    if all(_norm(c) in known for c in classes):
        return {c: known[_norm(c)] for c in classes}, len(pretrained_label2id)
    return {c: i for i, c in enumerate(classes)}, len(classes)


def split_items(items, val_fraction: float, seed: int = 0):
    """Random train/val split by video."""
    items = list(items)
    random.Random(seed).shuffle(items)
    n_val = int(len(items) * val_fraction)
    return items[n_val:], items[:n_val]


# --------------------------------------------------------------------------- decoding


def _indices(total: int, num_frames: int, stride: int) -> np.ndarray:
    span = num_frames * stride
    start = max(0, (total - span) // 2)
    return np.clip(np.arange(start, start + span, stride), 0, total - 1)


def read_frames(path: str | Path, num_frames: int = 32, stride: int = 4) -> list[np.ndarray]:
    """Centered clip of ``num_frames`` RGB frames, every ``stride``-th frame."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        total = stream.frames
        if total > 0:  # decode only up to the last frame we need
            idx = _indices(total, num_frames, stride)
            wanted, last_needed = set(idx.tolist()), int(idx.max())
            got: dict[int, np.ndarray] = {}
            for i, frame in enumerate(container.decode(stream)):
                if i in wanted:
                    got[i] = frame.to_ndarray(format="rgb24")
                if i >= last_needed:
                    break
            if not got:
                raise ValueError(f"no frames decoded from {path}")
            fallback = got[max(got)]  # file shorter than it claims
            return [got.get(int(j), fallback) for j in idx]
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(stream)]
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    return [frames[i] for i in _indices(len(frames), num_frames, stride)]


def preprocess(frames: list[np.ndarray], size: int = 224, short_side: int = 256) -> torch.Tensor:
    """(T, 3, size, size) float tensor in [-1, 1]. Put your own preprocessing here.

    Resize so the short side is ``short_side`` (aspect ratio kept), center-crop to ``size``,
    scale to [-1, 1]: the same recipe as the checkpoint's image processor.
    """
    x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0  # (T, 3, H, W)
    h, w = x.shape[-2:]
    scale = short_side / min(h, w)
    x = F.interpolate(
        x, size=(round(h * scale), round(w * scale)), mode="bilinear", align_corners=False
    )
    h, w = x.shape[-2:]
    top, left = (h - size) // 2, (w - size) // 2
    x = x[:, :, top : top + size, left : left + size]
    return (x - 0.5) / 0.5


# --------------------------------------------------------------------------- dataset


class ClipDataset(Dataset):
    def __init__(self, items, label_of: dict[str, int]):
        self.items = list(items)
        self.label_of = label_of

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        # Kinetics contains some broken files: skip to the next one rather than kill the run.
        for step in range(5):
            path, cls = self.items[(i + step) % len(self.items)]
            try:
                return preprocess(read_frames(path)), self.label_of[cls]
            except Exception as err:  # noqa: BLE001
                print(f"[clips] skipping {path}: {err}")
        raise RuntimeError(f"5 unreadable videos in a row starting at index {i}")