"""Visualise ViViT tubelet tokens (PCA colours, k-means clusters, SLIC) for baseline/hybrid/full."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from skimage.segmentation import mark_boundaries, slic

from ..dataset.clips import preprocess, read_frames
from ..model.model import Hybrid, NofeTubelets, build_model, load_extra_state

GRID = (16, 14, 14)  # (t, y, x) tokens for 32 frames, 224 px, 2x16x16 tubelets


# ------------------------------------------------------------------ loading


def load_clip(path) -> torch.Tensor:
    """(1, 32, 3, 224, 224) in [-1, 1]."""
    return preprocess(read_frames(path))[None]


def load_model(mode: str, run_dir, device="cpu"):
    """Rebuild a trained run: LoRA adapter + classifier, plus the nofe weights from extra.pt."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    run_dir = Path(run_dir)
    adapter = run_dir / "adapter_model.safetensors"
    state = load_file(str(adapter))
    head = next(v for k, v in state.items() if "classifier" in k and k.endswith("weight"))
    model = build_model(head.shape[0], mode)
    set_peft_model_state_dict(model, state)
    extra = torch.load(run_dir / "extra.pt", map_location="cpu")["state"]
    if extra:
        load_extra_state(model, extra)
    return model.to(device).eval()


def _pe(model):
    return model.base_model.model.vivit.embeddings.patch_embeddings


def get_alpha(model):
    pe = _pe(model)
    return pe.alpha.item() if isinstance(pe, Hybrid) else None


@torch.no_grad()
def get_tokens(model, x, last_hidden: bool = False) -> dict[str, torch.Tensor]:
    """Token tensors of shape (3136, D) for one clip x of shape (1, 32, 3, 224, 224).

    Keys: "embed" (what enters the transformer), plus "orig" and "nofe" (alpha-scaled) in hybrid,
    and "last" (final hidden state) if last_hidden=True.
    """
    x = x.to(next(model.parameters()).device)
    pe = _pe(model)
    out = {}
    if isinstance(pe, Hybrid):
        out["orig"] = pe.orig(x)[0].cpu()
        out["nofe"] = (pe.alpha * pe.nofe(x))[0].cpu()
        out["embed"] = out["orig"] + out["nofe"]
    else:  # baseline (Conv3d) or full (nofe only)
        out["embed"] = pe(x)[0].cpu()
    if last_hidden:
        h = model(pixel_values=x, output_hidden_states=True).hidden_states[-1][0]
        out["last"] = h[-GRID[0] * GRID[1] * GRID[2]:].cpu()  # drop CLS if present
    return out


def norm_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean token norm of a divided by that of b."""
    return (a.norm(dim=-1).mean() / b.norm(dim=-1).mean()).item()


# ------------------------------------------------------------------ clustering


def cluster_tokens(tokens: torch.Tensor, k: int = 6, n_pca: int = 8, seed: int = 0):
    """One PCA and one k-means over the whole clip, so colours/ids are comparable across time."""
    flat = tokens.float().numpy()
    pca = PCA(n_components=n_pca, random_state=seed).fit(flat)
    low = pca.transform(flat)
    rgb = low[:, :3]
    lo, hi = np.percentile(rgb, 1, axis=0), np.percentile(rgb, 99, axis=0)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)
    km = KMeans(k, n_init=10, random_state=seed).fit(low)
    return {
        "km": km,
        "low": torch.from_numpy(low).reshape(*GRID, n_pca).float(),
        "rgb": torch.from_numpy(rgb).reshape(*GRID, 3).float(),
        "labels": km.labels_.reshape(GRID),
    }


def _up(grid: torch.Tensor, size: int = 224) -> np.ndarray:
    """(14, 14, D) -> (size, size, D), bilinear."""
    x = grid.permute(2, 0, 1)[None]
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)[0].permute(1, 2, 0).numpy()


def _up_labels(lab: np.ndarray, size: int = 224) -> np.ndarray:
    r = size // lab.shape[0]
    return np.repeat(np.repeat(lab, r, 0), r, 1)


# ------------------------------------------------------------------ plotting


def visualize(tokens, frames, ts=(0, 5, 10, 15), k=6, n_segments=150, title="", save=None, seed=0):
    """Rows: frame, PCA colours, k-means boundaries, k-means overlay, SLIC x token clusters.

    tokens: (3136, D); frames: (32, 3, 224, 224) in [-1, 1]. Token time step t covers frames 2t, 2t+1.
    """
    c = cluster_tokens(tokens, k=k, seed=seed)
    cmap = plt.get_cmap("tab20")
    rows = ["frame", "PCA", "k-means edges", "k-means overlay", "SLIC x tokens"]
    fig, ax = plt.subplots(len(rows), len(ts), figsize=(3 * len(ts), 3 * len(rows)), squeeze=False)
    for j, t in enumerate(ts):
        img = (frames[2 * t].permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).numpy()
        lab = _up_labels(c["labels"][t])
        seg = slic(img, n_segments=n_segments, compactness=10, start_label=0)
        low_up = _up(c["low"][t]).reshape(-1, c["low"].shape[-1])
        flat_seg = seg.ravel()
        counts = np.bincount(flat_seg)
        means = np.stack([np.bincount(flat_seg, weights=low_up[:, d]) / counts for d in range(low_up.shape[1])], 1)
        sp = c["km"].predict(means)[seg]
        views = [
            img,
            _up(c["rgb"][t]),
            mark_boundaries(img, lab, color=(1, 1, 0)),
            cmap(lab % 20)[..., :3] * 0.6 + img * 0.4,
            cmap(sp % 20)[..., :3] * 0.6 + img * 0.4,
        ]
        for i, v in enumerate(views):
            ax[i, j].imshow(np.clip(v, 0, 1))
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        ax[0, j].set_title(f"frame {2 * t}")
    for i, name in enumerate(rows):
        ax[i, 0].set_ylabel(name)
    fig.suptitle(title)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    plt.show()
    return fig