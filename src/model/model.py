"""ViViT with an optional nofe tubelet embedding, adapted with LoRA.

Modes
-----
baseline  pretrained ViViT, LoRA on attention only (your reference run)
hybrid    pretrained Conv3d tubelets + a nofe branch, scaled by a learnable weight that starts at 0
full      Conv3d tubelets replaced entirely by the nofe module

Not run end to end yet: print the loaded model once and confirm the attribute path
``vivit.embeddings.patch_embeddings`` and the ``query`` / ``value`` module names for your
``transformers`` version.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from ..nofe import KernelNN, PointGraph
from ..nofe.search import radius_edges
from peft import LoraConfig, get_peft_model
from transformers import VivitForVideoClassification

CHECKPOINT = "google/vivit-b-16x2-kinetics400"
MODES = ("baseline", "hybrid", "full")


# --------------------------------------------------------------------------- graph


class _GridGraph(PointGraph):
    """Radius graph on a regular grid; only the edges are used, the features are dummies."""

    def __init__(self, radius: float, **kwargs):
        super().__init__(**kwargs)
        self.radius = radius

    def edge_parameters(self) -> tuple[float, int]:
        return self.radius, 0

    def neighbour_pairs(self, points, radius, k):
        return radius_edges(points, radius)


def grid_graph(frames: int, size: int, tubelet: tuple[int, int, int], radius: float = 1.01):
    """``(edge_index, edge_attr)`` for the (t, y, x) tubelet grid.

    Unit grid spacing and radius 1.01 give each tubelet its 6 face neighbours. The grid is the
    same for every clip, so this is built once. Node order is (t, y, x) with x fastest, which is
    the token order of ViViT's Conv3d embedding and therefore of its position embeddings.
    """
    nt, nh, nw = frames // tubelet[0], size // tubelet[1], size // tubelet[2]
    grid = np.stack(
        np.meshgrid(np.arange(nt), np.arange(nh), np.arange(nw), indexing="ij"), axis=-1
    ).reshape(-1, 3)
    dummy = np.random.default_rng(0).normal(size=(len(grid), 1))
    data = _GridGraph(radius, normalize=False).build(dummy.astype(np.float32), grid)
    return data.edge_index, data.edge_attr


# --------------------------------------------------------------------------- modules


class NofeTubelets(nn.Module):
    """Raw tubelets -> KernelNN -> tokens of shape (B, num_tokens, out_dim)."""

    def __init__(self, edge_index, edge_attr, tubelet, out_dim, width=8, ker_width=32, channels=3):
        super().__init__()
        self.tubelet = tuple(tubelet)
        self.register_buffer("edge_index", edge_index)  # [2, E]
        self.register_buffer("edge_attr", edge_attr)  # [E, 7]
        self.net = KernelNN(
            out_dim,
            in_channels=channels * math.prod(self.tubelet),
            edge_channels=edge_attr.shape[1],
            width=width,
            ker_width=ker_width,
        )

    def forward(self, pixel_values, **kwargs):  # (B, T, C, H, W)
        B, T, C, H, W = pixel_values.shape
        pt, ph, pw = self.tubelet
        x = pixel_values.reshape(B, T // pt, pt, C, H // ph, ph, W // pw, pw)
        x = x.permute(0, 1, 4, 6, 2, 3, 5, 7).reshape(B, -1, pt * C * ph * pw)
        N, E = x.shape[1], self.edge_index.shape[1]
        offsets = (torch.arange(B, device=x.device) * N).repeat_interleave(E)
        edge_index = self.edge_index.repeat(1, B) + offsets
        z = self.net(x.reshape(B * N, -1), edge_index, self.edge_attr.repeat(B, 1))
        return z.view(B, N, -1)


class Hybrid(nn.Module):
    """Original tubelet embedding + alpha * nofe branch; alpha starts at 0 (= plain pretrained)."""

    def __init__(self, orig: nn.Module, branch: nn.Module):
        super().__init__()
        self.orig = orig
        self.nofe = branch
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, pixel_values, **kwargs):
        return self.orig(pixel_values, **kwargs) + self.alpha * self.nofe(pixel_values)


# --------------------------------------------------------------------------- build


import re

def is_nofe_param(name: str) -> bool:
    return re.search(r"patch_embeddings\.(nofe\.|net\.|alpha)", name) is not None


def build_model(
    num_labels: int,
    mode: str = "hybrid",
    *,
    checkpoint: str = CHECKPOINT,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    width: int = 8,
    ker_width: int = 32,
):
    """Pretrained ViViT in the chosen mode, wrapped with LoRA. Returns a PEFT model."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    model = VivitForVideoClassification.from_pretrained(
        checkpoint, num_labels=num_labels, ignore_mismatched_sizes=True
    )

    if mode != "baseline":
        cfg = model.config
        edge_index, edge_attr = grid_graph(cfg.num_frames, cfg.image_size, tuple(cfg.tubelet_size))
        branch = NofeTubelets(
            edge_index,
            edge_attr,
            tubelet=cfg.tubelet_size,
            out_dim=cfg.hidden_size,
            width=width,
            ker_width=ker_width,
            channels=cfg.num_channels,
        )
        emb = model.vivit.embeddings
        emb.patch_embeddings = (
            Hybrid(emb.patch_embeddings, branch) if mode == "hybrid" else branch
        )

    lora = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj"],
        modules_to_save=["classifier"],
    )
    model = get_peft_model(model, lora)

    # LoRA freezes everything else; the new tubelet parameters are trained in full.
    for name, param in model.named_parameters():
        if is_nofe_param(name):
            param.requires_grad = True
    return model


# --------------------------------------------------------------------------- saving


def extra_state(model) -> dict[str, torch.Tensor]:
    """Weights of the new tubelet module. ``save_pretrained`` does not store these."""
    return {n: p.detach().cpu() for n, p in model.named_parameters() if is_nofe_param(n)}


def load_extra_state(model, state: dict[str, torch.Tensor]) -> None:
    missing = model.load_state_dict(state, strict=False)
    unexpected = list(missing.unexpected_keys)
    if unexpected:
        raise KeyError(f"unexpected keys in extra state: {unexpected[:5]}")