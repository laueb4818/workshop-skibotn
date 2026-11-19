"""Fine-tune ViViT (LoRA, optional nofe tubelets) on a Kaggle video dataset.

Run from the repository root, so the ``src`` package is importable:

    python -m src.training.training --mode baseline
    python -m src.training.training --mode hybrid --max-videos 200 --epochs 1   # quick check

Modes: baseline | hybrid | full   (see src/model/model.py)
Needs: pip install kagglehub av peft transformers, plus the nofe package.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig

from ..dataset.clips import ClipDataset, build_manifest, make_label_map, split_items
from ..dataset.dataset_utils import get_dataset
from ..model.model import CHECKPOINT, MODES, build_model, extra_state, is_nofe_param


def make_optimizer(model, lr: float, lr_nofe: float, lr_head: float, weight_decay: float):
    """LoRA at ``lr``; the new classifier at ``lr_head``; the new tubelet module at ``lr_nofe``."""
    nofe, head, rest = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if is_nofe_param(name):
            nofe.append(param)
        elif "classifier" in name:
            head.append(param)
        else:
            rest.append(param)
    groups = [{"params": rest, "lr": lr}, {"params": head, "lr": lr_head}]
    if nofe:
        groups.append({"params": nofe, "lr": lr_nofe})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        logits = model(pixel_values=x.to(device)).logits
        correct += (logits.argmax(-1).cpu() == y).sum().item()
        total += len(y)
    return correct / max(total, 1)


def train(model, train_loader, val_loader, device, *, epochs, lr, lr_nofe, lr_head, weight_decay, amp):
    optimizer = make_optimizer(model, lr, lr_nofe, lr_head, weight_decay)
    use_amp = amp and device.type == "cuda"
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    best, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                loss = model(pixel_values=x, labels=y).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            running += loss.item()
        acc = evaluate(model, val_loader, device) if val_loader else float("nan")
        print(f"epoch {epoch:3d}  loss {running / len(train_loader):.4f}  val_acc {acc:.4f}")
        if val_loader and acc > best:
            best = acc
            params = dict(model.named_parameters())
            best_state = {n: params[n].detach().cpu().clone() for n in trainable}
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)  # restore the best epoch's trainable weights
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=MODES, default="hybrid")
    ap.add_argument("--dataset", default="kinetics-train-5per", help="REGISTRY name or owner/dataset")
    ap.add_argument("--data-dir", help="use this folder instead of downloading")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--max-videos", type=int, help="random subset, for quick tests")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4, help="LoRA and classifier")
    ap.add_argument("--lr-nofe", type=float, default=1e-3, help="new nofe tubelet module")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--width", type=int, default=8, help="KernelNN width; memory grows with width^2")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true", help="bfloat16 autocast on CUDA")
    ap.add_argument("--out", default="runs/vivit_lora")
    ap.add_argument("--lr-head", type=float, default=1e-3, help="new classifier head")
    args = ap.parse_args()

    root = Path(args.data_dir) if args.data_dir else get_dataset(args.dataset)
    items = build_manifest(root)
    if args.max_videos and args.max_videos < len(items):
        items = random.Random(args.seed).sample(items, args.max_videos)

    cfg = AutoConfig.from_pretrained(CHECKPOINT)
    label_of, num_labels = make_label_map({c for _, c in items}, cfg.label2id)
    print(f"{len(items)} videos, {len(label_of)} classes, head with {num_labels} outputs "
          f"({'pretrained Kinetics-400 labels' if num_labels == cfg.num_labels else 'new head'})")
    train_items, val_items = split_items(items, args.val_fraction, args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_labels, args.mode, lora_r=args.lora_r, width=args.width).to(device)
    model.print_trainable_parameters()

    train_loader = DataLoader(ClipDataset(train_items, label_of), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers)
    val_loader = (DataLoader(ClipDataset(val_items, label_of), batch_size=args.batch_size,
                             num_workers=args.workers) if val_items else None)

    train(model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr,
        lr_nofe=args.lr_nofe, lr_head=args.lr_head, weight_decay=args.weight_decay, amp=args.amp)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)  # LoRA adapter + classifier
    torch.save({"mode": args.mode, "state": extra_state(model)}, out / "extra.pt")  # nofe tubelets
    print(f"saved to {out}")


if __name__ == "__main__":
    main()