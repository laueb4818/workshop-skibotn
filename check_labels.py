import difflib
from transformers import AutoConfig
from src.dataset.clips import build_manifest, _norm
from src.dataset.dataset_utils import get_dataset
from src.model.model import CHECKPOINT

cfg = AutoConfig.from_pretrained(CHECKPOINT)
known = {_norm(k): k for k in cfg.label2id}
classes = sorted({c for _, c in build_manifest(get_dataset("kinetics-train-5per"))})

bad = [c for c in classes if _norm(c) not in known]
print(f"{len(classes)} classes in the dataset, {len(bad)} not found in Kinetics-400")
print("first 10 dataset folder names:", classes[:10])
for c in bad:
    near = difflib.get_close_matches(_norm(c), list(known), n=2, cutoff=0.6)
    print(f"  {c!r:40} closest K400: {[known[n] for n in near]}")