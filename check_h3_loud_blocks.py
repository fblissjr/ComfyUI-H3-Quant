#!/usr/bin/env python3
"""Which blocks of an H3 checkpoint have lopsided K-norm weights. No GPU, no ComfyUI.

    python check_h3_loud_blocks.py /path/to/minimax_h3_*.safetensors

Prints, per block, the share of `attn.k_norm.weight`'s energy in its four
loudest channels, sorted; the blocks at or above the node's default
threshold (0.15) are the ones the node folds. On every released H3 DiT
checkpoint measured, that is 45, 48 and 49 (about 30, 24 and 70 percent),
with every other block at 4 to 6 percent.
"""

import sys

import torch
from safetensors import safe_open


def main(path: str, threshold: float = 0.15) -> int:
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = [k for k in f.keys() if k.endswith("attn.k_norm.weight")]
        rows = []
        for k in keys:
            e = f.get_tensor(k).float().pow(2)
            share = (torch.topk(e, 4).values.sum() / e.sum()).item()
            top = torch.topk(e, 4).indices.tolist()
            rows.append((share, k, top))
    rows.sort()
    for share, k, top in rows:
        flag = "  <-- loud" if share >= threshold else ""
        print(f"{share:5.0%}  {k}  top channels {top}{flag}")
    loud = [k for s, k, _ in rows if s >= threshold]
    print(f"\n{len(loud)} of {len(rows)} blocks at or above {threshold:.0%}: "
          + ", ".join(k.split('.')[1] if k.startswith('blocks.') else k for k in loud))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.15))
