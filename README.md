# Better INT8 attention for MiniMax H3 in ComfyUI

MiniMax H3 at long lengths runs on INT8 attention, because bf16 attention
over a hundred thousand tokens is too slow: SageAttention on the dense
steps, the Sol / block-sparse kernel on the routed ones. Both round q and k
to eight bits with one scale per row, and that rounding is not free where
the model's weights make some channels loud. This repo is where the work
on making that rounding cheaper lives: what is measured, what is fixed,
what ships.

**What ships today: one node.** `MiniMax H3 Channel Balance` folds a
per-channel q/k rebalancing into the norm weights of the blocks whose
K-norm is lopsided (45, 48 and 49 on every released checkpoint), so
unrotated INT8 attention loses less there. Exact for every attention
score, applied through ComfyUI's own model patcher when the model loads,
zero cost at render time, off by default. No custom kernels, no forks,
works on whatever INT8 attention the graph already runs.

**What is behind it, and where the rest is going.** Two in-kernel forms of
the same idea (a per-head factor inside the quantizer, and a fixed
Hadamard rotation of q/k before it) live in the kitchen and sage forks and
do about twice to four times what the node does; they reach users as
comfy-kitchen PRs. Sol's routing decides which blocks are exact from INT8
centroid scores, so the same rounding moves the route; that is the next
thing on the list. The measurements (captured activations, per kernel,
against fp32 attention on the same inputs) are what every claim here rests
on, and the scripts to redo them are linked at the bottom.

**Who this is for.** Anyone whose H3 graph routes attention through
SageAttention (a Sage node, or the global flag) or through the Block
Sparse Attention node. If your graph runs plain pytorch attention, or the
Model Attention Backend node on "comfy kitchen attention" for every step,
the node does nothing useful: those paths do not have the problem, because
the kitchen dense kernel already rotates q/k before rounding.

## The problem, in four sentences

H3's last transformer blocks (45, 48 and 49 on every released checkpoint)
have a `k_norm.weight` that puts most of K's energy into four of 128
channels. INT8 attention kernels that quantize K unrotated use one scale
per key row across all 128 channels, so on those blocks the four loud
channels set the scale and the other 124 keep a couple of bits. Block 49 is
also the block that reads the prompt most sharply, so the rounding shows up
as prompt-adherence failures: objects that morph, text that garbles. On
captured activations the INT8 error at block 49 is about ten times block
0's on those kernels.

## What the node does

For any per-channel factor f, `q . k == (q * f) . (k / f)`. The node
computes f from the checkpoint's own norm weights, folds `k_norm.weight /
f` and `q_norm.weight * f` into the model as weight patches on the blocks
whose K-norm is lopsided, and stops there. Every attention score is
unchanged in exact arithmetic (the factor is made equal within each RoPE
pair so it commutes with the rotation); only the INT8 rounding moves.

Measured on captured block-49 activations, first 8 heads, relative L2
against fp32 attention on the same inputs: Sol's INT8 quantization term
0.0265 to 0.0231 (its total error, routing included, moves less), sage
fp8++ 0.0487 to 0.0453; block 0 unchanged. Per-head factors
inside the kernels do about twice as much (they live in the sage and
comfy-kitchen forks, see below); this node is the part that works on any
kernel without a build.

## Use

1. Put this folder in `custom_nodes/`.
2. Add "MiniMax H3 Channel Balance" between the model loader and your
   attention nodes, set `balance` to "loud blocks (from weights)".
3. Render the same prompt and seed with it off and on, and watch both. Same
   seed is not the same take once the numerics change, so judge the reading
   of the prompt and the consistency of objects, not frame matches.

`alpha` 0.5 and `loud_share` 0.15 are the measured values; "named blocks"
lets you name blocks by hand (`dense_blocks` syntax: `45,48,49`, negatives
count from the end).

## Check your own checkpoint first

    python check_h3_loud_blocks.py /path/to/minimax_h3_*.safetensors

No GPU, no ComfyUI. Prints every block's top-4 K-norm energy share; the
loud ones are what the node folds. If your checkpoint shows nothing above
the threshold, the node has nothing to do.

## What this is not

- Not a fix for the whole gap. bf16 attention on blocks 45/48/49 is still
  a little better than any rebalance, at a few percent more render time;
  the structural fix (rotating q/k before quantizing, as comfy-kitchen's
  `int8_attention` already does) belongs inside the kernels.
- Not measured blind. The renders behind this were judged on labelled
  clips by a handful of viewers on three scenes; every one ranked the
  balanced clip above the unbalanced one. That is why it exists and why it
  is off by default.

## Where the evidence lives

The ComfyUI-h3-explorations repo: `docs/h3_block49_quant_error.md` (the
mechanism, the measurements, what is not established),
`docs/research/smoothquant_for_attention_qk.md` (prior art: this is
SmoothQuant's migration pointed at the attention product),
`bench/results/2026-09-15_block49_*` (the clips' verdicts, wall times).
The in-kernel per-head versions: the sage fork's `qk_balance` (v0.7.19) and
the comfy-kitchen fork's `sol_attn(..., qk_balance=True)`.
