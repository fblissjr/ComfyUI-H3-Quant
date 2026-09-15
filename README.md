# ComfyUI-H3-Quant

Findings about low-precision attention on MiniMax H3 in ComfyUI, and one
node that acts on them. Everything here was measured on one machine (an
RTX 4090, one set of captured activations from one render, three scenes
watched by a handful of people on labelled clips). None of it is
established beyond that. The reason to publish is so other people with
other cards, other checkpoints and other prompts can see whether they get
the same thing.

## The finding, in plain terms

Long H3 renders run attention in INT8, because bf16 attention over a
hundred thousand tokens is too slow: SageAttention on the dense steps for
some people, the Sol / Block Sparse Attention kernel on the routed steps
for most. Both round q and k to eight bits with one scale per row across
all 128 channels of a head.

H3's weights make that expensive on three blocks. The `k_norm.weight` of
blocks 45, 48 and 49 puts most of its energy into four channels (about 30,
24 and 70 percent respectively, against 4 to 6 percent on every other
block), and it does so on every released checkpoint we could find. On
those blocks the four loud channels set the row's scale and the other 124
channels are left with a couple of levels. Block 49 is also the block that
reads the prompt most sharply, so on our clips the rounding showed up as
prompt-adherence failures: a porter and his crate morphing on a turn, a
door sign doubled and misspelled, a chef appearing from nothing.

Two things anyone can check without a GPU. The weights fact:

    python check_h3_loud_blocks.py /path/to/minimax_h3_*.safetensors

prints every block's top-4 K-norm energy share; if yours shows the same
three blocks, the premise holds on your checkpoint too. And the
mechanism is arithmetic, not a model claim: one scale per row starves
quiet channels whenever a few channels are loud.

## What we measured, and against what

"Error" here is never the video. It is: the same captured q, k, v run
through the INT8 kernel and through exact fp32 attention, the outputs
subtracted, divided by the size of the exact output. The captures are the
real tensors the kernel received during a render, saved per block and step.
On the block-49 cell, first 8 heads, relative L2:

| kernel | plain | with the weights fold (this node) |
|---|---|---|
| comfy-kitchen `sol_attn` (Block Sparse Attention) | 0.027 | 0.023 |
| SageAttention 2 fp8++ | 0.049 | 0.045 |
| comfy-kitchen `int8_attention` (Model Attention Backend) | 0.017 | 0.016 |

The third row is the important caveat: kitchen's dense INT8 kernel
rotates q and k with a Hadamard before rounding, which spreads the loud
channels, so it does not have this problem, and the node does nothing
useful for it. If your graph runs that backend on the dense steps and no
Sol, this repo has nothing for you. Plain pytorch attention never rounds
and is unaffected too.

Per-head factors applied inside the kernels (in our comfy-kitchen and
SageAttention forks) recover about twice what the weights fold does, and a
Hadamard rotation inside Sol's quantizer about four times; those are on
their way upstream as pull requests and are not in this repo. A route-level
grade (how far Sol's INT8 routing decision sits from the fp32 decision)
moves the same way. Costs: the fold is free at render time; the in-kernel
factor is one to two percent of a Sol call; the rotation about fifteen.

What we saw in clips, for what it is worth: on three scenes, every viewer
ranked the unrebalanced render below the rebalanced one, unprompted, on
morphs and text. Labelled originals, known order, a few people. Not a
blind study.

## What ships here

`MiniMax H3 Channel Balance`, one node. For any per-channel factor f,
`q . k == (q * f) . (k / f)`, so the node computes f from the checkpoint's
own norm weights, folds `k_norm.weight / f` and `q_norm.weight * f` into
the model through ComfyUI's model patcher when it loads, on the blocks
whose K-norm is lopsided, and stops. Every attention score is unchanged in
exact arithmetic (the factor is equal within each RoPE pair so it commutes
with the rotation); only the INT8 rounding changes. No custom kernels, no
forks, zero render-time cost, off by default.

## Try it, and tell us what you see

1. Put this folder in `custom_nodes/`.
2. Run the checkpoint scan above on your model.
3. Add "MiniMax H3 Channel Balance" between the model loader and your
   attention nodes. Render a prompt with `balance` off, then the same
   prompt and seed with "loud blocks (from weights)". Watch both.

Same seed is not the same take once the numerics change, so do not look
for matching frames; look at whether objects stay one thing, whether text
stays text, whether people keep their faces. If you see nothing, that is a
result too. An issue with your card, kernel chain (Sage, Block Sparse,
Model Attention Backend, plain), checkpoint, and what changed or did not
is exactly what this repo is for.

## What is uncertain

- One machine, one capture set (one render, five blocks, five steps), one
  checkpoint family. The weights fact is checked across fourteen files;
  everything on activations is from that one set.
- Viewers were not blind and there were few of them.
- fp8 attention was not measured; its per-element exponent should make it
  immune to this specific effect, with a different, uniform error instead.
- The fold is not the ceiling. bf16 attention on the three blocks was
  still a little better than any rebalance in our clips, at a few percent
  more render time. The rotation inside the kernel is the closest we have
  come.
- Nothing here says anything about the quantized linears (int8 convrot);
  the loud channels are created after the projection, by the norm gain.

## Where the rest is

The measurements, the scripts that produce them, the viewers' words and
the kernel work live in the ComfyUI-h3-explorations repo
(`docs/h3_block49_quant_error.md` is the page to start from;
`docs/research/smoothquant_for_attention_qk.md` says what is and is not
new about this, which is less than it sounds: it is SmoothQuant's
migration pointed at attention instead of a linear layer). The in-kernel
forms are branches of the comfy-kitchen and SageAttention forks named there.
