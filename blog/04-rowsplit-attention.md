# Splitting verify attention by rows, bit for bit

*Draft, 2026-10-06. Fourth post of the single-RTX-4090 Qwen3.8-27B series. Code:
`code/engine-patches/patch_rowsplit.py` and the kernel header `small_t_k8v4_rowsplit.cuh`.*

A speculative decoding round on this stack verifies a 16-node draft tree. Every full-attention layer (16 of the 64)
attends 16 new query tokens × 24 query heads against the cached keys. Before this change, one such layer cost about
22 µs, in five launches:

| kernel | µs |
|---|---|
| q/k RMSNorm + RoPE | 1.7 |
| query prepare (Hadamard rotation, FP8 row quantization) | 1.4 |
| wide split attention (FP8 K, INT8 V) | 14.5–16.9 |
| split reduce | 2.6–4.5 |
| sigmoid output gate | 1.4 |

That is 2% of a 17 ms round. The wide kernel's time barely moves with context: 12.2 / 13.9 / 14.5 µs at 4 / 9 / 33
splits. So the cost is not keys. It is a latency chain inside each CTA.

## Why the wide kernel is slow at short contexts

Spec-Bench decode contexts are short. Per generated token, the median context is 236 keys, p90 is 927 and p99 is 1,553.
55% of tokens see at most 256 keys.

The wide kernel gives one CTA all 96 query rows of a KV head (16 tokens × 6 grouped heads) and one 64-key split. At 236
keys that is 4 splits × 4 KV heads = **16 CTAs on a 128-SM GPU**, and each walks its two 32-key tiles serially through
FP8 QK MMAs, an online softmax and INT8-V PV MMAs. Phase probes (clock64 stamps) put a CTA's time in a fixed
metadata/page-table load (≈1 µs), the first tile's latency, and ≈1.3 µs of Tensor Core work per tile per row block.

## The row split

The fix is the classic one: more CTAs per split. A CTA owns one 16-row tile (or two) of one (KV head, split), and four
warps share it:
- warp *q* scores keys 8*q*..8*q*+7 of each 32-key tile, QK over all 256 dimensions;
- warp *q* then multiplies P by value dimensions 64*q*..64*q*+63, one INT8 scale group.

One split's work now spreads over 6 CTAs instead of 1, so a 236-key verify block runs on 96 CTAs.

The new K/V columns must be in the cache before any CTA reads it, because the six row tiles of a KV head all need them.
A small 16-block kernel appends them first. It also hoists the input loads ahead of the page lookups. Query
rotation and quantization move inside the attention kernel, where they overlap the first tile's copy, so the separate
query-prepare launch is gone.

## Keeping it bit-exact

The engine is tuned to a quality budget: every non-bit-exact change has to pass a fidelity check, and each one
re-randomizes near-ties. An attention kernel that reproduces the old split partials **bit for bit** costs nothing on
that budget and reuses the existing reduce. Every per-row operation and every reduction order has to match:

- **Row maxima** are exact under any order. Each warp's quad max over its 8 keys is published to shared memory and
  combined with the running max.
- **Row sums of P.** The wide kernel sums each lane's P values over its four 8-key score tiles in key order, then over
  the quad (`warp_sum<4>`), then `l = fma(l, alpha, sum)`. The row-split warps publish their per-lane partial
  (s0 + s1) per score tile, and every warp re-adds the four in key order. Same bits.
- **PV.** The same two chained FP16-accumulate m16n8k16 MMAs per 32 keys, then an FP32 add. A product does not depend
  on which MMA column a dimension sits in, so the dimension-to-column map can be chosen freely.
- **Online softmax rescale**, `alpha = exp2((m − m_new)·log2 e)`, uses the same approximate exp2 and the same `−inf`
  guards.

The freedom in the column map is what makes the INT8-V path cheap. MMA *j* of quarter *q* maps B column *n* to
dimension 64*q* + 8*n* + *j*. A lane's eight values of one key are then eight contiguous bytes: one LDS.64 and byte
permutes, with no FP16 staging of V. Storing key rows in the order *k* ⊕ ((*k* >> 1) & 1) spreads the four keys of a
load across bank halves.

The bench (`rs_bench.cu`, random caches, shuffled page tables, KV evicted from L2 before each call) compares the
caches, the split statistics, the split numerators and the reduced outputs. All are identical at every window from
40 to 32,768 keys, for 1, 2 and 3 row tiles per CTA. One trap: the reduce discards its partial lines from L2 after
reading them, so compare the partials before the reduce runs.

## Speed

Path time (append or prepare + attention + reduce), event medians:

| keys | wide | row split, 1 tile/CTA | 2 tiles/CTA |
|---|---|---|---|
| 100 | 19.5 µs | 14.3 | 15.4 |
| 236 | 22.5 | 14.3 | 16.4 |
| 512 | 22.5 | 17.4 | 19.5 |
| 927 | 23.6 | 23.6 | 23.8 |
| 2,048 | 29.7 | 34.8 | 34.8 |
| 8,192 | 57.3 | 69.6 | 57.3 |

At the median context the attention kernel itself drops from 16.9 to 7.3 µs (nsys per-call medians). Long contexts
are another matter: six CTAs re-read the same K/V, and the wide kernel's warp-specialized streaming wins.

## What still limits it

A row-split CTA is instruction-latency-bound. It runs about 1,000 instructions per 32-key tile per warp, a third of
them integer address arithmetic, with a single warp per scheduler. Warm and cold caches make no difference. Next, the
same arithmetic with eight warps per CTA: the two tiles of a split are scored concurrently, and PV is split into
32-dimension slices with a bank-rotated INT8 layout, so a lane's four values per key are a single 32-bit word.

## Eight warps: the row-pair kernel

The next version keeps the arithmetic and changes who does it. A CTA still owns one 16-row tile of one split, but it
runs eight warps:
- **Scores in pairs.** Warps 0–3 score key tile *t*, warps 4–7 score tile *t* + 1, each warp a quarter of the 32
  keys. Both tiles' row maxima go through shared memory. Tile *t* + 1's running max is taken after tile *t*'s, so
  the two-tile update is the same sequence of operations as before.
- **PV in 32-dimension slices.** Warp *w* multiplies both tiles' P by value dimensions 32*w* … 32*w* + 31. MMA *j*
  maps column *n* to dimension 32*w* + 4*n* + *j*, so a lane's four values of one key are a single 32-bit word.
  Key rows rotate their 32-byte chunks by (*k* >> 1) & 3, which makes those reads bank-conflict-free.
- **Separate pipelines for K and V.** They are separate cp.async groups. The next pair's keys are requested right
  after the scores, its values right after PV.

Same bench, same comparisons, bit-identical at every window. Path times (µs):

| keys | wide | row split, 1 tile | row pair |
|---|---|---|---|
| 236 | 22.5 | 14.3 | 14.3 |
| 512 | 21.6 | 17.4 | 16.4 |
| 927 | 22.6 | 23.6 | 21.5 |
| 1,553 | 26.6 | 29.7 | 26.8 |
| 2,048 | 29.7 | 34.8 | 31.7 |

## End to end, and the L2 fill it ate

The first end-to-end test of the row split read as noise, which was surprising for a −9 µs kernel. nsys explained it.
The wide kernel carried a passenger: grid rows past its splits that prefetch the first 8 MB of the next GEMM (the
attention output projection) into L2 while attention, which is latency-bound, leaves DRAM idle. The row-split
launcher dropped that request, and the output projections got 61 µs per round slower. That cancelled the 62 µs the
attention kernels saved.

The fix is to put the fill CTAs back as extra grid z-slices behind the attention slots. That restores the GEMM time
but costs most of the attention gain:
- 8 MB of fill is about 8 µs of DRAM, as long as the shorter kernel itself.
- 240 attention CTAs plus 48 fill CTAs exceed the 256 resident slots (two CTAs per SM), so a second wave of fill CTAs
  trails the attention work.

Per round (nsys, code prompt):

| | before | row pair + fill slices |
|---|---|---|
| attention class | 353 µs | 346 |
| GEMM class | ≈14,170 | 14,172 |

Shipped together with stage-tiled head screens (bit-identical as well; screens −48 µs per round), the forced-text
A B A B on Spec-Bench-480 gives **−0.72% and −0.58% ms/round**, against +0.23% / +0.37% drift between same-config
runs. Unforced outputs are identical on all 482 requests.

The lesson is about accounting rather than attention. A faster kernel is worth only the DRAM-idle time it gives
back, and a latency-bound kernel that already hides someone else's prefetch is not idle. The next step is sizing the
fill to the kernel it rides on (4–6 MB instead of 8) and fitting it into the spare slots.
