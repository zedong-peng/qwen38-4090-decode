# Microbenchmarks (Ada small-T GEMM, research35 / research41)

Each file includes the engine's `ops/linear/ada_small_t_mma.cuh` (and `ops/common/norm_handoff.cuh`). Build them
with the engine's flags, with and without `-rdc=true`:

```
nvcc -O3 -DNDEBUG -std=c++20 "--generate-code=arch=compute_89,code=[compute_89,sm_89]" -lineinfo [-rdc=true] \
  -I$REPO/src -I$REPO/include -o bench bench_rdc.cu
```

- `bench_handoff.cu`: the research35 norm-handoff producer epilogue vs the plain residual epilogue. Variants: atomics,
  per-token lines, per-CTA partials, and `__launch_bounds__(256, 3)`.
- `bench_rdc.cu`: output projection and Q4 gate/up, cold and with an 8 MB warm prefix in L2.
- `bench_outsweep.cu`: output-projection pipeline depth and evict-first hint (bit-exact variants).
- `bench_gu4.cu`: Q4 gate/up at 3 vs 4 CTAs per SM.

Weights rotate over at least 320 MB, so every launch streams from DRAM. Time a repeat of the first configuration at
the end: the first kernel timed runs on a GPU that is still raising its clocks.
