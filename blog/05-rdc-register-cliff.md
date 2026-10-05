# One CMake property, half a percent: relocatable device code on an inference engine

*Draft 5 of the Qwen3.8-27B single-RTX-4090 speed study. Local draft, not published.*

This is a short story about an optimization that lost, and the reason it lost. That reason turned out to be worth more
than the optimization.

## The optimization that lost

Between two weight GEMMs in each transformer block sits an RMSNorm: the attention output projection writes the
residual stream, the norm rescales it, and the gate/up projection reads the result. On the 4090 the norm is a
2.3 µs kernel plus a launch gap, 80 times per verify round. Folding it into its neighbours looks easy:

- The **producer**, the output projection's residual epilogue, also writes `x · (1 + w)` in BF16 and the per-token
  sum of squares. Each CTA sums its rows in a fixed order and adds the result into fixed-point slots with integer
  atomics, so the total is deterministic.
- The **consumer**, the gate/up GEMM, reads `x · (1 + w)` and multiplies every dot product by
  `rsqrt(sum / d + eps)` in its epilogue.

It is not bit-exact, because rounding happens before the scale instead of after. It was still cheap enough to
try. End to end it was **1.5% slower**. nsys showed the producer going from 20.0 µs to 25.5 µs per call. My first
guess was contention: 320 CTAs × 16 atomics land in one 128-byte line at the end of the kernel.

## The real reason

A microbenchmark of the producer, built with the same compiler flags as the engine, settled it:

| epilogue | µs per call |
|---|---|
| plain residual add | 19.5 |
| fold, with atomics | 24.7 |
| fold, atomics on separate lines | 24.6 |
| fold, no atomics (per-CTA partial sums) | 24.4 |
| fold, with atomics, `__launch_bounds__(256, 3)` | 19.8 |

The atomics cost nothing. Registers did. The fold epilogue pushed the kernel from 80 to 86 registers. At 256
threads per CTA that is the difference between three CTAs per SM and two:

```
88 regs (allocation unit 8) × 256 threads = 22,528 > 65,536 / 3
```

The output projection launches 320 CTAs on 128 SMs. At three per SM they all run in one wave. At two per SM they
take two.

## Where the 80 came from

The same epilogue in my microbenchmark used 73 registers, not 86. The engine's build differed in one flag. The
project's CMake sets `CUDA_SEPARABLE_COMPILATION ON` on every CUDA library, so every `.cu` is compiled with
`-rdc=true`: relocatable device code, linked by `nvlink` at the end. Nothing in this engine calls a device function
across translation units, so the flag buys nothing. Yet it changes code generation for every kernel. Same source,
same nvcc 12.8, `cuobjdump -res-usage`:

| kernel | `-rdc=true` | whole program |
|---|---|---|
| output projection (Q4, K = 6144) | 80 | 72 |
| the fold variant of it | 86 | 73 |
| Q4 gate/up (K = 5120, 34,816 rows) | 84 | 72 |
| 3-bit LM-head screen | 123 | 94 |
| GDN chunked record | 96 | 83 |
| RMSNorm | 56 | 39 |

Across the binary, 541 of the 2,801 kernels I could match by name use fewer registers whole-program, 336 use more,
and 1,924 are unchanged.

Most of these changes don't matter, because most kernels sit nowhere near an occupancy boundary. One that does is
the gate/up GEMM, the single biggest kernel in a round (32 calls, 3.1 ms). At 84 registers it ran two CTAs per SM.
At 72 it runs three, and its nsys median drops from 100.6 to 98.4 µs. It streams 100 MB in that time.

## Is three always better than two?

No. Forcing four CTAs per SM (`__launch_bounds__(256, 4)`, 64 registers, no spills) makes the same kernel 4 µs
slower: more CTAs mean more activation reloads and more L2 contention for the same DRAM stream. Occupancy is a
parameter to measure, not to maximize.

## End to end

A separate build directory with `-DNINFER_RDC=OFF` (a new CMake option; the default stays ON) links fine.
- Unforced Spec-Bench-480 outputs are identical on all 482 requests.
- Forced-text A B A B gives ms/round −0.43% and −0.24%, against same-config drift of +0.34% and +0.53%. The
  GPU warmed from 44 to 63 °C over the four runs. Corrected for linear drift: about **−0.5%**.

## Two smaller lessons

- **Time a repeat of the first configuration at the end of every microbenchmark.** My first rdc-vs-whole-program
  bench said the whole-program output projection was 1.8 µs slower. It was simply the first kernel timed, on a GPU
  still raising its clocks. The same kernel re-timed at the end matched the rdc build exactly.
- **Negative results deserve a root cause.** "Atomics are slow" was a plausible story that fit the profile. With
  that story the fold stays closed, and the 0.5% hiding in a build flag stays hidden.

The fold itself is back on the test bench on the whole-program build, where its producer has 73 registers. It is
still not bit-exact, so it also has to pass the decode-path fidelity check.
