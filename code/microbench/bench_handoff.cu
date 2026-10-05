// research40 probe: the research35 norm-handoff producer (out projection, Q4 K = 6144, 5120 rows, T = 16) against
// the plain residual epilogue. research35's epilogue took the kernel from 80 to 86 registers (2 instead of 3 CTAs
// per SM, so its 320 CTAs need two waves). Variants: plain; research35 as built; research35 under
// __launch_bounds__(256, 3); per-token slots on separate 128-byte lines; atomic-free per-CTA partials.
// Weights rotate over >= 320 MB (cold, as in a decode round). Times are graph replays, best of 7.
// usage: bench_handoff [T=16]
#include "ops/linear/ada_small_t_mma.cuh"
#include "ops/common/norm_handoff.cuh"

#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace ninfer::ops::detail;

namespace ninfer {
void cuda_check(cudaError_t e, const char* expr, const char* file, int line) {
    if (e != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", file, line, expr, cudaGetErrorString(e)); exit(1); }
}
}  // namespace ninfer
#define CK(x) ninfer::cuda_check((x), #x, __FILE__, __LINE__)

constexpr int kRowsOut = 5120, K = 6144;
using S1 = Q4AdaSmallTSchedule<2, 8, 2, 8, 2>;
struct S3 : S1 { static constexpr int kMinBlocks = 3; };

struct PlainResidual {
    static constexpr bool kPairedHalves = false;
    __nv_bfloat16* out;
    int ld;
    __device__ __forceinline__ void operator()(int row, int token, float dot) const {
        __nv_bfloat16* p = out + static_cast<long>(token) * ld + row;
        *p = __float2bfloat16(dot + __bfloat162float(*p));
    }
};

// research35's producer with the token stride of the slots as a parameter (16 = one 128-byte line per token), and
// optionally plain per-CTA partial stores instead of the atomics.
template <bool Partials>
struct HandoffVariant {
    static constexpr bool kPairedHalves = false;
    static constexpr bool kRowsHook     = true;
    static constexpr int kMaxRows       = 40;
    __nv_bfloat16* out;
    std::int32_t out_ld;
    const __nv_bfloat16* norm;
    __nv_bfloat16* xs;
    std::int32_t xs_ld;
    unsigned long long* slot;
    std::int32_t stride;
    float* partial;
    __device__ static float (&squares())[kMaxRows][16] {
        __shared__ float s[kMaxRows][16];
        return s;
    }
    __device__ __forceinline__ void operator()(int, int, float) const {}
    __device__ __forceinline__ void element(int weight_row, int row, int token, float value) const {
        __nv_bfloat16* p        = out + static_cast<std::int64_t>(token) * out_ld + weight_row;
        const __nv_bfloat16 sum = __float2bfloat16(value + __bfloat162float(*p));
        *p                      = sum;
        const float x           = __bfloat162float(sum);
        xs[static_cast<std::int64_t>(token) * xs_ld + weight_row] =
            __float2bfloat16_rn(x * (1.0f + __bfloat162float(norm[weight_row])));
        squares()[row][token] = x * x;
    }
    __device__ __forceinline__ void rows_done(int block, int rows, int tokens, int tid, int) const {
        __syncthreads();
        if (tid < 128) {
            const int token = tid >> 3;
            const int j     = tid & 7;
            float s         = 0.0f;
            for (int r = j; r < rows; r += 8) { s += squares()[r][token]; }
            s += __shfl_xor_sync(0xffffffffu, s, 1);
            s += __shfl_xor_sync(0xffffffffu, s, 2);
            s += __shfl_xor_sync(0xffffffffu, s, 4);
            if (j == 0 && token < tokens) {
                if constexpr (Partials) {
                    partial[block * 16 + token] = s;
                } else {
                    atomicAdd(slot + token * stride, __float2ull_rn(s * 65536.0f));
                }
            }
        }
    }
};

static void fill(std::vector<uint8_t>& v, unsigned seed) {
    for (auto& b : v) { seed = seed * 1664525u + 1013904223u; b = seed >> 24; }
}

template <class Launch>
static float time_graph(Launch&& launch, int reps) {
    cudaStream_t s; CK(cudaStreamCreate(&s));
    launch(s, 0); CK(cudaStreamSynchronize(s));
    cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < reps; ++i) launch(s, i);
    CK(cudaStreamEndCapture(s, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
    CK(cudaGraphLaunch(ge, s)); CK(cudaStreamSynchronize(s));
    cudaEvent_t a, e; cudaEventCreate(&a); cudaEventCreate(&e);
    float best = 1e30f;
    for (int t = 0; t < 7; ++t) {
        cudaEventRecord(a, s); cudaGraphLaunch(ge, s); cudaEventRecord(e, s); cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms, a, e); best = std::min(best, ms / reps);
    }
    CK(cudaGetLastError());
    cudaGraphExecDestroy(ge); cudaGraphDestroy(g); cudaStreamDestroy(s);
    return best * 1000.0f;
}

int main(int argc, char** argv) {
    const int T = argc > 1 ? atoi(argv[1]) : 16;
    const long groups = (long)kRowsOut * (K / 64);
    std::vector<uint8_t> codes(groups * 32), scales(groups * 2);
    fill(codes, 1);
    for (long g = 0; g < groups; ++g) { uint16_t s = 0x2000 + (static_cast<uint32_t>(g * 2654435761u) >> 22); memcpy(&scales[g * 2], &s, 2); }
    const size_t wbytes = codes.size() + scales.size();
    const int copies = (int)std::max<long>(2, ((320L << 20) + wbytes - 1) / wbytes);
    std::vector<uint8_t*> dc(copies), ds(copies);
    for (int c = 0; c < copies; ++c) {
        CK(cudaMalloc(&dc[c], codes.size())); CK(cudaMemcpy(dc[c], codes.data(), codes.size(), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&ds[c], scales.size())); CK(cudaMemcpy(ds[c], scales.data(), scales.size(), cudaMemcpyHostToDevice));
    }
    std::vector<__nv_bfloat16> hx((long)T * K), hr((long)T * kRowsOut), hn(kRowsOut);
    unsigned seed = 7;
    auto rnd = [&] { seed = seed * 1664525u + 1013904223u; return ((seed >> 8) & 0xffff) / 65536.0f - 0.5f; };
    for (auto& v : hx) v = __float2bfloat16(0.05f * rnd());
    for (auto& v : hr) v = __float2bfloat16(2.0f * rnd());
    for (auto& v : hn) v = __float2bfloat16(0.2f * rnd());
    __nv_bfloat16 *x, *res, *res0, *norm, *xs;
    unsigned long long* slots; float* partial;
    CK(cudaMalloc(&x, hx.size() * 2)); CK(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&res, hr.size() * 2)); CK(cudaMalloc(&res0, hr.size() * 2));
    CK(cudaMemcpy(res0, hr.data(), hr.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&norm, hn.size() * 2)); CK(cudaMemcpy(norm, hn.data(), hn.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&xs, hr.size() * 2));
    CK(cudaMalloc(&slots, 16 * 16 * 8)); CK(cudaMalloc(&partial, 320 * 16 * 4));

    const HandoffVariant<false> packed{res, kRowsOut, norm, xs, kRowsOut, slots, 1, partial};
    const HandoffVariant<false> lines{res, kRowsOut, norm, xs, kRowsOut, slots, 16, partial};
    const HandoffVariant<true> parts{res, kRowsOut, norm, xs, kRowsOut, slots, 16, partial};
    const AdaNormHandoffResidualEpilogue r35{res, kRowsOut, norm, xs, kRowsOut, slots, nullptr, 0};
    const PlainResidual plain{res, kRowsOut};

    // Correctness: one launch of each from the same residual; residuals must match plain, sums must agree.
    std::vector<__nv_bfloat16> ref(hr.size()), got(hr.size());
    std::vector<unsigned long long> hs(256);
    std::vector<float> hp(320 * 16);
    auto reset = [&] { CK(cudaMemcpy(res, res0, hr.size() * 2, cudaMemcpyDeviceToDevice)); CK(cudaMemset(slots, 0, 16 * 16 * 8)); };
    reset();
    ada_small_t_mma_launch<S1, K>(x, K, dc[0], nullptr, ds[0], kRowsOut, T, plain, 0);
    CK(cudaDeviceSynchronize()); CK(cudaMemcpy(ref.data(), res, ref.size() * 2, cudaMemcpyDeviceToHost));
    double host_sum[16] = {};
    for (int t = 0; t < T; ++t) for (int r = 0; r < kRowsOut; ++r) { double v = __bfloat162float(ref[(long)t * kRowsOut + r]); host_sum[t] += v * v; }
    auto check = [&](const char* name, auto launch, int stride, bool use_parts) {
        reset(); launch(); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(got.data(), res, got.size() * 2, cudaMemcpyDeviceToHost));
        const bool same = memcmp(got.data(), ref.data(), ref.size() * 2) == 0;
        double worst = 0;
        if (use_parts) {
            CK(cudaMemcpy(hp.data(), partial, hp.size() * 4, cudaMemcpyDeviceToHost));
            for (int t = 0; t < T; ++t) { double s = 0; for (int b = 0; b < 320; ++b) s += hp[b * 16 + t]; worst = std::max(worst, fabs(s / host_sum[t] - 1)); }
        } else {
            CK(cudaMemcpy(hs.data(), slots, hs.size() * 8, cudaMemcpyDeviceToHost));
            for (int t = 0; t < T; ++t) worst = std::max(worst, fabs(hs[t * stride] / 65536.0 / host_sum[t] - 1));
        }
        printf("check %-10s residual %s  max rel sum err %.2e\n", name, same ? "EXACT" : "DIFF", worst);
    };
    check("r35", [&] { ada_small_t_mma_launch<S1, K>(x, K, dc[0], nullptr, ds[0], kRowsOut, T, r35, 0); }, 1, false);
    check("r35/mb3", [&] { ada_small_t_mma_launch<S3, K>(x, K, dc[0], nullptr, ds[0], kRowsOut, T, r35, 0); }, 1, false);
    check("lines/mb3", [&] { ada_small_t_mma_launch<S3, K>(x, K, dc[0], nullptr, ds[0], kRowsOut, T, lines, 0); }, 16, false);
    check("parts/mb3", [&] { ada_small_t_mma_launch<S3, K>(x, K, dc[0], nullptr, ds[0], kRowsOut, T, parts, 0); }, 16, true);

    const int reps = 4 * copies;
    for (int round = 0; round < 2; ++round) {
        printf("T=%d copies=%d  plain %.2f  plain/mb3 %.2f  r35 %.2f  r35/mb3 %.2f  lines/mb3 %.2f  parts/mb3 %.2f  parts %.2f us\n", T, copies,
            time_graph([&](cudaStream_t s, int i) { ada_small_t_mma_launch<S1, K>(x, K, dc[i % copies], nullptr, ds[i % copies], kRowsOut, T, plain, s); }, reps),
            time_graph([&](cudaStream_t s, int i) { ada_small_t_mma_launch<S3, K>(x, K, dc[i % copies], nullptr, ds[i % copies], kRowsOut, T, plain, s); }, reps),
            time_graph([&](cudaStream_t s, int i) { ada_small_t_mma_launch<S1, K>(x, K, dc[i % copies], nullptr, ds[i % copies], kRowsOut, T, r35, s); }, reps),
            time_graph([&](cudaStream_t s, int i) { ada_small_t_mma_launch<S3, K>(x, K, dc[i % copies], nullptr, ds[i % copies], kRowsOut, T, r35, s); }, reps),
            time_graph([&](cudaStream_t s, int i) { ada_small_t_mma_launch<S3, K>(x, K, dc[i % copies], nullptr, ds[i % copies], kRowsOut, T, lines, s); }, reps),
            time_graph([&](cudaStream_t s, int i) { ada_small_t_mma_launch<S3, K>(x, K, dc[i % copies], nullptr, ds[i % copies], kRowsOut, T, parts, s); }, reps),
            time_graph([&](cudaStream_t s, int i) { ada_small_t_mma_launch<S1, K>(x, K, dc[i % copies], nullptr, ds[i % copies], kRowsOut, T, parts, s); }, reps));
    }
    return 0;
}
