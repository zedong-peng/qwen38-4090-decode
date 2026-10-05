// research41 probe: the same Ada small-T kernels built with -rdc=true (as the engine) or whole program. Out projection
// (Q4, K = 6144, 5120 rows, residual epilogue) and Q4 gate/up (K = 5120, 34816 rows, paired SwiGLU halves) at T = 16,
// cold (weights rotate over >= 320 MB) and "warm prefix" (a read kernel pulls the first WARM_MB of the next copy's
// codes into L2 right before each GEMM, as the engine's L2 fills do).
// usage: bench_rdc [WARM_MB=8]
#include "ops/linear/ada_small_t_mma.cuh"

#include <cuda_bf16.h>
#include <algorithm>
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

using SOut = Q4AdaSmallTSchedule<2, 8, 2, 8, 2>;
using SGu  = Q4AdaSmallTSchedule<2, 8, 2, 8, 2>;

struct Residual {
    static constexpr bool kPairedHalves = false;
    __nv_bfloat16* out;
    int ld;
    __device__ __forceinline__ void operator()(int row, int token, float v) const {
        __nv_bfloat16* p = out + static_cast<long>(token) * ld + row;
        *p = __float2bfloat16(v + __bfloat162float(*p));
    }
};
struct SwiGlu {
    static constexpr bool kPairedHalves = true;
    __nv_bfloat16* out;
    int ld;
    __device__ __forceinline__ void operator()(int row, int token, float g, float u) const {
        out[static_cast<long>(token) * ld + row] = __float2bfloat16(g / (1.0f + __expf(-g)) * u);
    }
};

__global__ void warm_kernel(const uint4* __restrict__ p, long n, unsigned* sink) {
    unsigned acc = 0;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        const uint4 v = __ldcg(p + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) sink[0] = acc;
}

struct Weights {
    std::vector<uint8_t*> codes, scales;
    size_t code_bytes = 0;
    int copies = 0;
};

static Weights make(int rows, int k) {
    Weights w;
    const long groups = (long)rows * (k / 64);
    std::vector<uint8_t> codes(groups * 32), scales(groups * 2);
    unsigned seed = 1;
    for (auto& b : codes) { seed = seed * 1664525u + 1013904223u; b = seed >> 24; }
    for (long g = 0; g < groups; ++g) { uint16_t s = 0x2000 + (static_cast<uint32_t>(g * 2654435761u) >> 22); memcpy(&scales[g * 2], &s, 2); }
    w.code_bytes = codes.size();
    w.copies = (int)std::max<long>(2, ((320L << 20) + codes.size() + scales.size() - 1) / (codes.size() + scales.size()));
    for (int c = 0; c < w.copies; ++c) {
        uint8_t *a, *b;
        CK(cudaMalloc(&a, codes.size())); CK(cudaMemcpy(a, codes.data(), codes.size(), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&b, scales.size())); CK(cudaMemcpy(b, scales.data(), scales.size(), cudaMemcpyHostToDevice));
        w.codes.push_back(a); w.scales.push_back(b);
    }
    return w;
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
    const int T = 16;
    const long warm = (argc > 1 ? atol(argv[1]) : 8) << 20;
    std::vector<__nv_bfloat16> hx((long)T * 6144);
    unsigned seed = 7;
    for (auto& v : hx) { seed = seed * 1664525u + 1013904223u; v = __float2bfloat16(0.05f * (((seed >> 8) & 0xffff) / 65536.0f - 0.5f)); }
    __nv_bfloat16 *x, *res, *act; unsigned* sink;
    CK(cudaMalloc(&x, hx.size() * 2)); CK(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&res, (long)T * 5120 * 2)); CK(cudaMemset(res, 0, (long)T * 5120 * 2));
    CK(cudaMalloc(&act, (long)T * 17408 * 2)); CK(cudaMalloc(&sink, 4));
    Weights wo = make(5120, 6144), wg = make(34816, 5120);
    const Residual re{res, 5120};
    const SwiGlu sw{act, 17408};
    auto out = [&](cudaStream_t s, int i, bool w) {
        const int c = i % wo.copies;
        if (w) warm_kernel<<<256, 256, 0, s>>>(reinterpret_cast<const uint4*>(wo.codes[c]), std::min<long>(warm, wo.code_bytes) / 16, sink);
        ada_small_t_mma_launch<SOut, 6144>(x, 6144, wo.codes[c], nullptr, wo.scales[c], 5120, T, re, s);
    };
    auto gu = [&](cudaStream_t s, int i, bool w) {
        const int c = i % wg.copies;
        if (w) warm_kernel<<<256, 256, 0, s>>>(reinterpret_cast<const uint4*>(wg.codes[c]), std::min<long>(warm, wg.code_bytes) / 16, sink);
        ada_small_t_mma_launch<SGu, 5120, AdaPairedHalfRows<17408>>(x, 5120, wg.codes[c], nullptr, wg.scales[c], 34816, T, sw, s);
    };
    auto warm_only = [&](cudaStream_t s, int i, const Weights& w) {
        warm_kernel<<<256, 256, 0, s>>>(reinterpret_cast<const uint4*>(w.codes[i % w.copies]), std::min<long>(warm, w.code_bytes) / 16, sink);
    };
    for (int round = 0; round < 2; ++round) {
        const float oc = time_graph([&](cudaStream_t s, int i) { out(s, i, false); }, 4 * wo.copies);
        const float ow = time_graph([&](cudaStream_t s, int i) { out(s, i, true); }, 4 * wo.copies);
        const float owo = time_graph([&](cudaStream_t s, int i) { warm_only(s, i, wo); }, 4 * wo.copies);
        const float gc = time_graph([&](cudaStream_t s, int i) { gu(s, i, false); }, 4 * wg.copies);
        const float gw = time_graph([&](cudaStream_t s, int i) { gu(s, i, true); }, 4 * wg.copies);
        const float gwo = time_graph([&](cudaStream_t s, int i) { warm_only(s, i, wg); }, 4 * wg.copies);
        printf("out-proj cold %.2f  warm-prefix %.2f (warm-only %.2f, net %.2f)   gate/up cold %.2f  warm-prefix %.2f (warm-only %.2f, net %.2f) us\n",
               oc, ow, owo, ow - owo, gc, gw, gwo, gw - gwo);
    }
    return 0;
}
