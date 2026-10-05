#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Research (NINFER_FORCE_TOKENS=FILE, research23): forced-text speculative decoding for clean A/Bs.
Spec-Bench A/Bs of a drafter (or target) change mix speed with content: under the verify tree, near-tie flips change
which greedy continuation is produced, and a near-null drafter perturbation reads +0.44% tok/step. With this knob the
greedy tree walk replaces the target's token at sequence position p by a reference id (FILE = the JSONL written by
NINFER_TOKEN_DUMP: {"prompt_tokens": P, "ids": [...]} per request, ids[0] = the prefill token). Requests are matched in
order by prompt length; unmatched requests run unforced. Every configuration then walks the same text: acceptance and
round time are compared on identical tokens. The target still computes every column, so round time is unchanged except
for two mapped-host reads in the accept kernel. Row 0 only (the benches run batch 1). NINFER_FORCE_SHIFT adds to the
position base. The prefill token is not forced (identical for a fixed target).
usage: patch_force_tokens.py REPO"""
import pathlib, sys

repo = pathlib.Path(sys.argv[1])


def edit(rel, old, new, count=1):
    p = repo / rel
    s = p.read_text()
    if new in s:
        return
    assert s.count(old) == count, (rel, old[:70], s.count(old))
    p.write_text(s.replace(old, new))


# 1. Kernel: optional forced ids for row 0.
edit("src/ops/kernel/speculative_round.cuh",
     """// Greedy rows without penalties walk the raw target argmax.
__global__ __launch_bounds__(256) void speculative_accept_tree_warp_greedy_kernel(
    const int* target_tokens, const int* drafts, const int* tree_parents,
    const int* current_extents, int* lengths, int* anchors, int* licensed_tokens,
    int* licensed_counts, int* accepted, int* accepted_columns, int k) {
    const int row    = threadIdx.x / 32;
    const int extent = min(k, max(0, current_extents[row]));
    speculative_tree_warp_walk(
        drafts, tree_parents, k, row, extent,
        [&](int node, int) { return target_tokens[row * (k + 1) + node]; }, lengths, anchors,
        licensed_tokens, licensed_counts, accepted, accepted_columns);
}
""", """// Greedy rows without penalties walk the raw target argmax.
// Research (NINFER_FORCE_TOKENS): `force` is a mapped host buffer {active, base, count, -, ids[count]}. While active,
// row 0 takes ids[lengths[row] + depth - base] instead of the target token (base = P - 1: ids[0] is the prefill
// token, and the first round's frontier is P).
__global__ __launch_bounds__(256) void speculative_accept_tree_warp_greedy_kernel(
    const int* target_tokens, const int* drafts, const int* tree_parents,
    const int* current_extents, int* lengths, int* anchors, int* licensed_tokens,
    int* licensed_counts, int* accepted, int* accepted_columns, int k, const int* force) {
    const int row    = threadIdx.x / 32;
    const int extent = min(k, max(0, current_extents[row]));
    int forced       = -1;
    if (force != nullptr && row == 0) {
        const volatile int* f = force;
        if (f[0] != 0) {
            const int idx = lengths[row] - f[1] + static_cast<int>(threadIdx.x & 31);
            if (idx >= 0 && idx < f[2]) forced = f[4 + idx];
        }
    }
    speculative_tree_warp_walk(
        drafts, tree_parents, k, row, extent,
        [&](int node, int depth) {
            const int t = __shfl_sync(0xffffffffU, forced, depth & 31);
            return t >= 0 ? t : target_tokens[row * (k + 1) + node];
        },
        lengths, anchors, licensed_tokens, licensed_counts, accepted, accepted_columns);
}
""")

# 2. Launcher: host side (reference records, mapped buffer, per-request arming) + the kernel argument.
edit("src/ops/launcher/speculative_round.cu",
     """#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
""", """#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::ops {
namespace {
// Research (NINFER_FORCE_TOKENS): reference ids per request (see speculative_accept_tree_warp_greedy_kernel).
struct ForcedRecord {
    std::uint32_t prompt_tokens = 0;
    std::vector<int> ids;
};
constexpr int kForceMax = 1 << 16;
std::vector<ForcedRecord> g_force_records;
std::size_t g_force_cursor = 0;
int* g_force_host          = nullptr;
int* g_force_device        = nullptr;
int g_force_shift          = 0;
bool g_force_init          = false;
long g_force_hits = 0, g_force_misses = 0;

void force_init() {
    if (g_force_init) return;
    g_force_init     = true;
    const char* path = std::getenv("NINFER_FORCE_TOKENS");
    if (path == nullptr || *path == 0) return;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        const auto p = line.find("\\"prompt_tokens\\":");
        const auto b = line.find('[');
        const auto e = line.rfind(']');
        if (p == std::string::npos || b == std::string::npos || e == std::string::npos) continue;
        ForcedRecord r;
        r.prompt_tokens =
            static_cast<std::uint32_t>(std::strtoul(line.c_str() + p + 16, nullptr, 10));
        const char* s   = line.c_str() + b + 1;
        const char* end = line.c_str() + e;
        while (s < end) {
            char* next   = nullptr;
            const long v = std::strtol(s, &next, 10);
            if (next == s) break;
            r.ids.push_back(static_cast<int>(v));
            s = next;
            while (s < end && (*s == ',' || *s == ' ')) ++s;
        }
        g_force_records.push_back(std::move(r));
    }
    if (const char* sh = std::getenv("NINFER_FORCE_SHIFT")) g_force_shift = std::atoi(sh);
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&g_force_host), sizeof(int) * (4 + kForceMax),
                             cudaHostAllocMapped));
    std::memset(g_force_host, 0, sizeof(int) * 4);
    CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&g_force_device), g_force_host, 0));
    std::fprintf(stderr, "NINFER_FORCE_TOKENS: %zu records from %s, shift %d\\n",
                 g_force_records.size(), path, g_force_shift);
}

const int* force_device_ptr(cudaStream_t stream) {
    if (!g_force_init) {
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
        if (status == cudaStreamCaptureStatusNone) {
            force_init();
        } else if (std::getenv("NINFER_FORCE_TOKENS") != nullptr) {
            std::fprintf(stderr, "NINFER_FORCE_TOKENS: first tree accept is under capture; not forced\\n");
        }
    }
    return g_force_device;
}
} // namespace

// Called by the engine when a request is admitted (before its prefill and decode rounds are launched).
void force_tokens_begin_request(std::uint32_t prompt_tokens) {
    force_init();
    if (g_force_host == nullptr) return;
    volatile int* h = g_force_host;
    h[0]            = 0;
    if (g_force_cursor < g_force_records.size() &&
        g_force_records[g_force_cursor].prompt_tokens == prompt_tokens) {
        const ForcedRecord& r = g_force_records[g_force_cursor++];
        const int n           = std::min<int>(static_cast<int>(r.ids.size()), kForceMax);
        for (int i = 0; i < n; ++i) h[4 + i] = r.ids[i];
        // ids[0] is the prefill token (position P); at the first round the frontier is P and the root predicts
        // ids[1], so the base is P - 1.
        h[1] = static_cast<int>(prompt_tokens) - 1 + g_force_shift;
        h[2] = n;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        h[0] = 1;
        ++g_force_hits;
    } else {
        ++g_force_misses;
        std::fprintf(stderr, "NINFER_FORCE_TOKENS: no record for a %u-token prompt at record %zu\\n",
                     prompt_tokens, g_force_cursor);
    }
    if ((g_force_hits + g_force_misses) % 100 == 0) {
        std::fprintf(stderr, "NINFER_FORCE_TOKENS: %ld requests forced, %ld not\\n", g_force_hits,
                     g_force_misses);
    }
}
} // namespace ninfer::ops

namespace ninfer::ops::detail {
""")
edit("src/ops/launcher/speculative_round.cu",
     """            static_cast<int*>(accepted_drafts.data), static_cast<int*>(accepted_columns.data), k);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
""", """            static_cast<int*>(accepted_drafts.data), static_cast<int*>(accepted_columns.data), k,
            force_device_ptr(stream));
        CUDA_CHECK(cudaGetLastError());
        return;
    }
""")

# 3. Engine: arm the reference for each admitted request.
edit("src/runtime/engine/engine_core.h",
     "namespace ninfer::runtime {\n",
     "namespace ninfer::ops {\n"
     "// Research (NINFER_FORCE_TOKENS, src/ops/launcher/speculative_round.cu).\n"
     "void force_tokens_begin_request(std::uint32_t prompt_tokens);\n"
     "} // namespace ninfer::ops\n\n"
     "namespace ninfer::runtime {\n")
edit("src/runtime/engine/engine_core.h",
     """        publish_generation_start(
            request, BeginSummary{.prompt_tokens        = summary.prompt_tokens,""",
     """        ninfer::ops::force_tokens_begin_request(summary.prompt_tokens);
        publish_generation_start(
            request, BeginSummary{.prompt_tokens        = summary.prompt_tokens,""")
print("ok")
