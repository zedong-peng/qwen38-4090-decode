#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Research (NINFER_HIDDEN_DUMP=FILE): decode-path fidelity. The prefill fidelity tool never runs the verify forward,
so per-round numerics (FP16 state, norm handoff, ...) are invisible to it. With this knob a forced Spec-Bench run
appends to FILE, for batch row 0:
- at a request's first prefill chunk: i32 1, i32 P, i32 prompt_ids[P];
- per verify round, after the accept: i32 2, i32 frontier (before the accept), i32 count, bf16 [count][hidden],
  the final-normed head inputs of the accepted path (count = accepted drafts + 1).
Column c of a round predicts the token at sequence position frontier + 1 + c. tools/dfid.py turns the dumps into
logits with the checkpoint's head and scores them against tools/ref_var.py on the same text.
Run with --no-cuda-graph: the host code of a captured round runs once.
usage: patch_hidden_dump.py REPO"""
import os, sys

repo = sys.argv[1]
MARK = "research_hidden_dump"

HEADER = '''// Research (NINFER_HIDDEN_DUMP=FILE, tools/patch_hidden_dump.py): decode-path fidelity dumps.
#pragma once

#include <cstdio>
#include <cstdlib>

namespace ninfer::models::qwen3_5::execution {

inline std::FILE* research_hidden_dump() {
    static std::FILE* const file = []() -> std::FILE* {
        const char* path = std::getenv("NINFER_HIDDEN_DUMP");
        return path != nullptr && *path != 0 ? std::fopen(path, "wb") : nullptr;
    }();
    return file;
}

} // namespace ninfer::models::qwen3_5::execution
'''


def patch(rel, pairs):
    p = os.path.join(repo, rel)
    s = open(p).read()
    if MARK in s:
        print("already patched", rel)
        return
    for old, new in pairs:
        assert s.count(old) == 1, (rel, old[:80], s.count(old))
        s = s.replace(old, new)
    open(p + ".tmp", "w").write(s)
    os.replace(p + ".tmp", p)
    print("patched", rel)


h = os.path.join(repo, "src/models/qwen3_5/program/research_hidden_dump.h")
if not os.path.exists(h):
    open(h, "w").write(HEADER)
    print("wrote", h)

patch("src/models/qwen3_5/program/speculative/target_verification.cpp", [
    ('''#include "models/qwen3_5/program/context.h"
''', '''#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/program/research_hidden_dump.h"
#include "core/device.h"
'''),
    ('''#include "ninfer/ops/speculative_round.h"
''', '''#include "ninfer/ops/speculative_round.h"

#include <cstdint>
#include <vector>
'''),
    ('''    card.set_verify_tree(nullptr, nullptr);
    if (tree) {''', '''    card.set_verify_tree(nullptr, nullptr);
    std::FILE* const dump      = research_hidden_dump();
    std::int32_t dump_frontier = 0;
    if (dump != nullptr) {
        CUDA_CHECK(cudaMemcpyAsync(&dump_frontier, frame.frontiers.data, sizeof(dump_frontier),
                                   cudaMemcpyDeviceToHost, execution.device.stream));
        CUDA_CHECK(cudaStreamSynchronize(execution.device.stream));
    }
    if (tree) {'''),
    ('''    ops::speculative_select_accepted_hidden(frame.target_hidden, frame.accepted_drafts,''',
     '''    if (dump != nullptr) {
        std::int32_t accepted = 0;
        CUDA_CHECK(cudaMemcpyAsync(&accepted, frame.accepted_drafts.data, sizeof(accepted),
                                   cudaMemcpyDeviceToHost, execution.device.stream));
        CUDA_CHECK(cudaStreamSynchronize(execution.device.stream));
        const std::int32_t count = accepted + 1;
        std::vector<std::uint16_t> host(static_cast<std::size_t>(count) *
                                        static_cast<std::size_t>(frame.target_hidden.ne[0]));
        CUDA_CHECK(cudaMemcpy(host.data(), frame.target_hidden.data, host.size() * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost));
        const std::int32_t head[3] = {2, dump_frontier, count};
        std::fwrite(head, sizeof(head), 1, dump);
        std::fwrite(host.data(), sizeof(std::uint16_t), host.size(), dump);
    }
    ops::speculative_select_accepted_hidden(frame.target_hidden, frame.accepted_drafts,'''),
])

patch("src/models/qwen3_5/program/prefill.cpp", [
    ('''#include "models/qwen3_5/program/context.h"
''', '''#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/program/research_hidden_dump.h"
'''),
    ('''    const std::span<const int> prompt(ids.data(), ids.size());
    if (state.dflash != nullptr) {
        DFlashFeatureSink sink = make_dflash_prefill_sink(state);
        return card.prefill_chunk(prompt, state.text_kv_base, nominal_length, finalize_at_end,
                                  sink);
    }''', '''    const std::span<const int> prompt(ids.data(), ids.size());
    if (std::FILE* const dump = research_hidden_dump(); dump != nullptr && state.text_kv_base == 0) {
        const std::int32_t head[2] = {1, static_cast<std::int32_t>(ids.size())};
        std::fwrite(head, sizeof(head), 1, dump);
        std::fwrite(ids.data(), sizeof(TokenId), ids.size(), dump);
        std::fflush(dump);
    }
    if (state.dflash != nullptr) {
        DFlashFeatureSink sink = make_dflash_prefill_sink(state);
        return card.prefill_chunk(prompt, state.text_kv_base, nominal_length, finalize_at_end,
                                  sink);
    }'''),
])
