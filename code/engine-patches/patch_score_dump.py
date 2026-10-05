#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Research: dump causal-scoring distributions for a cross-engine fidelity metric (KL / top-1).

NINFER_SCORE_DUMP=PREFIX makes ninfer-perplexity's scoring path write, in scoring order:
  PREFIX.windows.bin  per window: u32 token_count, u32 first_target, i32 tokens[token_count]
  PREFIX.topk.bin     per scored column: f32 logsumexp, i32 ids[64], f32 logits[64] (descending)
and, with NINFER_SCORE_REF=REF.topk.bin (a reference run's topk file over the same windows),
  PREFIX.atref.bin    per scored column: f32 logits at the reference's 64 ids.
Logits are the BF16 head outputs over the public vocabulary rows. No effect when unset.
"""
import sys

root = sys.argv[1]
P = f"{root}/src/models/qwen3_5/program/program_impl.cpp"
s = open(P).read()

s = s.replace('#include "ninfer/ops/target_logprobs.h"\n', '''#include "ninfer/ops/target_logprobs.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
''', 1)

anchor = "std::vector<float> ProgramImpl::causal_score(PreparedPromptData&& prompt,"
assert s.count(anchor) == 1
s = s.replace(anchor, r'''namespace {
// Research dump for cross-engine fidelity (see tools/patch_score_dump.py on the box).
class ScoreDump {
public:
    static constexpr int kTop = 64;
    static ScoreDump* get() {
        static ScoreDump* dump = [] () -> ScoreDump* {
            const char* prefix = std::getenv("NINFER_SCORE_DUMP");
            return prefix != nullptr ? new ScoreDump(prefix, std::getenv("NINFER_SCORE_REF")) : nullptr;
        }();
        return dump;
    }
    void window(const std::vector<TokenId>& tokens, std::uint32_t first_target) {
        const std::uint32_t n = static_cast<std::uint32_t>(tokens.size());
        std::fwrite(&n, 4, 1, windows_);
        std::fwrite(&first_target, 4, 1, windows_);
        for (const TokenId t : tokens) {
            const std::int32_t v = static_cast<std::int32_t>(t);
            std::fwrite(&v, 4, 1, windows_);
        }
        std::fflush(windows_);
    }
    // logits: device BF16 [physical_rows, columns], column-contiguous.
    void tile(const Tensor& logits, std::uint32_t columns, std::int32_t valid_rows,
              cudaStream_t stream) {
        const std::int64_t rows = logits.ne[0];
        host_.resize(static_cast<std::size_t>(rows) * columns);
        CUDA_CHECK(cudaMemcpyAsync(host_.data(), logits.data, host_.size() * 2,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<std::int32_t> ref_ids;
        if (ref_ != nullptr) {
            ref_ids.resize(static_cast<std::size_t>(columns) * kTop);
            for (std::uint32_t c = 0; c < columns; ++c) {
                float lse;
                float logits_unused[kTop];
                if (std::fread(&lse, 4, 1, ref_) != 1 ||
                    std::fread(&ref_ids[static_cast<std::size_t>(c) * kTop], 4, kTop, ref_) != kTop ||
                    std::fread(logits_unused, 4, kTop, ref_) != kTop) {
                    throw std::runtime_error("NINFER_SCORE_REF ended before the scored columns");
                }
            }
        }
        struct Record { float lse; std::int32_t ids[kTop]; float logits[kTop]; float at_ref[kTop]; };
        std::vector<Record> records(columns);
        const auto work = [&](std::uint32_t begin, std::uint32_t end) {
            std::vector<std::pair<float, std::int32_t>> heap;
            for (std::uint32_t c = begin; c < end; ++c) {
                const std::uint16_t* col = host_.data() + static_cast<std::size_t>(c) * rows;
                const auto value = [&](std::int32_t r) {
                    std::uint32_t u = static_cast<std::uint32_t>(col[r]) << 16;
                    float f;
                    std::memcpy(&f, &u, 4);
                    return f;
                };
                float m = -INFINITY;
                for (std::int32_t r = 0; r < valid_rows; ++r) m = std::max(m, value(r));
                double z = 0.0;
                heap.clear();
                for (std::int32_t r = 0; r < valid_rows; ++r) {
                    const float v = value(r);
                    z += std::exp(static_cast<double>(v - m));
                    if (static_cast<int>(heap.size()) < kTop) {
                        heap.emplace_back(v, r);
                        if (static_cast<int>(heap.size()) == kTop) {
                            std::make_heap(heap.begin(), heap.end(), std::greater<>());
                        }
                    } else if (v > heap.front().first) {
                        std::pop_heap(heap.begin(), heap.end(), std::greater<>());
                        heap.back() = {v, r};
                        std::push_heap(heap.begin(), heap.end(), std::greater<>());
                    }
                }
                std::sort(heap.begin(), heap.end(), [](const auto& a, const auto& b) {
                    return a.first > b.first || (a.first == b.first && a.second < b.second);
                });
                Record& rec = records[c];
                rec.lse     = m + static_cast<float>(std::log(z));
                for (int i = 0; i < kTop; ++i) {
                    rec.ids[i]    = heap[i].second;
                    rec.logits[i] = heap[i].first;
                    if (!ref_ids.empty()) {
                        rec.at_ref[i] = value(ref_ids[static_cast<std::size_t>(c) * kTop + i]);
                    }
                }
            }
        };
        const unsigned threads = 16;
        std::vector<std::thread> pool;
        for (unsigned t = 0; t < threads; ++t) {
            const std::uint32_t b = columns * t / threads, e = columns * (t + 1) / threads;
            pool.emplace_back(work, b, e);
        }
        for (auto& th : pool) th.join();
        for (const Record& rec : records) {
            std::fwrite(&rec.lse, 4, 1, topk_);
            std::fwrite(rec.ids, 4, kTop, topk_);
            std::fwrite(rec.logits, 4, kTop, topk_);
            if (atref_ != nullptr) std::fwrite(rec.at_ref, 4, kTop, atref_);
        }
        std::fflush(topk_);
        if (atref_ != nullptr) std::fflush(atref_);
    }

private:
    ScoreDump(const std::string& prefix, const char* ref) {
        windows_ = std::fopen((prefix + ".windows.bin").c_str(), "wb");
        topk_    = std::fopen((prefix + ".topk.bin").c_str(), "wb");
        if (ref != nullptr) {
            ref_   = std::fopen(ref, "rb");
            atref_ = std::fopen((prefix + ".atref.bin").c_str(), "wb");
            if (ref_ == nullptr) throw std::runtime_error("cannot open NINFER_SCORE_REF");
        }
        if (windows_ == nullptr || topk_ == nullptr) throw std::runtime_error("cannot open NINFER_SCORE_DUMP files");
    }
    std::FILE* windows_ = nullptr;
    std::FILE* topk_    = nullptr;
    std::FILE* ref_     = nullptr;
    std::FILE* atref_   = nullptr;
    std::vector<std::uint16_t> host_;
};
} // namespace

''' + anchor)

old = '''    if (prompt.has_media()) {
        throw std::invalid_argument("causal scoring accepts text tokens only");
    }
'''
assert s.count(old) == 1
s = s.replace(old, old + '''    ScoreDump* const dump = ScoreDump::get();
    if (dump != nullptr) { dump->window(prompt.token_ids, first_target); }
''')
old = '''            device.synchronize();
            const auto* host = static_cast<const float*>(score_logprobs_host->data());'''
assert s.count(old) == 1
s = s.replace(old, '''            device.synchronize();
            if (dump != nullptr) {
                dump->tile(logits, staged_columns,
                           dimension(parameters.model.resources().public_token_count),
                           device.stream);
            }
            const auto* host = static_cast<const float*>(score_logprobs_host->data());''')
open(P, "w").write(s)
print("patched")
