# Data behind the site's figures

| tool | produces | figure |
|---|---|---|
| `race_capture.py` | arrival times and texts of streamed answers from any OpenAI-compatible server (one SSE chunk per verify round on Cinference and vLLM, one per token on llama.cpp) | real decoding replay |
| `blog_race.py` | rounds (llama.cpp events less than 2 ms apart are merged) and tokens per round (re-tokenized) → `site/data/blog-race.json` | real decoding replay |
| `famous_compile.py` | Spec-Bench per task group (tok/s, tokens per round, ms per round) and suite medians for every engine of the one-session comparison; vLLM's tokens per round come from its Prometheus spec-decode counters → `site/data/famous.json` | leaderboard, throughput plane |
| `blog_trees.py` | the deployed 15-node trees per round from `NINFER_TREE_DUMP` + `NINFER_TOKEN_DUMP`, accepted path, bonus token → `site/data/blog-trees.json` | verify trees |
| `round_timeline.py` | the median decode round's kernel list from an nsys sqlite → `site/data/blog-rounds.json` | round timeline |

`blog_trees.py` replays the selector's best-first growth (as `tools/tree_sim.py` does) and keeps only rounds where
the replay reproduces the engine's own acceptance (153 of 155 on the published prompts). Prompts are the median
prompt, by tokens per round, of four Spec-Bench categories (question ids 122, 88, 413, 258); set
`SPEC_BENCH_QUESTIONS` to Spec-Bench's `question.jsonl`.
