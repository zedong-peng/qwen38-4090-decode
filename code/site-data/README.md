# Data behind the site's figures

| tool | produces | figure |
|---|---|---|
| `race_capture.py` | per-chunk arrival times and texts of streamed answers (one SSE chunk = one verify round on Cinference) | real decoding replay |
| `blog_race.py` | tokens per chunk (re-tokenized; matches the server's token count on every stream) → `site/data/blog-race.json` | real decoding replay |
| `blog_trees.py` | the deployed 15-node trees per round from `NINFER_TREE_DUMP` + `NINFER_TOKEN_DUMP`, accepted path, bonus token → `site/data/blog-trees.json` | verify trees |
| `round_timeline.py` | the median decode round's kernel list from an nsys sqlite → `site/data/blog-rounds.json` | round timeline |

`blog_trees.py` replays the selector's best-first growth (as `tools/tree_sim.py` does) and keeps only rounds where
the replay reproduces the engine's own acceptance (153 of 155 on the published prompts). Prompts are the median
prompt, by tokens per round, of four Spec-Bench categories (question ids 122, 88, 413, 258); set
`SPEC_BENCH_QUESTIONS` to Spec-Bench's `question.jsonl`.
