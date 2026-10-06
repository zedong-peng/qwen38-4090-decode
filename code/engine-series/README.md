# Engine patch series (53 commits)

These are the Ada (RTX 4090) and research commits behind the speed study, exported with `git format-patch`. Every
research feature is behind an environment knob that defaults to off, and each commit message states its measured
effect. Adopted knobs (best stack):

```
NINFER_TWO_LEVEL_HEAD=1 NINFER_LS8=1 NINFER_HEAD_SCREEN=3 NINFER_GDN_CHUNKED_RECORD=1 NINFER_TREE_TEMPERATURE=1.25
NINFER_Q3_SHADOW=1 NINFER_Q3_TILED=1 NINFER_DRAFT_CONV_Q4=1 NINFER_KV_VI8=1 NINFER_PROPOSAL_SCREEN=3 NINFER_CONV_EPILOGUE=1
NINFER_GDN_STATE_F16=1 NINFER_L2_FILL=121 NINFER_L2_FILL_NORM_KB=1536 NINFER_L2_FILL_POST_KB=768
NINFER_L2_FILL_GATED_KB=768 NINFER_L2_FILL_RECORD_KB=2048 NINFER_L2_FILL_ATTN_KB=4096 NINFER_ATTN_ROWSPLIT=3
NINFER_SCREEN_TILED=1
ninfer-serve ... --spec dflash2 --draft-tokens 15 --verify-tree --kv-dtype k8v4 --no-prefix-reuse
```

Build the engine whole-program (patch 0052): configure a separate build directory with `-DNINFER_RDC=OFF`. It is
bit-exact and about 0.5% faster than the default `-rdc=true` build.

## Base

The series applies on `base.diff`, which takes `jram4/ninfer-4090` at `70ebb1290dc7abe246c20696a24d21f77faee8d4` to
our local merge commit:
- `cbf7b7e` ("feat: sample MTP drafts with exact rejection correction") reverted. It conflicts with upstream's own
  sparse-proposal and tree logic.
- `satellitedown/cinference` at `383e5dbbaf927b562b11d7d48dbe4d10c1bf39bd` merged in, which brings verify trees,
  prompt-lookup chains and the DFlash2 kernel passes. The merge conflicts in eight files; `base.diff` carries our
  resolution.

```bash
git checkout 70ebb1290dc7abe246c20696a24d21f77faee8d4
git apply --index base.diff && git commit -m "Base: Cinference 383e5db merged into jram4 70ebb12, cbf7b7e reverted"
git am *.patch
```

The result has the same tree as the source of the binary behind our numbers. ../../REPRODUCE.md has the build and run
steps.

Patch 0001 holds the Ada fixes the merge needs: E2M1 decode by FP16 bit placement, an mbarrier `test_wait`
fallback, and NVFP4 stubs that fail closed.

## License

The engine is Apache-2.0: NInfer (Neroued) → Cinference (satellitedown) → the jram4 Ada port. These patches are
modifications of that code under the same license; see ../LICENSE and ../NOTICE.
