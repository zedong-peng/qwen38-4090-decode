# SPDX-License-Identifier: Apache-2.0
"""NINFER_ANCHOR_BEFORE_LAST_USER=1 for ninfer-serve (off by default).

usage: python3 anchor_patch.py ENGINE_DIR     (the patched tree of code/engine-series; then rebuild ninfer-serve)

An agent harness may end each prompt with a user message that the next prompt no longer contains (a turn counter, a
reminder). The server resumes a conversation from the checkpoint it keeps at the end of the prompt, which sits after
that message, so every such turn is prefilled from the first token. With the flag set, a prompt whose last message is
a user message that follows an assistant or tool message keeps that checkpoint in front of the message instead. The
next request then resumes there whether or not the message is still in the prompt; if it is, that one message is
prefilled again.
"""
import sys
from pathlib import Path

root = Path(sys.argv[1])
p = root / "src/models/qwen3_5/frontend/chat_template.cpp"
s = p.read_text()
if "NINFER_ANCHOR_BEFORE_LAST_USER" in s:
    print("already patched")
    sys.exit(0)


def edit(old, new):
    global s
    assert s.count(old) == 1, old
    s = s.replace(old, new)


edit("#include <algorithm>\n#include <map>\n", "#include <algorithm>\n#include <cstdlib>\n#include <map>\n")
edit("""        if (begin > 0)
            result.rewrite_checkpoint = RewriteCheckpointByteSpec{
                .kind = RewriteCheckpointKind::TurnClosure, .offset = begin};
    }
""", """        if (begin > 0)
            result.rewrite_checkpoint = RewriteCheckpointByteSpec{
                .kind = RewriteCheckpointKind::TurnClosure, .offset = begin};
    }
    // NINFER_ANCHOR_BEFORE_LAST_USER=1: a trailing user message that follows an assistant or tool
    // message may be harness text the next prompt drops (a turn counter). Keep the recovery point
    // in front of it, so the next request resumes there instead of from the first token.
    static const bool anchor_before_last_user = [] {
        const char* value = std::getenv("NINFER_ANCHOR_BEFORE_LAST_USER");
        return value != nullptr && value[0] == '1';
    }();
    if (anchor_before_last_user && !continuation && options.add_generation_prompt &&
        messages.size() >= 3 && real_user(messages.back()) &&
        (messages[messages.size() - 2].role == ChatRole::Assistant ||
         messages[messages.size() - 2].role == ChatRole::Tool)) {
        const auto before = prefix(messages.size() - 1);
        if (before && *before > 0)
            result.rewrite_checkpoint = RewriteCheckpointByteSpec{
                .kind = RewriteCheckpointKind::TurnClosure, .offset = *before};
    }
""")
p.write_text(s)
print("patched", p)
