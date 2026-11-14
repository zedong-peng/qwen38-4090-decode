# SPDX-License-Identifier: Apache-2.0
"""Two protocol fields AA-AgentPerf-Local needs from ninfer-serve's chat completions; neither touches the engine.

usage: python3 aa_serve_patch.py ENGINE_DIR     (the patched tree of code/engine-series; then rebuild ninfer-serve)

  ignore_eos     request field: keep generating past the model's stop tokens, so a turn has its recorded length.
  finish_reason  "length" whenever the output limit ended the turn. Stock ninfer-serve answers "tool_calls" if the
                 cut-off text happens to parse as a complete tool call, which the tool counts as a short turn.
"""
import sys
from pathlib import Path

root = Path(sys.argv[1])


def edit(rel, old, new, count=1):
    p = root / rel
    s = p.read_text()
    if new in s:
        return
    assert s.count(old) == count, (rel, old)
    p.write_text(s.replace(old, new))


edit("src/serve/request.h",
     "    bool stop_strings_apply_to_reasoning = false;\n",
     "    bool stop_strings_apply_to_reasoning = false;\n"
     "    bool ignore_eos                      = false; // keep generating past the model's stop tokens\n")
edit("src/serve/openai_chat_request.cpp",
     "    parse_stop(body, output.generation);\n",
     "    parse_stop(body, output.generation);\n"
     "    if (body.contains(\"ignore_eos\") && !body.at(\"ignore_eos\").is_null()) {\n"
     "        if (!body.at(\"ignore_eos\").is_boolean()) { bad_request(\"ignore_eos must be a boolean\", \"ignore_eos\"); }\n"
     "        output.generation.ignore_eos = body.at(\"ignore_eos\").get<bool>();\n"
     "    }\n")
edit("src/serve/translate.cpp",
     "    options.output.raw                     = false;\n",
     "    options.output.raw                     = false;\n"
     "    options.stop.include_model_defaults    = !request.ignore_eos;\n")

R = "src/serve/openai_chat_response.cpp"
edit(R,
     "std::vector<ToolCall>\nmaterialize_tool_calls(",
     "bool hit_limit(ninfer::FinishReason reason) {\n"
     "    return reason == ninfer::FinishReason::OutputLimit || reason == ninfer::FinishReason::ContextCapacity;\n"
     "}\n\n"
     "std::vector<ToolCall>\nmaterialize_tool_calls(")
edit(R,
     "               has_tool_calls ? Json(\"tool_calls\") : Json(finish_reason(outcome.finish_reason))}}});\n",
     "               has_tool_calls && !hit_limit(outcome.finish_reason)\n"
     "                   ? Json(\"tool_calls\")\n"
     "                   : Json(finish_reason(outcome.finish_reason))}}});\n")
edit(R,
     "        events.push_back(chunk(identity_, Json::object(), \"tool_calls\", include_usage_,\n",
     "        events.push_back(chunk(identity_, Json::object(),\n"
     "                               hit_limit(outcome.finish_reason) ? \"length\" : \"tool_calls\", include_usage_,\n")
print("patched")
