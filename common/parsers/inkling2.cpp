#include "parsers.h"

// Thinking Machines Inkling-2-Mini chat template parser:
//   assistant := [<|message_model|><|content_thinking|>{reasoning}<|end_message|>]
//                [<|message_model|><|content_text|>{content}<|end_message|>]
//                {<|message_model|>{fn_name}<|content_invoke_tool_json|>{"name":"{fn_name}","args":{...}}<|end_message|>}
//                [<|content_model_end_sampling|>]
common_chat_params common_chat_params_init_inkling2(const common_chat_template &          tmpl,
                                                    const autoparser::generation_params & inputs) {
    common_chat_params data;

    const std::string MSG_MODEL  = "<|message_model|>";
    const std::string MSG_USER   = "<|message_user|>";
    const std::string MSG_SYSTEM = "<|message_system|>";
    const std::string MSG_TOOL   = "<|message_tool|>";
    const std::string CNT_THINK  = "<|content_thinking|>";
    const std::string CNT_TEXT   = "<|content_text|>";
    const std::string CNT_IMG    = "<|content_image|>";
    const std::string CNT_XML    = "<|content_xml|>";
    const std::string CNT_TOOL   = "<|content_invoke_tool_json|>";
    const std::string END_MSG    = "<|end_message|>";
    const std::string END_SAMP   = "<|content_model_end_sampling|>";
    const std::string IMG_TOK    = "<|unused_200054|>";

    // Normalize messages for Inkling's strict Jinja checks:
    //  1) Rewrite server-side `media_marker` content parts to `image` before Jinja rendering,
    //     then substitute `<|unused_200054|>` with each media marker in the rendered prompt.
    //  2) Parse stringified `tool_calls[].function.arguments` into JSON objects.
    const char * env_marker = std::getenv("LLAMA_MEDIA_MARKER");
    const std::string default_marker = (env_marker && env_marker[0] != '\0') ? env_marker : "<__media__>";

    std::vector<std::string> media_markers;
    auto adjusted_messages = json::array();
    for (auto msg : inputs.messages) {
        if (msg.contains("content") && msg["content"].is_string()) {
            const std::string s = msg["content"].get<std::string>();
            if (s.find(default_marker) != std::string::npos) {
                auto new_parts = json::array();
                size_t pos = 0;
                while (pos < s.size()) {
                    const size_t hit = s.find(default_marker, pos);
                    if (hit == std::string::npos) {
                        const std::string rem = s.substr(pos);
                        if (!rem.empty()) {
                            new_parts.push_back(json{{"type", "text"}, {"text", rem}});
                        }
                        break;
                    }
                    if (hit > pos) {
                        new_parts.push_back(json{{"type", "text"}, {"text", s.substr(pos, hit - pos)}});
                    }
                    media_markers.push_back(default_marker);
                    new_parts.push_back(json{{"type", "image"}});
                    pos = hit + default_marker.size();
                }
                msg["content"] = std::move(new_parts);
            }
        } else if (msg.contains("content") && msg["content"].is_array()) {
            auto new_parts = json::array();
            for (auto part : msg["content"]) {
                if (part.is_object() && part.value("type", "") == "media_marker") {
                    media_markers.push_back(part.value("text", default_marker));
                    new_parts.push_back(json{{"type", "image"}});
                } else {
                    new_parts.push_back(std::move(part));
                }
            }
            msg["content"] = std::move(new_parts);
        }
        if (msg.contains("tool_calls") && msg["tool_calls"].is_array()) {
            for (auto & tc : msg["tool_calls"]) {
                if (tc.is_object() && tc.contains("function") && tc["function"].is_object() &&
                    tc["function"].contains("arguments") && tc["function"]["arguments"].is_string()) {
                    const std::string raw = tc["function"]["arguments"].get<std::string>();
                    tc["function"]["arguments"] = raw.empty() ? json::object() : json::parse(raw);
                }
            }
        }
        adjusted_messages.push_back(std::move(msg));
    }

    // Map `enable_thinking: false` to `reasoning_effort: "none"` unless explicitly overridden.
    std::optional<json> add_ctx = std::nullopt;
    if (!inputs.enable_thinking && !inputs.extra_context.contains("reasoning_effort")) {
        add_ctx = json{{"reasoning_effort", "none"}};
    }

    auto prompt = common_chat_template_direct_apply_impl(tmpl, inputs, adjusted_messages, std::nullopt, add_ctx);
    if (!media_markers.empty()) {
        const std::string needle = CNT_IMG + IMG_TOK + END_MSG;
        size_t m_idx = 0;
        for (auto & part : prompt.parts) {
            if (part.is_input) {
                continue;
            }
            size_t search_pos = 0;
            while (m_idx < media_markers.size()) {
                const size_t hit = part.text.find(needle, search_pos);
                if (hit == std::string::npos) {
                    break;
                }
                const std::string repl = CNT_IMG + media_markers[m_idx++] + END_MSG;
                part.text.replace(hit, needle.size(), repl);
                search_pos = hit + repl.size();
            }
        }
    }

    data.prompt             = std::move(prompt);
    data.generation_prompt  = common_chat_template_generation_prompt_impl(tmpl, inputs, adjusted_messages, std::nullopt, add_ctx);
    data.format             = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking  = true;
    data.thinking_start_tag = MSG_MODEL + CNT_THINK;
    data.thinking_end_tags  = { END_MSG };
    data.additional_stops   = { END_SAMP };
    data.preserved_tokens   = {
        MSG_MODEL, MSG_USER, MSG_SYSTEM, MSG_TOOL,
        CNT_THINK, CNT_TEXT, CNT_IMG, CNT_XML, CNT_TOOL,
        END_MSG, END_SAMP,
    };

    data.message_delimiters = {
        { COMMON_CHAT_ROLE_ASSISTANT, MSG_MODEL  },
        { COMMON_CHAT_ROLE_USER,      MSG_USER   },
        { COMMON_CHAT_ROLE_TOOL,      MSG_TOOL   },
        { COMMON_CHAT_ROLE_SYSTEM,    MSG_SYSTEM },
    };

    if (inputs.has_continuation()) {
        const auto & msg = inputs.continue_msg;
        if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_REASONING) {
            data.generation_prompt = MSG_MODEL + CNT_THINK + msg.reasoning_content;
        } else {
            data.generation_prompt.clear();
            if (!msg.reasoning_content.empty()) {
                data.generation_prompt += MSG_MODEL + CNT_THINK + msg.reasoning_content + END_MSG;
            }
            data.generation_prompt += MSG_MODEL + CNT_TEXT + msg.render_content();
        }
        data.prompt.append(data.generation_prompt);
    }

    const bool has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
    const bool has_response_format = inputs.json_schema.is_object() && !inputs.json_schema.empty();
    const bool extract_reasoning   = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    const bool include_grammar     = has_response_format || (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE);

    std::vector<std::string> tool_triggers;

    data.parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto end      = p.end();
        auto end_samp = p.optional(p.literal(END_SAMP));

        auto think_block = extract_reasoning
            ? p.rule("ink2-thinking", p.literal(MSG_MODEL + CNT_THINK) + p.reasoning(p.until(END_MSG)) + p.literal(END_MSG))
            : p.rule("ink2-thinking",
                     p.atomic(p.content(p.literal(MSG_MODEL + CNT_THINK))) +
                         p.content(p.until(END_MSG)) +
                         p.atomic(p.content(p.literal(END_MSG))));

        auto text_block = has_response_format
            ? p.rule("ink2-text",
                     p.literal(MSG_MODEL + CNT_TEXT) +
                         p.content(p.schema(p.json(), "response-format-schema", inputs.json_schema)) +
                         p.optional(p.literal(END_MSG)))
            : p.rule("ink2-text",
                     p.literal(MSG_MODEL + CNT_TEXT) + p.content(p.until(END_MSG)) + p.optional(p.literal(END_MSG)));

        if (!has_tools || inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE) {
            if (has_response_format) {
                return p.zero_or_more(think_block) + text_block + end_samp + end;
            }
            return p.zero_or_more(think_block) + p.optional(text_block) + end_samp + end;
        }

        auto tool_choice = p.choice();
        foreach_function(inputs.tools, [&](size_t tool_index, const json & tool) {
            const auto & function = tool.at("function");
            std::string  name     = function.at("name");
            const auto   params   = common_chat_tool_parameters(function);

            tool_triggers.push_back(MSG_MODEL + name + CNT_TOOL);

            auto open = p.tool_open(
                p.literal(MSG_MODEL) + p.tool_name(p.literal(name)) + p.literal(CNT_TOOL) +
                p.space() + p.literal("{") + p.space() +
                p.literal("\"name\"") + p.space() + p.literal(":") + p.space() +
                p.literal(json(name).dump()) + p.space() + p.literal(",") + p.space() +
                p.literal("\"args\"") + p.space() + p.literal(":") + p.space());

            auto args = p.tool_args(p.schema(p.json(), "ink2-tool-" + std::to_string(tool_index) + "-schema", params));
            auto close = p.tool_close(p.space() + p.literal("}") + p.optional(p.literal(END_MSG)));

            tool_choice |= p.rule("tool-" + std::to_string(tool_index), p.tool(open + args + close));
        });

        auto calls = inputs.parallel_tool_calls ? p.one_or_more(tool_choice) : tool_choice;
        auto tool_section = p.trigger_rule("tool-call", calls + end_samp);
        auto tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? tool_section : p.optional(tool_section);

        return p.zero_or_more(think_block) + p.optional(text_block) + tools + end_samp + end;
    });

    if (include_grammar) {
        data.grammar_lazy = !(has_response_format || (has_tools && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED));
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            data.parser.build_grammar(builder, data.grammar_lazy);
        });
        if (data.grammar_lazy) {
            for (const auto & trig : tool_triggers) {
                data.grammar_triggers.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_WORD, trig });
            }
        }
    }

    return data;
}
