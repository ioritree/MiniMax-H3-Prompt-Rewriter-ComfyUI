import { app } from "../../scripts/app.js";

// Workflows saved on this fork before it took upstream 0.36.0 (e7cbe6c) carry
// output_language and the Options node's CPU/KV widgets at positions upstream
// has since filled: aspect_ratio after keep_model_loaded, repeat_last and
// library_pick after bypass, and six new Options widgets ahead of threads.
// ComfyUI restores widgets_values by position, so those graphs came back with
// output_language '' and self_check -1, and validation refused them.
//
// Each layout below is the widget order of that release. A saved array of
// exactly that length, with a language name where output_language sat, is
// dealt back out by name; every widget the layout does not name gets its
// default back, since positional restore has already written a stray value
// into it. trust_remote_code is gone upstream and its value is dropped.
const LANGUAGES = ["繁體中文", "简体中文", "English", "日本語", "한국어"];
const SEED = ["seed", "control_after_generate"];

const LAYOUTS = {
    MiniMaxH3PromptRewriter: [
        "prompt", "model", "resolution", "duration", "quantization", "greedy", ...SEED,
        "keep_model_loaded", "bypass", "output_language",
    ],
    MiniMaxH3GuidedWriter: [
        "prompt", "model", "task", "resolution", "duration", "greedy", ...SEED,
        "keep_model_loaded", "reference_material", "bypass", "output_language", "system_prompt",
    ],
    MiniMaxH3GuidedWriterRef: [
        "prompt", "reference_assets", "model", "resolution", "duration", "greedy", ...SEED,
        "keep_model_loaded", "bypass", "output_language", "system_prompt",
    ],
    MiniMaxH3GuidePrompt: [
        "prompt", "task", "resolution", "duration", "reference_material", "auto_download",
        "format", "output_language",
    ],
    MiniMaxH3PromptWriter8B: [
        "prompt", "model", "task", "resolution", "duration", "quantization", "greedy", ...SEED,
        "keep_model_loaded", "bypass", "output_language",
    ],
    MiniMaxH3PromptWriterOmni: [
        "reference_layout", "task", "resolution", "duration", "prompt", "model", "quantization",
        "greedy", ...SEED, "keep_model_loaded", "max_frames", "bypass", "output_language",
    ],
    MiniMaxH3RewriterOptions: [
        "max_new_tokens", "temperature", "top_p", "top_k", "repetition_penalty",
        "attn_implementation", "adapter", "use_lora", "auto_download", "gpu_layers", "n_ctx",
        "gguf_runtime", "device", "llama_backend", "trust_remote_code",
        "threads", "poll", "cache_type", "flash_attn",
    ],
};

function savedOnOldFork(layout, values) {
    if (!Array.isArray(values) || values.length !== layout.length) return false;
    const language = layout.indexOf("output_language");
    if (language >= 0) return LANGUAGES.includes(values[language]);
    // Options: trust_remote_code's boolean, then threads' integer.
    const threads = layout.indexOf("threads");
    return typeof values[threads - 1] === "boolean" && Number.isInteger(values[threads]);
}

app.registerExtension({
    name: "minimax_h3_rewriter.fork_migration",
    beforeRegisterNodeDef(nodeType, nodeData) {
        const layout = LAYOUTS[nodeData.name];
        if (!layout) return;

        const configure = nodeType.prototype.configure;
        nodeType.prototype.configure = function (info, ...rest) {
            const defaults = new Map((this.widgets ?? []).map((widget) => [widget.name, widget.value]));
            const result = configure.call(this, info, ...rest);
            const values = info?.widgets_values;
            if (!savedOnOldFork(layout, values)) return result;

            for (const widget of this.widgets ?? []) {
                if (defaults.has(widget.name)) widget.value = defaults.get(widget.name);
            }
            layout.forEach((name, index) => {
                const widget = this.widgets?.find((one) => one.name === name);
                if (widget) widget.value = values[index];
            });
            this.setDirtyCanvas?.(true, true);
            console.log(
                `[minimax_h3_rewriter] ${nodeData.name} saved before 0.36.0 on this fork: realigned its widget values`
            );
            return result;
        };
    },
});
