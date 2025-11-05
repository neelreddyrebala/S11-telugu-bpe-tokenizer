import gradio as gr
import json, os
from bpe import BPETokenizer

# --- Handle filename/prefix differences --------------------------------------
ART_DIR = "artifacts"
PREFIX = os.path.join(ART_DIR, "telugu_bpe_tokenizer")       # load/save uses a *prefix*
EXPECTED = PREFIX + "_tokenizer.json"                        # what BPETokenizer.save() writes
LEGACY = os.path.join(ART_DIR, "telugu_bpe_tokenizer.json")  # if someone saved full filename

os.makedirs(ART_DIR, exist_ok=True)
if not os.path.exists(EXPECTED) and os.path.exists(LEGACY):
    try:
        os.replace(LEGACY, EXPECTED)  # rename legacy file to expected name
    except Exception:
        pass

# --- Load tokenizer -----------------------------------------------------------
tok = BPETokenizer.load(PREFIX)  # IMPORTANT: pass prefix, not full filename

# --- Load training stats (for model validation CR display) --------------------
MODEL_STATS = {"vocab_size": "n/a", "compression_ratio_val": "n/a"}
stats_path = os.path.join(ART_DIR, "stats.json")
try:
    with open(stats_path, "r", encoding="utf-8") as f:
        MODEL_STATS.update(json.load(f))
except Exception:
    pass

def tokenize(text: str):
    text = (text or "").strip()
    if not text:
        return "", 0, "0.0000", "", str(MODEL_STATS.get("compression_ratio_val", "n/a"))
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    n_tokens = len(ids)
    cr = (len(text) / n_tokens) if n_tokens else 0.0
    return str(ids), n_tokens, f"{cr:.4f}", decoded, str(MODEL_STATS.get("compression_ratio_val", "n/a"))

with gr.Blocks(title="Telugu (Indic) BPE Tokenizer") as demo:
    gr.Markdown(
        "### Telugu (or Indic) BPE Tokenizer\n"
        "Enter Telugu text to tokenize using the trained BPE model. "
        "You’ll see token IDs, token count, **per-sentence** compression ratio, "
        "and the model’s **validation** compression ratio from training."
    )

    inp = gr.Textbox(label="Input text", placeholder="ఉదాహరణకు: తెలుగు అందమైన భాష.")
    btn = gr.Button("Tokenize")

    ids_box = gr.Textbox(label="Token IDs", interactive=False)
    tok_count = gr.Number(label="Number of tokens", interactive=False)
    cr_box = gr.Textbox(label="Compression ratio (chars/tok)", interactive=False)
    dec_box = gr.Textbox(label="Decoded (reconstruction)", interactive=False)
    model_cr_box = gr.Textbox(label="Model validation compression ratio (val)", interactive=False)

    btn.click(tokenize, inputs=[inp],
              outputs=[ids_box, tok_count, cr_box, dec_box, model_cr_box])

demo.launch()
