#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gradio as gr
from pathlib import Path
from bpe import BPETokenizer, compression_ratio, normalize

ARTIFACTS = Path("artifacts")
MODEL_PREFIX = ARTIFACTS / "telugu_bpe"

def load_tokenizer():
    try:
        tok = BPETokenizer.load(str(MODEL_PREFIX))
        return tok
    except Exception as e:
        return None

tok = load_tokenizer()

def tokenize(text):
    if not text.strip():
        return "", 0, 0.0, "Please enter some text."
    if tok is None:
        return "", 0, 0.0, "Tokenizer not found. Please train first (run train_bpe.py) and push artifacts."
    text = normalize(text)
    ids = tok.encode(text)
    recon = tok.decode(ids)
    cr = compression_ratio(text, ids)
    return str(ids), len(ids), float(f"{cr:.4f}"), recon

with gr.Blocks() as demo:
    gr.Markdown("# Telugu (or Indic) BPE Tokenizer")
    gr.Markdown(
        "Enter text in your chosen language. The app will tokenize with the trained BPE, "
        "report token count and compression ratio, and reconstruct the text."
    )
    inp = gr.Textbox(label="Input text", lines=4, placeholder="తెలుగు లో వాక్యాలు ఇక్కడ ఇవ్వండి... (or use Hindi, etc.)")
    btn = gr.Button("Tokenize")
    out_ids = gr.Textbox(label="Token IDs")
    out_len = gr.Number(label="Number of tokens", precision=0)
    out_cr = gr.Number(label="Compression ratio (chars/tok)")
    out_recon = gr.Textbox(label="Decoded (reconstruction)")

    btn.click(tokenize, inputs=inp, outputs=[out_ids, out_len, out_cr, out_recon])

if __name__ == "__main__":
    demo.launch()
