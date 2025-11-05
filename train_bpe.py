import os, json, glob, random, unicodedata
from bpe import BPETokenizer

DATA_DIR = "data"
ART_DIR = "artifacts"
PREFIX = os.path.join(ART_DIR, "telugu_bpe_tokenizer")  # save/load uses this *prefix*
README = "README.md"

os.makedirs(ART_DIR, exist_ok=True)

def read_texts():
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
    if not paths:
        raise FileNotFoundError(f"No .txt files found in {DATA_DIR}/")
    texts = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            t = f.read()
            texts.append(unicodedata.normalize("NFC", t))
    return texts

def split_corpus(texts, val_ratio=0.1, seed=42):
    random.Random(seed).shuffle(texts)
    if len(texts) <= 1:
        # with one file, use it for both train and val so training isn't empty
        return texts, texts
    n_val = max(1, int(len(texts) * val_ratio))
    return texts[n_val:], texts[:n_val]

def avg_compression_ratio(tokenizer, texts):
    total_chars, total_tokens = 0, 0
    for t in texts:
        ids = tokenizer.encode(t)
        total_chars += len(t)
        total_tokens += len(ids)
    return (total_chars / total_tokens) if total_tokens else 0.0

def _get_vocab_size(tok):
    # 1) method vocab_size()
    if hasattr(tok, "vocab_size") and callable(getattr(tok, "vocab_size")):
        try:
            return int(tok.vocab_size())
        except Exception:
            pass
    # 2) method get_vocab_size()
    if hasattr(tok, "get_vocab_size") and callable(getattr(tok, "get_vocab_size")):
        try:
            return int(tok.get_vocab_size())
        except Exception:
            pass
    # 3) common dicts
    for attr in ("token_to_id", "id_to_token", "vocab"):
        if hasattr(tok, attr):
            try:
                return int(len(getattr(tok, attr)))
            except Exception:
                pass
    raise RuntimeError("Could not determine vocab size from tokenizer.")

def update_readme(vocab_size, cr_val):
    header = "### Results (Auto-filled)"
    new_block = f"""{header}
- **Tokenizer vocabulary size**: `{vocab_size}`
- **Compression ratio (val)**: `{cr_val:.4f}`
"""
    body = ""
    if os.path.exists(README):
        with open(README, "r", encoding="utf-8") as f:
            body = f.read()
    if header in body:
        start = body.index(header)
        # replace until next markdown header if present
        end = len(body)
        for mark in ["\n## ", "\n# "]:
            idx = body.find(mark, start + 1)
            if idx != -1:
                end = min(end, idx)
        body = body[:start] + new_block + body[end:]
    else:
        body = body + "\n\n" + new_block
    with open(README, "w", encoding="utf-8") as f:
        f.write(body)

def main():
    texts = read_texts()

    # If you placed difficult demo lines in data/demo_sentences.txt they'll be included
    booster = os.path.join(DATA_DIR, "demo_sentences.txt")
    if os.path.exists(booster):
        with open(booster, "r", encoding="utf-8") as f:
            texts.append(unicodedata.normalize("NFC", f.read()))

    train_texts, val_texts = split_corpus(texts, val_ratio=0.1)

    tok = BPETokenizer()
    vocab_limit = 4000                      # stays < 5000 (assignment rule)
    tok.train(train_texts, vocab_size_limit=vocab_limit, min_pair_freq=1, progress=True)

    # Save tokenizer: will create artifacts/telugu_bpe_tokenizer_tokenizer.json
    tok.save(PREFIX)

    cr_val = avg_compression_ratio(tok, val_texts)
    vsize = _get_vocab_size(tok)

    # Persist stats for the app
    stats = {"vocab_size": vsize, "compression_ratio_val": cr_val}
    with open(os.path.join(ART_DIR, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # Update README block
    update_readme(vsize, cr_val)

    print("\n=== Training done ===")
    print(f"Vocab size: {vsize}")
    print(f"Validation compression ratio: {cr_val:.4f}")

if __name__ == "__main__":
    main()
