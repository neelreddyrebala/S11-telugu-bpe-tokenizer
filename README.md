---
title: S11 Telugu BPE Tokenizer
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
examples:
  - తెలుగు అందమైన భాష.
  - నాకు తెలుగు చదవడం చాలా ఇష్టం.
  - ఈ టోకనైజర్ బాగా పనిచేస్తోంది.
  - తెలుగు సినిమాలు ప్రపంచవ్యాప్తంగా ప్రేక్షకుల మనసులను గెలుచుకున్నాయి.
---

# 🇮🇳 Telugu (Indic) BPE Tokenizer — ERA Session 11 Assignment

Custom **Byte Pair Encoding (BPE)** tokenizer trained on Telugu text as part of **TSAI ERA V4 Session 11**.

### 📊 Model Results
| Metric | Value | Requirement | Status |
|---------|--------|-------------|---------|
| Vocabulary Size | 608 | < 5000 | ✅ |
| Compression Ratio (val) | 3.7875 | ≥ 3.2 | ✅ |

Trained using `train_bpe.py` on a ~1 MB Telugu corpus. Artifacts saved under `artifacts/`.

## 🧠 App Features
- Tokenizes Telugu input using the trained BPE model
- Displays token IDs, token count, **per-sentence** compression ratio,
  **model validation** compression ratio (average), and decoded text (`</w>` markers).

## 🚀 Try Examples
1. తెలుగు అందమైన భాష.
2. నాకు తెలుగు చదవడం చాలా ఇష్టం.
3. ఈ టోకనైజర్ బాగా పనిచేస్తోంది.
4. తెలుగు సినిమాలు ప్రపంచవ్యాప్తంగా ప్రేక్షకుల మనసులను గెలుచుకున్నాయి.

## 🧩 Repo Overview
| File | Purpose |
|------|----------|
| `train_bpe.py` | Trains BPE tokenizer & logs stats |
| `bpe.py` | Core BPE implementation |
| `app.py` | Gradio interface (this Space) |
| `requirements.txt` | Dependencies |
| `artifacts/` | Saved tokenizer + stats |

Built with ❤️ using **Python 3.13** and **Gradio 4.x**.


### Results (Auto-filled)
- **Tokenizer vocabulary size**: `681`
- **Compression ratio (val)**: `3.4837`
