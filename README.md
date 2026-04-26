# BiLSTM-Attention Machine Translation
### Lab Assignment 6 — Review, Implementation, and Comparative Analysis of Encoder–Decoder Models with and without Attention Mechanism

> **Paper:** Wu & Xing (2024). *Efficient Machine Translation with a BiLSTM-Attention Approach.* arXiv:2410.22335  
> **Course:** Deep Learning  
> **Dataset:** Tatoeba English-French (OPUS, v2023-04-12) — 500 filtered pairs  

## 👥 Group Members

| Name | Roll Number |
|---|---|
| Mohit Patil | 202301040272 |
| Parimal Ahire | 202301040067 |
| Rajveersinh Kher | 202301040233 |
| Atharva Suryawanshi | 202301040283 |

---

## 📁 Repository Structure

```
├── BiLSTM_Attention_Tatoeba.ipynb   # Main notebook (all 5 parts)
├── Full_Assignment_Report.docx      # Complete written report (Parts 1–5)
├── Encoder_Decoder_Outputs_code_sc.pdf  # Execution screenshots
├── README.md                        # This file
└── outputs/
    ├── bilstm_attention_results.png     # Loss curves + bar chart + attention heatmap
    └── attention_heatmaps_analysis.png  # Multi-sentence decoder alignment heatmaps
```

---

## 🧠 Architecture Overview

| Component | Detail |
|---|---|
| **Encoder** | BiLSTM with learnable initial states (h0, c0) |
| **Attention** | Bahdanau (Additive) — `score = v · tanh(W1·h + W2·s)` |
| **Decoder** | LSTM + Attention + 3-way linear projection |
| **Baseline** | Same BiLSTM encoder, but fixed mean-vector context (no attention) |

---

## 🚀 How to Run

### Google Colab
1. Open `BiLSTM_Attention_Tatoeba.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set Runtime → **GPU**
3. Click **Run All**

### Local
```bash
pip install torch matplotlib numpy
jupyter notebook BiLSTM_Attention_Tatoeba.ipynb
```

---

## 📊 Key Results

| Metric | With Attention | Without Attention |
|---|---|---|
| Final Train Loss | **0.5092** | 0.6195 |
| Final Val Loss | 5.4683 | 4.5422 |
| Val Perplexity | 237.05 | 93.90 |
| **Avg BLEU Score** | **0.4896 ✓** | 0.4479 |
| Training Time | 286.2 s | 190.1 s |
| Parameters | 787,811 | 604,451 |

> **Note on val loss anomaly:** The attention model has higher val loss due to overfitting on 500 pairs (30% more parameters). BLEU is the more reliable translation quality metric — attention wins there. This is expected and discussed in detail in Part 4 of the report.

---

## 🔍 What This Project Covers

- **Paper Review** — Problem statement, architecture breakdown, Bahdanau attention, WMT14 dataset, contributions and limitations
- **Code Walkthrough** — BiLSTMEncoder, BahdanauAttention, AttentionDecoder modules explained with training pipeline
- **Model Comparison** — Both models (with and without attention) trained and compared on loss, BLEU, training time, and translation quality
- **Result Analysis** — How attention improves alignment, explanation of the val loss anomaly, comparison with paper results
- **Conclusion** — Key findings, importance of the attention mechanism, and real-world applicability

---

## 🛠️ Hyperparameters

```python
EMB_DIM    = 64      # Embedding dimension
HID_DIM    = 128     # LSTM hidden size (encoder output = 256 due to BiLSTM)
N_LAYERS   = 1
DROPOUT    = 0.3
LR         = 0.001   # Adam optimizer
N_EPOCHS   = 30
CLIP       = 1.0     # Gradient clipping
TF_RATIO   = 0.5     # Teacher forcing ratio
MAX_PAIRS  = 500     # Dataset size cap
```

---

## 📖 References

- Wu, Y., & Xing, Y. (2024). Efficient Machine Translation with a BiLSTM-Attention Approach. [arXiv:2410.22335](https://arxiv.org/abs/2410.22335)
- Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. ICLR 2015.
- Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS 2017.
- Tiedemann, J. (2012). Parallel Data, Tools and Interfaces in OPUS. LREC 2012.
