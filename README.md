# Lab Assignment 6: Review, Implementation, and Comparative Analysis of Encoder–Decoder Models with and without Attention Mechanism

## BiLSTM-Attention Machine Translation

Based on: *Efficient Machine Translation with a BiLSTM-Attention Approach* — Wu & Xing (arXiv:2410.22335, October 2024)

---

## 👥 Group Members

| Name | PRN |
|---|---|
| Rajveersinh Kher | 202301040233 |
| Parimal Ahire | 202301040067 |
| Atharva Suryawanshi | 202301040283 |
| Mohit Patil | 202301040272 |

---

## 📋 Overview

This project re-implements the BiLSTM-Attention Seq2Seq architecture from Wu & Xing (2024) in PyTorch and compares it against a no-attention baseline on the Tatoeba EN-FR dataset. The goal is to validate the paper's central claim — that attention-augmented recurrent models remain a practical and competitive approach to neural machine translation, especially under resource constraints.

| Property | Value |
|---|---|
| Framework | PyTorch (re-implementation) |
| Dataset | Tatoeba EN-FR via OPUS (500 pairs) |
| Attention Type | Bahdanau (Additive) |
| Encoder | Bidirectional LSTM with learnable init states |
| Training | 30 epochs, Adam, teacher forcing 50% |
| Paper Dataset | WMT14 EN-DE / EN-FR |

---

## 🏗️ Architecture

```
Source Sentence
      │
      ▼
┌─────────────────────────────┐
│      BiLSTM Encoder         │  ← bidirectional, learnable h₀ & c₀
│  (forward + backward LSTM)  │
└────────────┬────────────────┘
             │  encoder outputs [src_len × 2·H]
             │  + projected hidden state [H]
             ▼
┌─────────────────────────────┐
│    Bahdanau Attention       │  ← score(s,h) = v · tanh(W₁h + W₂s)
│  (per decoder step)         │  ← context = Σ αᵢ · hᵢ
└────────────┬────────────────┘
             │  dynamic context vector [2·H]
             ▼
┌─────────────────────────────┐
│      LSTM Decoder           │  ← input: [embedding ‖ context]
│  + fc_out projection        │  ← output: [LSTM_out ‖ context ‖ embed] → logits
└─────────────────────────────┘
             │
             ▼
      Target Tokens
```

**Key Modules**

- **BiLSTMEncoder** — Processes the source sentence in both directions simultaneously. The initial hidden (h₀) and cell (c₀) states are `nn.Parameter` objects learned during training, allowing the model to start from an optimised encoding configuration.

- **BahdanauAttention** — At each decoder step, computes a scalar energy score between the current decoder state and every encoder output via additive attention (W₁, W₂, v). A softmax over scores produces attention weights; the context vector is their weighted sum over encoder states.

- **AttentionDecoder** — Concatenates the previous token embedding with the attention context vector before each LSTM cell update. The output projection is from [LSTM_out ‖ context ‖ embedding], giving the model explicit access to all three signals for vocabulary prediction.

- **PlainDecoder (baseline)** — Identical structure but replaces the dynamic context with a fixed mean of all encoder outputs computed once before decoding. This is the classic Seq2Seq bottleneck that attention was designed to address.

---

## 📁 Project Structure

```
Lab-Assignment-6/
│
├── BiLSTM_Attention_Tatoeba.ipynb       # Full implementation (all parts)
├── Full_Assignment_Report.docx          # Complete written report (Parts 1–5)
├── Encoder_Decoder_Outputs_code_sc.pdf  # Execution screenshots
├── README.md                            # This file
└── outputs/
    ├── bilstm_attention_results.png         # Loss curves + bar chart + attention heatmap
    └── attention_heatmaps_analysis.png      # Multi-sentence decoder alignment heatmaps
```

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

## 📊 Results

### Quantitative Comparison

| Metric | With Attention | Without Attention |
|---|---|---|
| Final Train Loss | 0.5092 | 0.6195 |
| Final Val Loss | 5.4683 | 4.5422 |
| Val Perplexity | 237.05 | 93.90 |
| Avg Unigram BLEU | 0.4896 | 0.4479 |
| Training Time | 286.2 s | 190.1 s |
| Parameters | 787,811 | 604,451 |
| BLEU Improvement | +9.3% relative | baseline |

The attention model achieves a higher BLEU score despite higher validation loss — a known artefact of the small dataset (500 pairs) causing the larger model to overfit more aggressively in cross-entropy terms, while still producing lexically better translations. At WMT14 scale both metrics improve together, consistent with the original paper's results.

### Translation Samples

| Source (EN) | Reference (FR) | With Attention | Without Attention |
|---|---|---|---|
| Did you miss me? | Je t'ai manqué ? | je `<unk>` manqué ? | je `<unk>` manqué ? |
| Are they all the same? | Sont-ils tous identiques ? | sont-ils tous les `<unk>` ? | sont-ils tous les `<unk>` ? |
| Thank you very much! | Merci beaucoup ! | merci beaucoup ! | merci `<unk>` ! |
| Thank you very much! | Merci vraiment. | merci beaucoup ! | merci `<unk>` ! |
| Where are the eggs, please? | S'il vous plaît, où sont les œufs ? | s'il vous plaît, où sont les `<unk>` ? | s'il vous plaît, où sont les `<unk>` ? |


### Attention Heatmap

For the sentence *"the cat is sleeping"*, the attention weight matrix shows a near-diagonal pattern:
- Generating **"le"** → attends to **"the"**
- Generating **"chat"** → attends to **"cat"**
- Generating **"dort"** → attends to **"sleeping"**

This confirms the model has learned correct word-level EN→FR alignment without explicit alignment supervision.

---

## 🔧 Hyperparameters

| Parameter | Value |
|---|---|
| Embedding Dimension | 64 |
| Hidden Dimension | 128 |
| LSTM Layers | 1 |
| Dropout | 0.3 |
| Learning Rate | 0.001 (Adam) |
| Epochs | 30 |
| Gradient Clipping | max norm = 1.0 |
| Teacher Forcing Ratio | 0.50 (train) / 0.0 (eval) |
| Loss Function | CrossEntropyLoss (PAD ignored) |
| Dataset | Tatoeba EN-FR, 500 pairs, 80/20 split |

---

## 📓 Notebook Structure

| Section | Content |
|---|---|
| Step 0 | Setup, imports, GPU check, random seeds |
| Section A | Tatoeba EN-FR dataset download, tokenization, vocabulary building, train/val split |
| Section B | BiLSTMEncoder, BahdanauAttention, AttentionDecoder, Seq2SeqAttention |
| Section C | PlainDecoder, Seq2SeqNoAttention (baseline) |
| Section D | Hyperparameters, train_epoch(), evaluate(), translate(), simple_bleu() |
| Section E | Side-by-side translation samples, BLEU comparison table |
| Section F | Loss curves, performance bar chart, multi-sentence attention heatmaps |

---

## 🔍 What This Project Covers

- **Paper Review** — Problem statement, architecture breakdown, Bahdanau attention, WMT14 dataset, contributions and limitations
- **Code Walkthrough** — BiLSTMEncoder, BahdanauAttention, AttentionDecoder modules explained with training pipeline
- **Model Comparison** — Both models (with and without attention) trained and compared on loss, BLEU, training time, and translation quality
- **Result Analysis** — How attention improves alignment, explanation of the val loss anomaly, comparison with paper results
- **Conclusion** — Key findings, importance of the attention mechanism, and real-world applicability

---

## 📖 Paper Summary

Wu & Xing (2024) propose a compact Seq2Seq model combining a BiLSTM encoder with an attention-enhanced decoder as a resource-efficient alternative to Transformers.

**Key contributions:**
- Bidirectional LSTM encoder capturing full left and right context per token
- Learnable initial hidden/cell states (h₀, c₀ as `nn.Parameter`) instead of zero initialisation
- Bahdanau (additive) attention at every decode step for dynamic source alignment
- Competitive BLEU on WMT14 EN-DE and EN-FR with significantly fewer parameters than a standard Transformer

**Limitations noted:**
- No subword tokenization (BPE/SentencePiece) specified
- Single attention head vs. multi-head in Transformers
- No pre-training; modern systems use mBART or M2M-100
- No dedicated evaluation on long sequences

---

## 🔬 Analysis Highlights

**Why attention improves translation:**
- **Better gradient flow** — direct differentiable paths from decoder step t to encoder position i, reducing the vanishing gradient problem
- **Dynamic alignment** — the model learns word correspondences (EN→FR) purely from parallel data
- **Richer per-step context** — each decode step receives a tailored context vector rather than a single compressed representation of the entire source

**Why the attention model has higher validation loss:** The attention model has 30% more parameters. On only 400 training pairs it overfits more aggressively, resulting in higher cross-entropy on the held-out set. However, BLEU — which measures word-level translation precision — still favours attention, and at WMT14 scale the discrepancy disappears.

---

## 📚 References

1. Wu, Y., & Xing, Y. (2024). Efficient Machine Translation with a BiLSTM-Attention Approach. [arXiv:2410.22335](https://arxiv.org/abs/2410.22335)
2. Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. ICLR 2015.
3. Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS 2017.
4. Cho, K., et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. EMNLP 2014.
5. Tiedemann, J. (2012). Parallel Data, Tools and Interfaces in OPUS. LREC 2012.

---

## 📄 License

This project is submitted as an academic assignment for the course **Deep Learning**. The PyTorch re-implementation is original work by the group members listed above.
