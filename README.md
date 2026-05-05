# Automated-Text-Condensation

A seq2seq abstractive text summarization model built from scratch using the Transformer architecture in TensorFlow/Keras. Given a news article body, the model generates a concise headline.

---

## Overview

This project implements the full Transformer encoder-decoder architecture including multi-head attention, positional encoding, masking, and custom learning rate scheduling — trained on a news dataset to map article text (`Short`) to headlines (`Headline`).

Both **greedy decoding** and **beam search decoding** are supported at inference time.

---

## Dataset

- **File:** `data/news.xlsx`
- **Columns used:**
  - `Short` — the article body (encoder input)
  - `Headline` — the news headline (decoder target)
- Columns dropped: `Source`, `Time`, `Publish Date`

---

## Project Structure

```
summarization/
├── data/
│   └── news.xlsx
├── checkpoints/          # Saved model checkpoints
└── summarizer.ipynb      # Main notebook
```

---

## Model Architecture

The model is a standard Transformer encoder-decoder:

| Component | Details |
|---|---|
| Encoder layers | 4 |
| Decoder layers | 4 |
| Model dimension (`d_model`) | 256 |
| Feed-forward dimension (`dff`) | 1024 |
| Attention heads | 8 |
| Dropout rate | 0.1 |
| Positional encoding | Sinusoidal |

### Key Components

- **Positional Encoding**: sinusoidal encoding to inject word position information (since the Transformer is non-directional unlike RNNs)
- **Scaled Dot-Product Attention**: core attention mechanism scaled by √dk
- **Multi-Head Attention**: parallel attention across 8 heads
- **Encoder Layer**: self-attention + feed-forward + layer norm + dropout
- **Decoder Layer**: masked self-attention + cross-attention + feed-forward + layer norm + dropout
- **Padding Mask**: ignores padding tokens during attention
- **Look-Ahead Mask**: prevents the decoder from attending to future tokens

---

## Training

- **Optimizer:** Adam with custom warmup-based learning rate schedule
- **Loss:** Sparse Categorical Crossentropy (masked to ignore padding)
- **Epochs:** Up to 10 with early stopping (patience = 5)
- **Batch size:** 64
- **Checkpointing:** Best model checkpoint saved automatically

---

## Inference

Two decoding strategies are available:

### Greedy Decoding
```python
summarize("Your article text here.")
```
Predicts one token at a time by always picking the highest-probability next token.

### Beam Search Decoding
```python
summarize_beam("Your article text here.", beam_width=4)
```
Maintains the top-k (beam_width=4) most probable sequences at each step and picks the best completed sequence using length-normalized scoring.

---

## Requirements

```
tensorflow
pandas
numpy
openpyxl       # for reading .xlsx files
```

Run in **Google Colab** with Google Drive mounted:

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
%cd /content/drive/MyDrive/summarization
```

---

## Example Usage

```python
summarize(
    "US-based private equity firm General Atlantic is in talks to invest about "
    "$850 million to $950 million in Reliance Industries' digital unit Jio Platforms."
)

summarize_beam(
    "The Indian government announced a new policy aimed at boosting electric "
    "vehicle adoption by providing subsidies and expanding charging infrastructure."
)
```

---

## Notes

- Sequence lengths are set to the **90th percentile** of the training data distribution to balance coverage and padding overhead.
- Decoder inputs are wrapped with `<go>` and `<stop>` special tokens during training.
- Separate tokenizers are used for the encoder (document) and decoder (summary) vocabularies.
- Beam search uses length normalization to avoid bias toward shorter sequences.

---

## Contributers
- Sindhu Kumari

  Enrollment Number- SBU220013
- Aksha Perween

  Enrollment Number- SBU220211
- Anshu Kumari

  Enrollment Number- SBU220257
- Asra Yasmeen

  Enrollment Number- SBU220335

---
