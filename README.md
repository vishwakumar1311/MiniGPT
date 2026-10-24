# MiniGPT

A small character-level GPT in PyTorch, trained on the "Tiny Shakespeare" text. Self-attention, multi-head attention, the feed-forward block, layer normalization and the embeddings are all written by hand with plain `nn.Linear` / `nn.Embedding` layers, without `nn.Transformer`, `nn.MultiheadAttention` or `nn.LayerNorm`. The overall structure follows Andrej Karpathy's "Let's build GPT" lecture.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vishwakumar1311/MiniGPT/blob/main/GPT.ipynb)

## Model

| | |
| --- | --- |
| Type | Decoder-only transformer, causal self-attention |
| Layers / heads | 4 blocks, 6 heads of size 15 |
| Embedding size | 90 |
| Context length | 128 characters |
| Parameters | 415,865 |
| Vocabulary | 65 characters |
| Regularization | Dropout 0.1 (attention weights, projections, feed-forward, embeddings) |
| Optimizer | AdamW, constant learning rate 1e-3, batch size 16, 11,000 steps |

Data: about 1.1M characters, split 90% train (1,003,854) and 10% validation (111,540). The text is `input.txt` in this repo; the notebook reads it from the working directory, or downloads it from this repo if it isn't there (for example on Colab).

## Results

Cross-entropy loss (nats per character), each value averaged over 200 random batches:

| Step | Train | Validation |
| --- | --- | --- |
| 0 | 4.192 | 4.195 |
| 1,000 | 2.097 | 2.148 |
| 5,000 | 1.468 | 1.647 |
| 11,000 | 1.356 | 1.557 |

The starting loss of 4.19 is close to `ln(65) = 4.17`, i.e. random guessing. Validation loss was still improving at the end, and it sits about 0.2 above training loss, so there is some overfitting; more data or stronger regularization would help.

A short sample of generated text (sampled from the model, starting from a newline):

```
CORIOLANUS:
The Either, long heaven,
That an one a power of this termand of now.

KING EDWARD IV:
But you did have come too provost?
```

The model has learned the play format, character names, punctuation and word shapes, but the text is not coherent. That is expected for a model this small.

## Run it

Use a GPU runtime on Colab (Runtime > Change runtime type) via the badge above. Locally:

```bash
pip install torch matplotlib jupyter
jupyter notebook GPT.ipynb
```

Training takes about 45 minutes on a 2-core CPU (roughly 250 ms per step) and a few minutes on a GPU. The weights are saved to `minigpt.pt` at the end of training.

## Limitations

- Character-level tokens, so the model has to learn spelling as well as language.
- Small model and short training run; no learning-rate schedule.
- Each attention head is a separate module (clear to read, slower than one batched attention layer).
