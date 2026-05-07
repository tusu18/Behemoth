<div align="center">

# 🦣 Behemoth-70M

**Semantic Memory Routing in Sub-100M Neural Long-Term Memory**

[[arXiv](Coming Soon)
[[HuggingFace](Coming soon)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)


*70M parameters · 1.42B training tokens · 141 MB in BF16*

</div>

---

Titans (Behrouz et al., NeurIPS 2025) showed that an MLP updated via gradient
descent at inference time can store and retrieve specific facts across arbitrarily
long contexts. The idea is elegant: instead of a fixed recurrent state or a growing
KV cache, the model's memory weights themselves are the long-term storage.

We built Behemoth-70M to study what this looks like at 70M parameters
a scale where most long-context approaches break down and found two things
we did not go in expecting.

---

## The two findings

### Routing by meaning, not by token

Standard analyses of Mixture-of-Experts routing in decoder models find that
experts specialise by token identity: the word *the* routes to the same expert
whether it appears in a legal clause or a fairy tale. The routing decision happens
on local surface features, not on what the sentence means.

In Behemoth, the pattern is different. One expert handles 96.4% of technical
passages and 78% of factual passages. A different expert handles 77.8% of dialogue.
No auxiliary routing objective was used this emerged from pretraining alone.

The mechanism appears to be the memory update itself. When an expert is updated
repeatedly on factual retrieval patterns, it gets better at factual retrieval.
The router then directs more factual content to the expert that performs best on it.
Standard feedforward MoEs have no equivalent pressure.

```
Content type    Dominant expert   Concentration
──────────────────────────────────────────────
Technical       Expert 3          96.4%
Factual         Expert 3          78.0%
Dialogue        Expert 1          77.8%
Procedural      Expert 3          72.1%
Narrative       Expert 3          42.2%

Uniform baseline: 25% per expert. No auxiliary routing loss.
```

### Specialisation only at the final layer

The routing pattern described above does not exist in layers 0–6.
Those layers show near-random routing — consistent with the token-level
pattern from the large-model literature. Sharp semantic concentration
appears only at layer 7, the final layer.

```
Layer 0:  E1 53%  E2 36%  E3 12%  E4  0%  ← random
Layer 3:  E1 27%  E2 10%  E3 25%  E4 37%  ← mixed
Layer 6:  E1 12%  E2 29%  E3 31%  E4 29%  ← converging
Layer 7:  E1  2%  E2  0%  E3 78%  E4 20%  ← sharp
```

The plausible explanation is that layer 7 receives fully integrated
semantic representations, while earlier layers work with syntactic and
local features that are not informative enough for content-type routing.
Whether this is a property of neural memory architectures specifically,
or a more general feature of final-layer representations at small scale,
we cannot say from a single architecture.

---

## BABILong results

Zero-shot evaluation on BABILong (Kuratov et al. 2024), QA1+QA2,
perplexity scoring with real answer-vocabulary distractors.

| Model | Params | Data | BL-1K | BL-2K | BL-4K | BL-8K |
|---|---|---|---|---|---|---|
| GPT-2-124M | 124M | 300B | 13% | 8% | 5% | — |
| Mamba-130M | 130M | 300B | 24% | 24% | 12% | — |
| Titans-760M | 760M | 300B | 54% | 52% | 49% | 38% |
| Behemoth-70M v1 | 70M | 0.92B | 20% | 26% | 20% | — |
| **Behemoth-70M v2** | **70M** | **1.42B** | **28%** | **26%** | **39%** | **27%** |

BL-8K at 27% is the first published zero-shot result for any sub-100M
parameter model at this context length. The column was previously empty
because learned absolute position embeddings crash on sequences longer
than their training length. Replacing them with RoPE removed this ceiling —
and accounts for most of the improvement between v1 and v2:

```
v1  SEQ=1024  absolute embeddings   BL-4K: 20%
v2  SEQ=4096  RoPE                  BL-4K: 39%   +19pp
```

On data efficiency: reaching 39% at 4K on 1.42B tokens versus Mamba's
12% on 300B tokens gives a score ratio of 3.25× and an
accuracy-per-training-token ratio of 688×.

---

## Architecture

Behemoth is a transformer variant that extends Titans with components
drawn from how memory actually consolidates in multi-level systems.
The key design choices:

**Memory is separated by content type.** Four expert MLPs replace the
single memory MLP in Titans. Only the top-2 experts per token are updated
at inference time. This prevents factual memories from being overwritten
by narrative content and vice versa.

**Surprise operates at three scales.** The signal that drives memory writes
combines immediate token-level prediction error, sentence-level EMA divergence,
and document-level KL divergence. The original Titans signal is token-only —
this misses sentence and document novelty, under-writing important facts that
appear unsurprising at the word level.

**Chunk boundaries respect semantic units.** Attention entropy identifies
positions where the model is confident about local context — natural
processing unit boundaries. Memory writes happen at these positions rather
than at fixed token counts.

**Keys co-evolve with memory.** Low-rank adapters (rank 8) on the query
and key projections allow the backbone's key space to stay aligned with
the memory's evolving key space during inference. Without this, the backbone
drifts from the memory after many update steps.

**Consolidation transfers across tiers.** At document boundaries, important
content from the bounded attention cache is scored, written to the
best-routing expert, and the result is propagated to persistent memory slots.
This is a discrete three-tier instantiation of the Continuum Memory System
from Nested Learning (Behrouz et al. 2025).

**RoPE for position.** Parameter-free rotary embeddings replace learned
absolute position embeddings. Applied after LoRA's residual addition so
inference-time key adaptation and positional encoding do not interfere.

```
d_model    512        seq_len    4096
n_layers   8          window     1024
n_heads    8          persistent 16 slots
n_experts  4 top-2    lora_rank  8
params     70.28M     bf16 size  141 MB
```

---

## Usage

```python
from behemoth import load_pretrained

model, tokenizer = load_pretrained("tsingh-umd/Behemoth-70M")
model.eval()

# Perplexity-based QA — score correct answer vs distractors
def score(context, completion):
    text = context + " " + completion
    ids  = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        # update=False is required — without it memory gradient
        # descent runs on every forward pass, ~200x slower
        _, loss = model(ids, labels=ids, update=False)
    return loss.item()

candidates = ["kitchen", "bedroom", "garden", "hallway"]
answer = min(candidates, key=lambda c: score(long_document, c))
```

---

## Reproducing the results

```bash
git clone https://github.com/tsingh-umd/behemoth
cd behemoth
pip install -e .
```

```python
# BABILong evaluation
from behemoth.eval import eval_babilong

results = eval_babilong(
    model, tokenizer,
    configs=["1k", "2k", "4k", "8k"],
    tasks=["qa1", "qa2"],
    n=100,
    front_ratio=0.60,   # 60% front + 40% back — facts planted early
    reset_memory=True,  # reset state between examples
)
# returns {"1k": 28.0, "2k": 26.0, "4k": 39.0, "8k": 27.0}
```

```bash
# Training from scratch — single L4 GPU, ~55 hours
python train.py --config configs/behemoth_70m.yaml
```

Training notebook is in `notebooks/Training.ipynb`.

---

## Repository layout

```
behemoth/
├── behemoth/
│   ├── model.py          # full architecture
│   ├── memory.py         # MoE memory, consolidator, surprise
│   ├── attention.py      # local window attention with LoRA + RoPE
│   └── eval.py           # BABILong evaluation utilities
├── notebooks/
│   ├── Training.ipynb
│   ├── Ablation.ipynb
│   └── Evaluation.ipynb
├── configs/
│   └── behemoth_70m.yaml
├── paper/
│   ├── behemoth.pdf
│   └── behemoth.tex
└── README.md
```

---

## Limitations

The routing findings are from one architecture trained on one dataset.
Whether they generalise is unknown. 70M parameters and 1.42B tokens is
small for drawing strong conclusions about MoE routing in general.
Multi hop BABILong tasks (QA3–QA5) are not solved they require
fact chaining that base pretrained models at this scale cannot do.
Training data is English-only.

---

## Citation

```bibtex
@article{singh2026behemoth,
  title   = {Behemoth-70M: What Happens When You Actually Fix
             Neural Long-Term Memory at Small Scale},
  author  = {Singh, Tushar},
  year    = {2026},
  url     = {https://arxiv.org/abs/2026.XXXXX}
}
```


---

<div align="center">
<sub>University of Maryland, College Park · 2026</sub>
</div>
