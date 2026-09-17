# Cross-modal models: shortlist and evidence

Research for [issue #16](https://github.com/dangth1101/IS6303/issues/16), under map [#13](https://github.com/dangth1101/IS6303/issues/13).
Compiled 2026-09-17. Every number below is cited to a primary source (model card, config,
paper, or the model's own inference code).

**Scope.** We need a model that embeds a query image and Recipe **Retrieval Text** into one
shared space, so that **Cross-modal Retrieval** matches an image against text. Recipe images
are never embedded (premise 3 of #13). Embedding inference is local (premise 8), so
API-only models are disqualified by construction.

---

## 1. Answer to the decisive question

**Yes — one candidate escapes the 77-token ceiling, but not by as much as its marketing says.**

| Model | Text tokens usable for **Cross-modal Retrieval** | Evidence |
|---|---|---|
| OpenAI CLIP ViT-B/32, ViT-L/14 | **77** (75 content + `[SOS]`/`[EOS]`) | `max_position_embeddings: 77` in [config.json](https://huggingface.co/openai/clip-vit-large-patch14/raw/main/config.json); CLIP paper §2.4: *"For computational efficiency, the max sequence length was capped at 76."* ([arXiv:2103.00020](https://arxiv.org/html/2103.00020v1)) |
| OpenCLIP (all ViT variants) | **77** | `"context_length": 77` in every [model config](https://raw.githubusercontent.com/mlfoundations/open_clip/main/src/open_clip/model_configs/ViT-B-32.json) (B/32, L/14, H/14 checked) |
| SigLIP / SigLIP 2 (all sizes) | **64** — *worse than CLIP* | `SiglipTextConfig.max_position_embeddings` defaults to `64`; `SiglipTokenizer.model_max_length = 64` ([HF docs](https://huggingface.co/docs/transformers/en/model_doc/siglip)). SigLIP 2 paper: *"We set the text length to 64 and use the multilingual Gemma tokenizer with vocabulary size 256k."* ([arXiv:2502.14786](https://arxiv.org/html/2502.14786v1)) |
| **jina-clip-v2** | **512** (advertised 8,192) | See §2 |
| **jina-clip-v1** | **512** (advertised 8,192) | Same architecture lineage; see §2 |
| jina-embeddings-v4 | 32,768 | [Model card](https://huggingface.co/jinaai/jina-embeddings-v4) — but 4B params, see §4 |
| Long-CLIP | 248 | [arXiv:2403.15378](https://arxiv.org/abs/2403.15378) — research checkpoint, see §4 |

### Why this matters for a Chunk

The CLIP budget of ~75 content tokens is roughly **300 characters** of English. Map #13's
verified corpus facts put `directions` at an **average of 608 characters (~150 tokens), max
4,263**. So a field **Chunk** over `directions` is on average **2× over the CLIP budget**, and
an `ingredients` Chunk — a `;`-delimited list, where every item is a content word and BPE
compresses poorly — is exactly the case the ticket worried about. Under CLIP or SigLIP the
back half of an ingredient list is simply **discarded before it is ever embedded**, silently.

At 512 tokens jina-clip-v2 clears the average `directions` field with ~3.4× headroom and
clears all but the extreme tail (4,263 chars ≈ 1,000 tokens) of the corpus.

---

## 2. Verifying jina-clip-v2's long context — and its shared space

The ticket asked two things: is the 8,192 real, and is the long-context text encoder still
aligned with the vision encoder? Both were checked against primary sources.

### 2a. The 8,192 figure is an architectural ceiling, not a cross-modal one

The model card does advertise it — *"Input Specification | 8,192 tokens (max) | 512×512 pixels"*
([model card](https://huggingface.co/jinaai/jina-clip-v2)), and `tokenizer_config.json` carries
`"model_max_length": 8194`.

But three independent primary sources cap the **useful** cross-modal length at **512**:

1. **The training recipe.** The jina-clip-v2 technical report
   ([arXiv:2412.08802](https://arxiv.org/html/2412.08802v1)) trains in three stages:
   - Stage 1 — text pairs and **short image captions**, context length **77**
   - Stage 2 — text pairs and **long image captions**, context length **512**
   - Stage 3 — hard negatives, context length **512**

   There is **no training stage in which image-text alignment sees text longer than 512
   tokens.** Beyond 512 the text tower is extrapolating with no cross-modal supervision.

2. **The model's own inference code.** In
   [`modeling_clip.py`](https://huggingface.co/jinaai/jina-clip-implementation/raw/main/modeling_clip.py),
   `encode_text` sets:
   ```python
   tokenizer_kwargs['max_length'] = tokenizer_kwargs.get('max_length', 512)
   tokenizer_kwargs['truncation'] = tokenizer_kwargs.get('truncation', True)
   ```
   Jina's own default **truncates at 512**. You would have to override this deliberately to
   feed more, and you would be leaving the trained regime if you did.

3. **jina-clip-v1 is the same story.** Its paper
   ([arXiv:2405.20204](https://arxiv.org/html/2405.20204v1)) uses 77 tokens in stage 1 and
   truncates to **512** in stages 2 and 3 (long synthetic ShareGPT4V captions).

**Verdict: treat jina-clip-v2's cross-modal text limit as 512 tokens, not 8,192.** That is
still a **6.6× improvement over CLIP** and **8× over SigLIP**, and it is enough for our Chunks.
Any claim in an ADR should say 512, not 8,192.

### 2b. The long-context text encoder *does* share the vision encoder's space

This was the real risk — a strong text tower bolted on but not aligned. The evidence says it
is aligned:

- **One projection dimension, no separate heads.** `config.json` has `"projection_dim": 1024`,
  `"add_projections": false`, and **both** `text_config.embed_dim` and
  `vision_config.embed_dim` are `1024`. Both towers write into the same 1024-d space with no
  intervening projection ([config.json](https://huggingface.co/jinaai/jina-clip-v2/raw/main/config.json)).
- **Contrastively trained against images at 512 tokens.** Stages 2 and 3 are image-text
  contrastive stages run at 512-token context (above). The alignment is not inherited from a
  77-token stage and then abandoned.
- **Cross-modal scores improve as context grows.** In the paper's own ablation, Flickr30K
  image→text r@5 rises 86.61 (stage 1, 77 tokens) → 89.57 (stage 2, 512 tokens), and MS COCO
  image→text r@5 rises 77.12 → 81.74. Lengthening the cross-modal context *helped*; it did not
  degrade alignment.
- **Matryoshka applies to both modalities.** The card states truncation "of both text and image
  embeddings from 1024 down to 64" — they are the same kind of vector.

**Verdict: yes, shared space, up to 512 tokens.** This is the finding that unblocks the ticket.

---

## 3. Candidate comparison

| Model | Embed dim | Params | Text tokens | Licence |
|---|---|---|---|---|
| CLIP ViT-B/32 | 512 | ≈151M | 77 | MIT ([repo LICENSE](https://github.com/openai/CLIP/blob/main/LICENSE)) — but model card says *"Any deployed use case of the model—whether commercial or not—is currently out of scope"* |
| CLIP ViT-L/14 | 768 | 427.6M | 77 | as above |
| OpenCLIP ViT-B/32 | 512 | — | 77 | MIT (repo) |
| OpenCLIP ViT-L/14 | 768 | — | 77 | MIT |
| OpenCLIP ViT-H/14 | 1024 | — | 77 | MIT |
| SigLIP 2 base/16-256 | 768 | 375.2M | 64 | **apache-2.0** ([card front matter](https://huggingface.co/google/siglip2-base-patch16-256)) |
| SigLIP 2 so400m/14-384 | 1152 | 1.136B | 64 | **apache-2.0** |
| **jina-clip-v1** | 768 | **222.7M** | **512** | **apache-2.0** ([card](https://huggingface.co/jinaai/jina-clip-v1)) |
| **jina-clip-v2** | 1024 (Matryoshka 32–1024) | **865.3M** | **512** | **cc-by-nc-4.0** ([card](https://huggingface.co/jinaai/jina-clip-v2)) |
| jina-embeddings-v4 | 2048 single-vector / 128 multi | ~4B | 32,768 | Qwen Research Licence (non-commercial) |

Param counts are from the Hugging Face API `safetensors.total` field for each repo
(`https://huggingface.co/api/models/<id>?expand[]=safetensors`). CLIP ViT-B/32 does not expose
one; ≈151M is the figure from the CLIP release.

**Licence notes worth flagging to the dev:**
- **jina-clip-v2 is CC-BY-NC-4.0 — non-commercial.** Fine for a graded course project; it would
  block any commercial reuse. There is also a **discrepancy**: the model card front matter says
  `cc-by-nc-4.0` while the technical report states `CC BY-NC-SA 4.0`. Cite the model card.
- **jina-clip-v1 is Apache-2.0** and is the only long-context candidate with a permissive licence.
- **SigLIP 2 checkpoints are Apache-2.0** — the cleanest licence of any strong model here, but
  they have the worst token limit.
- OpenAI CLIP's MIT repo licence and its model card's "out of scope for any deployed use case"
  language are in tension. Research/coursework use is squarely in the card's stated intent.

### Zero-shot cross-modal retrieval evidence

From SigLIP 2 paper Table 1, *"Zero-shot classification, 10-shot (10s) classification (on the
validation set), and retrieval performance (recall@1)"* ([arXiv:2502.14786](https://arxiv.org/html/2502.14786v1)).
Column order in the source is COCO T→I, COCO I→T, Flickr T→I, Flickr I→T:

| Model | COCO T→I r@1 | COCO I→T r@1 | Flickr T→I r@1 | Flickr I→T r@1 |
|---|---|---|---|---|
| CLIP B/16 (224) | 33.1 | 52.4 | 62.1 | 81.9 |
| CLIP L/14 (224) | 36.5 | 56.3 | 65.2 | 85.2 |
| OpenCLIP B/32 (256) | 39.9 | 57.9 | 64.9 | 84.8 |
| OpenCLIP L/14 (224) | 46.1 | 62.1 | 75.0 | 88.7 |
| SigLIP B/16 (256) | 47.4 | 65.1 | 78.3 | 91.1 |
| **SigLIP 2 B/16 (256)** | **53.2** | **69.7** | **81.7** | **94.4** |
| **SigLIP 2 L/16 (256)** | **54.7** | **71.5** | **84.1** | **94.5** |
| **SigLIP 2 So/14 (384)** | **55.8** | **71.7** | **85.7** | **94.9** |
| SigLIP 2 g/16 (384) | 56.1 | 72.8 | 86.0 | 95.4 |

**SigLIP 2 is the strongest pure cross-modal retriever by a wide margin** — SigLIP 2 B/16 beats
CLIP L/14 by ~18 points on COCO T→I while being smaller. Its abstract claims this outright:
*"SigLIP 2 models outperform their SigLIP counterparts at all model scales in core capabilities,
including zero-shot classification, image-text retrieval, and transfer performance."*

jina-clip-v2's own numbers are **recall@5, not recall@1**, so they are **not directly comparable**
to the table above ([arXiv:2412.08802](https://arxiv.org/html/2412.08802v1) Table 2):

| Model | Flickr30K T→I r@5 | Flickr30K I→T r@5 | MS COCO T→I r@5 | MS COCO I→T r@5 |
|---|---|---|---|---|
| jina-clip-v1 | 77.75 | 87.65 | — | — |
| jina-clip-v2 stage 1 (77 tok) | 73.87 | 86.61 | 60.91 | 77.12 |
| jina-clip-v2 stage 2 (512 tok) | 79.81 | 89.57 | 69.59 | 81.74 |
| **jina-clip-v2 (final)** | **79.09** | **89.73** | **68.35** | **81.46** |
| NLLB-CLIP-SigLIP Large | 81.54 | 88.15 | 70.84 | 79.20 |

**Caveat that matters for us:** every one of these benchmarks is a **short-caption** benchmark.
COCO and Flickr30k captions are one sentence — well inside 77 tokens. **No published benchmark
measures the thing we actually care about: matching an image query against a 200–500 token
Recipe Chunk.** The benchmark ranking (SigLIP 2 wins) and the structural argument (only jina
can see the whole Chunk) point in *opposite* directions, and no external number resolves it.
This is a strong argument for measuring it ourselves on our own Qrels — which premise 9 of #13
already commits us to.

### CPU inference speed

**Establishing this from primary sources failed, and I want to be explicit about that rather
than quote a blog.** None of OpenAI, LAION/OpenCLIP, Google or Jina publishes CPU throughput
figures for these checkpoints. Apple's MobileCLIP family publishes latency, but on mobile
silicon, not desktop CPU.

What can be said from primary data:

- Parameter count is the usable proxy, and it spans **~6×**: CLIP ViT-B/32 ≈151M →
  jina-clip-v2 865M → jina-embeddings-v4 ~4B.
- **The text tower cost scales with the token budget, and this cuts against jina.** Encoding a
  512-token Chunk costs substantially more attention work than a 77-token one, and every Chunk
  in the Corpus must be encoded once at ingest. jina-clip-v2's text encoder is also the larger
  half of the model (561M of 865M, per the technical report) — the opposite of CLIP, where the
  text tower is 63M.
- Practical consequence: **jina-clip-v2 is roughly an order of magnitude more expensive per
  Chunk at ingest than CLIP ViT-B/32**, combining ~6× the parameters with ~6.6× the tokens.

**Recommendation: measure this locally before the comparison matrix is sized.** A single timed
pass over a few hundred Chunks per candidate on the target machine would settle it, and #13
flags that "the real cost of an ingest pass" is exactly what the matrix is waiting on.

---

## 4. Other current alternatives

- **jina-embeddings-v4** ([card](https://huggingface.co/jinaai/jina-embeddings-v4)) — 32,768
  tokens, unified text/image space on a Qwen2.5-VL-3B backbone, 2048-d single-vector (Matryoshka
  to 128) plus a ColBERT-style 128-d multi-vector mode, ~4B params. Genuinely solves the token
  problem outright. **Disqualified on cost**: a 4B-parameter model doing local CPU inference
  over 32,700 Recipes × N Chunks × multiple Chunking Strategies is not a course-project ingest
  budget. Licence is the Qwen Research Licence (non-commercial); the card notes an initial
  `cc-by-nc-4.0` designation was corrected.
- **GME-Qwen2-VL-2B/7B** (Alibaba, Apache-2.0) — universal multimodal embeddings, 1536-d,
  any-to-any retrieval. Permissively licensed and long-context, but 2.2B+ params puts it in the
  same cost bracket as jina-v4. Worth a line in an ADR as "considered, rejected on cost".
- **Long-CLIP** ([arXiv:2403.15378](https://arxiv.org/abs/2403.15378), ECCV 2024) — stretches
  CLIP's positional embeddings from 77 to **248** tokens and reports **+20% r@5 on long-caption
  text-image retrieval**. This is direct third-party evidence that *the 77-token limit is itself
  costing retrieval quality on long text*. It is a research checkpoint, not a maintained model
  card, so it is a supporting citation rather than a shortlist entry.
- **MobileCLIP / MobileCLIP2** (Apple) — the speed play, 3–15ms latency at 50–150M params, but
  it is CLIP-architecture-compatible and therefore **inherits the 77-token limit**. No help here.
- **voyage-multimodal-3** and other hosted embedding APIs — **disqualified by premise 8** of #13
  (embedding inference is local, for reproducibility).

---

## 5. Second question: can one model serve both Arms?

**The fact, established:** jina-clip-v2's text encoder is a genuinely competent text retriever —
far better than any CLIP-family text tower — but it is **measurably worse than a dedicated text
embedder**.

From the jina-clip-v2 technical report Table 3, English text-only
([arXiv:2412.08802](https://arxiv.org/html/2412.08802v1)):

| Model | MTEB Retrieval nDCG@10 | STS |
|---|---|---|
| NLLB-CLIP-SigLIP Large *(a CLIP-family text tower)* | **24.91** | 74.89 |
| jina-clip-v1 | 48.33 | 80.92 |
| jina-clip-v2 stage 1 | 40.91 | 79.67 |
| jina-clip-v2 stage 2 | 43.17 | 80.33 |
| **jina-clip-v2 (final)** | **49.32** | 81.29 |
| jina-embeddings-v3 *(dedicated text embedder)* | **53.87** | 85.80 |

Three things follow:

1. **CLIP-family text encoders are unusable as a text Arm.** 24.91 nDCG@10 is not a retrieval
   system. The jina-clip-v1 card says so plainly: models like `openai/clip-vit-base-patch32`
   *"effectively align image and text embeddings but are not optimized for text-to-text retrieval
   due to their training methodologies and context limitations."* If we pick CLIP or SigLIP 2 for
   Cross-modal Retrieval, **two models is not a choice, it is a requirement.**
2. **jina-clip-v2 is the only candidate where one model for both Arms is even on the table.**
   49.32 vs 53.87 is a **~4.5 point / ~8% relative** deficit against jina's own dedicated text
   model. That is a real cost, not a rounding error, but it is the cost of a viable system rather
   than a broken one.
3. **The architecture explains why.** jina-clip-v2's text tower *is* jina-embeddings-v3 —
   `config.json` shows `text_config.hf_model_name_or_path: "jinaai/jina-embeddings-v3"` with a
   `retrieval.query` LoRA adapter and mean pooling. The 4.5-point gap is the price of
   contrastively aligning that encoder to images.

**The decision is not mine to make** (per the ticket), but the fact is clean: *only jina-clip-v2
(or v1) makes a one-model, two-Arm design possible at all, and it costs ~8% relative MTEB
retrieval against a dedicated text embedder.* Note also that jina-clip-v2's text tower is
multilingual and scores higher on multilingual MTEB retrieval (69.86%) than English (49.33%) —
irrelevant for an English allrecipes.com Corpus, and it means we pay for capacity we do not use.

---

## 6. Summary for the shortlist

| | Cross-modal quality | Token budget | Size / CPU cost | Licence | One-model-both-Arms |
|---|---|---|---|---|---|
| **SigLIP 2 (base or so400m)** | **Best measured** | **Worst (64)** | Good at base (375M) | **Best (Apache-2.0)** | No |
| **jina-clip-v2** | Good (r@5 only; not directly comparable) | **Best usable (512)** | Worst of the three (865M, big text tower) | Non-commercial | **Only candidate** |
| **jina-clip-v1** | Below v2 | **512** | **Best of the long-context options (223M)** | **Apache-2.0** | Yes, slightly below v2 |
| CLIP / OpenCLIP | Weakest | 77 | Cheapest | MIT | No |

The shortlist writes itself as a **three-way** comparison, and it is a genuine trade-off rather
than a winner:

- **SigLIP 2 base/16-256** — the benchmark leader, Apache-2.0, but it can only see ~64 tokens of
  a Chunk and forces a second model for the text Arm.
- **jina-clip-v2** — the only model that can read a whole ingredient list *and* the only one that
  could collapse both Arms into one model; costs a non-commercial licence and the heaviest ingest.
- **jina-clip-v1** — the underrated middle: same 512-token cross-modal budget, Apache-2.0, and
  **less than half the parameters** of any other long-context option. If ingest cost or licence
  bites, this is the fallback that keeps the token-limit win.

**The one thing no source can settle:** whether SigLIP 2's benchmark lead survives being fed a
truncated Chunk. That is precisely what our own Qrels and Pool are for.

---

## Sources

- CLIP paper — https://arxiv.org/html/2103.00020v1
- CLIP ViT-L/14 config — https://huggingface.co/openai/clip-vit-large-patch14/raw/main/config.json
- CLIP ViT-B/32 card — https://huggingface.co/openai/clip-vit-base-patch32
- CLIP repo licence — https://github.com/openai/CLIP/blob/main/LICENSE
- OpenCLIP repo + model configs — https://github.com/mlfoundations/open_clip
- SigLIP 2 paper — https://arxiv.org/html/2502.14786v1
- SigLIP 2 so400m card — https://huggingface.co/google/siglip2-so400m-patch14-384
- SigLIP HF docs (`SiglipTextConfig`, `SiglipTokenizer`) — https://huggingface.co/docs/transformers/en/model_doc/siglip
- jina-clip-v2 card — https://huggingface.co/jinaai/jina-clip-v2
- jina-clip-v2 config — https://huggingface.co/jinaai/jina-clip-v2/raw/main/config.json
- jina-clip-v2 technical report — https://arxiv.org/html/2412.08802v1
- jina-clip implementation (`encode_text`) — https://huggingface.co/jinaai/jina-clip-implementation/raw/main/modeling_clip.py
- jina-clip-v1 card — https://huggingface.co/jinaai/jina-clip-v1
- jina-clip-v1 paper — https://arxiv.org/html/2405.20204v1
- jina-embeddings-v4 card — https://huggingface.co/jinaai/jina-embeddings-v4
- Long-CLIP — https://arxiv.org/abs/2403.15378
- GME (Qwen2-VL) — https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-7B-Instruct
- Parameter counts — `https://huggingface.co/api/models/<id>?expand[]=safetensors`
