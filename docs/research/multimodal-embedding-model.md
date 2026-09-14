# Research: selecting the local multimodal embedding model

Resolves [#2](https://github.com/dangth1101/IS6303/issues/2). Part of the Milestone 1 map, [#1](https://github.com/dangth1101/IS6303/issues/1).
Date: 2026-09-14.

## Recommendation

**Primary: `nomic-ai/nomic-embed-text-v1.5` + `nomic-ai/nomic-embed-vision-v1.5`.**
768 dimensions, 8192-token text tower, Apache-2.0 on both halves, 0.46 GB of fp16 weights,
**zero documents truncated**, measured 11.1 docs/s on this M4 (~49 min for the full corpus).

**Runner-up: `jinaai/jina-clip-v2`.** 1024 dimensions, 8192-token text tower, also zero truncation,
measured 6.5 docs/s (~84 min), 1.73 GB fp16 — but CC-BY-NC-4.0 and a single jointly-trained tower.

**Answer to the map's explicit escape hatch:** it is *not* triggered. A model exists that satisfies
both a shared text+image space and a long-enough text tower inside 16 GB — in fact two do. No split
(multimodal + long-context-text) arm design is needed, and the dense arm can use one identical
`title + description + ingredients` string with no truncation whatsoever.

**Dimensionality figure the schema ticket needs: `vector(768)`.** Well under `pgvector`'s 2000-dim
HNSW/IVFFlat ceiling, so both `hnsw` and `ivfflat` indexes are available with no dimension games.
(Runner-up would be `vector(1024)`, also fine.)

## The decisive criterion: how much of the corpus would each model truncate?

### Method

Sampled 2,900 of the 32,722 rows through the HF datasets-server `/rows` API — 29 evenly-spaced
pages of 100 rows across the whole split, so the sample is a systematic sample, not a head sample.
The full dataset was never downloaded. Built the exact project document
(`title` + `description` + `ingredients`, newline-joined, per the map's fixed document definition)
and tokenized it with each candidate's **real tokenizer** from the Hugging Face hub. Counts include
special tokens, which is the right comparison because CLIP's 77 and SigLIP's 64 are total position
counts, not content budgets.

Field lengths in the sample (characters):

| field | mean | median | p90 | max | empty |
|---|---|---|---|---|---|
| `title` | 26 | 24 | 39 | 79 | 0 |
| `description` | 173 | 149 | 316 | 723 | 0 |
| `ingredients` | 279 | 259 | 443 | 1041 | 0 |
| **whole document** | **480** | **454** | **717** | **1502** | 0 |

No field is ever empty, so every document carries the full three-field load.

### Truncation table

Token counts for the project document, per tokenizer, extrapolated to 32,722:

| tokenizer | mean | median | p90 | p99 | max | docs over 64 | docs over 77 |
|---|---|---|---|---|---|---|---|
| `openai/clip-vit-base-patch32` | 113.6 | 107 | 169 | 242 | 397 | 91.2% (~29,856) | **81.2% (~26,584)** |
| `google/siglip-base-patch16-224` | 99.9 | 94 | 150 | 217 | 341 | **83.9% (~27,441)** | 69.2% (~22,646) |
| `google/siglip2-base-patch16-224` (Gemma) | 116.3 | 110 | 173 | 246 | 424 | **92.3% (~30,195)** | 82.9% (~27,137) |
| `nomic-ai/nomic-embed-text-v1.5` (BERT) | 122.1 | 116 | 183 | 260 | **411** | 93.7% | 85.2% |
| `xlm-roberta-base` (proxy for `jina-clip-v2`) | 138.9 | 132 | 207 | 289 | **462** | 96.8% | 92.1% |

The bolded cells are the numbers that matter: **CLIP at 77 tokens truncates ~26,584 of 32,722
documents (81%); SigLIP and SigLIP 2 at 64 tokens truncate ~27,441 and ~30,195 (84% and 92%).**
That is exactly the artifact the map warns about — a dense arm built on any CLIP-family or
SigLIP-family text tower would be scored on documents whose ingredient lists are mostly discarded,
and any BM25 win would be uninterpretable.

The mirror-image finding is just as decisive: **not one sampled document exceeds 512 tokens under
any tokenizer** (longest observed: 462 tokens with the XLM-RoBERTa tokenizer, 411 with BERT). So the
8192-token towers of both recommended candidates truncate **0 documents**, and even a 512-token
tower would have been sufficient. The recipe document is genuinely small; the problem was never
document length, it was that CLIP-family towers are unusually short.

Two consequences for the map's open questions:

- **Chunking is a no-op.** A p99 of ~289 tokens against an 8192-token window leaves nothing to gain
  from splitting a recipe, and splitting would only dilute the title signal. This closes the
  "Chunking" item in the map's *Not yet specified* section in the negative.
- **512-token models are back on the table** if a future ticket wants a text-only comparison point.

*Caveat:* `jina-clip-v2`'s own tokenizer could not be loaded in the tokenizer-only environment
(its `trust_remote_code` module imports `torch`), so `xlm-roberta-base` stands in — the v2 text tower
is the Jina XLM-RoBERTa backbone, so this is a close proxy, and it is the *most* token-hungry row in
the table, i.e. the conservative direction. Its verdict (0 documents over 512, let alone 8192) is
robust to any proxy error.

## Measured on this machine (Apple M4, 16 GB, 10-core GPU)

Both candidates were actually run on MPS, not estimated.

| | `nomic` pair | `jina-clip-v2` |
|---|---|---|
| params | 137M text + 93M vision = 230M | 865M (561M text + 304M vision) |
| weights | 0.46 GB fp16 (ran fp32 here) | 1.73 GB fp16; peak MPS allocation **1.75 GB** |
| text throughput | **11.1 docs/s** (fp32, batch 32, 512 docs) | **6.5 docs/s** (fp16, batch 32, 256 docs) |
| full-corpus text encode | **~49 min** | **~84 min** |
| image encode | 3 images in 1.94 s cold | not benchmarked |
| text dim / image dim | **768 / 768** (verified equal) | **1024** |
| model load time | seconds | 97.6 s first load |

Neither comes close to the 16 GB ceiling — peak MPS allocation for the larger of the two was
1.75 GB. Memory is a non-issue for this decision; a **fp16 batch job needs no checkpointing on
memory grounds**, though checkpointing is still worth it for the ~33k image fetches.

At ~49 min for text, the encode job is a coffee break, not a project risk. This closes the map's
*"Embedding batch job — wall-clock cost"* open item on the text side.

### The shared space was verified empirically, not just claimed

Nomic's shared-space claim is the load-bearing one (it is two checkpoints, not one model), so it was
tested rather than trusted. Encoding three recipe titles as `search_query:` text and their three
allrecipes images through the vision tower, then taking the dot product of the L2-normalised 768-dim
vectors:

```
             img0     img1     img2
text0      0.0901   0.0499   0.0438
text1      0.0509   0.1044   0.0316
text2      0.0414   0.0260   0.0960
```

The diagonal wins on every row — text and image embeddings are directly comparable in one space, as
the model card claims. Two things to carry forward, though:

1. **Cross-modal similarities are compressed** (~0.09–0.10 for a correct match, vs the ~0.5–0.9 one
   expects from text-text cosine in this family). The ranking is right, but the *magnitudes* are not
   on the same scale as text-text scores. Since RRF fuses **ranks**, not scores, this is harmless for
   the fusion arm — which is a point in RRF's favour and worth a sentence in the write-up. It would
   *not* be harmless for any score-based fusion, so do not switch to weighted-score fusion later
   without revisiting this.
2. Nomic's alignment method is LiT-style with the **text tower frozen** ("instead lock the text
   embedder"). That is a genuine advantage for this project: the text tower is unmodified
   `nomic-embed-text-v1.5`, a strong standalone MTEB retriever, so the dense arm is a *fair*
   representative of dense retrieval. `jina-clip-v2`'s text tower was trained jointly against an
   image contrastive loss, a known source of text-only retrieval degradation — which would weaken
   the dense arm for a reason unrelated to sparse-vs-dense, the same class of artifact the map is
   trying to avoid. **This, not the licence, is the main reason nomic is primary.**

## Full candidate comparison

| model | shared space | text tokens | dim | params | fp16 | licence | verdict |
|---|---|---|---|---|---|---|---|
| **`nomic-embed-text-v1.5` + `nomic-embed-vision-v1.5`** | yes (verified above) | **8192** | **768** (Matryoshka 768/512/256/128/64) | 230M | 0.46 GB | **Apache-2.0** (both) | **RECOMMENDED** |
| **`jinaai/jina-clip-v2`** | yes (single model) | **8192** | **1024** (Matryoshka to 64) | 865M | 1.73 GB | CC-BY-NC-4.0 | **RUNNER-UP** |
| `jinaai/jina-clip-v1` | yes | 8192 (`model_max_length`) | 768 | 223M | 0.45 GB | Apache-2.0 | viable fallback; paper only documents training to 512 tokens, English-only |
| `openai/clip-vit-base-patch32` | yes | 77 | 512 | 151M | 0.28 GB | MIT (upstream repo; **no licence tag on HF**) | rejected: truncates 81% |
| `openai/clip-vit-large-patch14` | yes | 77 | 768 | 428M | 0.80 GB | MIT (upstream) | rejected: truncates 81% |
| `laion/CLIP-ViT-H-14-laion2B` | yes | 77 | 1024 | 986M | 1.84 GB | MIT | rejected: truncates 81% |
| `laion/CLIP-ViT-bigG-14` | yes | 77 | 1280 | 2540M | 4.73 GB | MIT | rejected: truncates 81% |
| `google/siglip-base-patch16-224` | yes | **64** | 768 | 203M | 0.38 GB | Apache-2.0 | rejected: truncates 84% |
| `google/siglip-so400m-patch14-384` | yes | 64 | 1152 | 878M | 1.64 GB | Apache-2.0 | rejected: truncates 84% |
| `google/siglip2-base-patch16-224` | yes | **64** | 768 | 375M | 0.70 GB | Apache-2.0 | rejected: truncates 92% |
| `google/siglip2-so400m-patch14-384` | yes | 64 | 1152 | 1136M | 2.12 GB | Apache-2.0 | rejected: truncates 92% |
| `google/siglip2-giant-opt-patch16-384` | yes | 64 | **1536** | 1872M | 3.49 GB | Apache-2.0 | rejected: truncates 92% |
| `apple/DFN5B-CLIP-ViT-H-14` | yes | 77 | 1024 | 986M | 1.84 GB | `apple-amlr` (not OSI) | rejected: truncation + licence |
| `apple/aimv2-large-patch14-224-lit` | yes | 77 | — | 437M | 0.87 GB | `apple-amlr` | rejected: truncation + licence |
| `apple/aimv2-large-patch14-*` (non-LiT) | **no** — vision only | n/a | — | 0.3B | — | `apple-amlr` | rejected: no shared space |
| MobileCLIP / MobileCLIP2 | yes | 77 (a `context256` variant exists) | — | 11M–428M | small | code MIT, **weights Apple ML Research TOU** | rejected: truncation + licence |
| `mlx-community/clip-*`, `mlx-community/siglip*` | yes | 77 / 64 | 512–1152 | 88M–0.9B | small | Apache-2.0 | rejected: inherit upstream truncation |
| `jinaai/jina-embeddings-v4` | yes | 32768 | 2048 (Matryoshka) | 3.75B | ~7.5 GB | **Qwen Research License** | overkill; 7.5 GB + 32k activations is the wrong shape for 16 GB and a 3-week project |
| `TIGER-Lab/VLM2Vec-Full` | yes | 131072 (`original` 4096) | **3072** | 4.15B | ~8.3 GB | Apache-2.0 | rejected: **3072 dim exceeds pgvector's 2000-dim index ceiling**, and 8.3 GB is heavy |
| `royokong/e5-v` | yes | not stated | 4096 | 8.36B | ~16.7 GB | **none declared** | rejected: does not fit 16 GB in fp16; licence unresolved |

### On the "any Apple/MLX-native option" question

**There is no MLX-native long-context multimodal embedder.** Every Apple- or MLX-published option
caps text at 64–77 tokens (256 at best, for MobileCLIP `context256`), and the Apple-published weights
are `apple-amlr` / Apple ML Research TOU rather than open licences. MLX buys nothing here; both
recommended models run fine through PyTorch MPS, as measured above.

### On `pgvector` dimensionality

`pgvector`'s `vector` type stores up to 16,000 dimensions but **HNSW and IVFFlat index only up to
2,000**. That eliminates `VLM2Vec` (3072) and `e5-v` (4096) outright for an indexed dense arm, and it
is why the LLM-backbone tier is a dead end for this project regardless of memory. Both recommended
models (768 / 1024) are comfortably indexable.

## Licence assessment for a graded academic project

- **Nomic pair: Apache-2.0 on both halves.** Unambiguously fine, and the only candidate with a clean
  open licence *and* a long text tower. **Gotcha:** the official repo `nomic-ai/nomic-embed-vision-v1.5`
  is Apache-2.0, but the third-party mirror `tomaarsen/nomic-embed-vision-v1.5-st` is tagged
  CC-BY-NC-4.0. Pull the `nomic-ai` org repo.
- **`jina-clip-v2`: CC-BY-NC-4.0.** Non-commercial. Acceptable for coursework and research, which is
  what this is, but it must be stated in the write-up's limitations and it forecloses any later
  deployment. This is a real reason to keep it as runner-up rather than primary.
- OpenAI's HF CLIP repos carry **no licence metadata at all** — the only authoritative licence is the
  MIT `LICENSE` in `github.com/openai/CLIP`. Moot given they are rejected on truncation.

## Implementation gotchas found while testing

These are all first-hand from running the models, and each one would have cost an hour later.

1. **Nomic text requires a task prefix.** Documents must be prefixed `search_document: ` and queries
   `search_query: `. Omitting these silently degrades retrieval. The prefix must be applied
   identically in the ingest job and the query path, and it does **not** change the document
   definition the map fixes across arms — the prefix is a model input convention, not corpus text,
   and the sparse arm must not receive it.
2. **`jina-clip-v2` has real version friction.** Its `trust_remote_code` module fails on
   transformers 5.x two different ways: `configuration_clip.py` calls `hasattr(torch, torch_dtype)`
   so a `torch.dtype` object raises `TypeError` (pass `torch_dtype="float16"` as a *string*), and
   `modeling_clip.py` imports `clip_loss` from `transformers.models.clip.modeling_clip`, which no
   longer exists. It only ran under a pinned `transformers==4.49.0`. If the runner-up is ever
   promoted, that pin is mandatory. The nomic pair loaded cleanly on current transformers.
3. **SigLIP 2's tokenizer lies about its limit.** `tokenizer_config.json` reports
   `model_max_length: 1000000000000000019884624838656` (unbounded) while the model still has only 64
   position embeddings, so naive truncation checks silently pass and the model then indexes out of
   range. Pass `padding="max_length"`, as the transformers SigLIP docs instruct. Recorded because
   this is precisely the trap that could have produced a "SigLIP handles our documents" false
   conclusion. `google/siglip-so400m-patch14-224` is a worse outlier still: 16 tokens.
4. **SigLIP's 64 is a library default, not a card statement.** The `config.json` files omit
   `max_position_embeddings` entirely; the 64 comes from `SiglipTextConfig`'s default in
   transformers. The map's stated "SigLIP 64" is correct, now sourced.
5. **The allrecipes CDN rejects non-browser user agents with HTTP 460.** A plain
   `urllib.request.urlopen` on an `image` URL from the dataset fails; adding a browser
   `User-Agent` header succeeds. This affects Milestone 2's image-fetch step, not model selection,
   but it is a hard blocker discovered by accident and belongs in the ingest ticket — along with
   rate-limiting, since ~33k requests to a third party need throttling and resumability. (The HF
   datasets-server itself also returned HTTP 429 during sampling, so throttle that path too.)

## Limitations of this analysis

- Truncation percentages are extrapolated from a **2,900-row systematic sample (8.9%)**, not the full
  32,722 rows. The margins are so wide (81–92% vs 0%) that no plausible sampling error changes any
  verdict, but the exact counts will shift slightly on the full corpus. Recompute cheaply during
  ingest if an exact figure is wanted for the write-up.
- Throughput was measured on 256–512 documents with a cold-ish cache, single process, batch 32, no
  attempt at tuning batch size or dtype. Nomic ran in fp32; fp16 should be faster than the 49 min
  quoted. Treat both figures as conservative upper bounds on wall-clock.
- **Retrieval *quality* was not measured.** This ticket selected on context length, memory, licence
  and dimensionality — the hard constraints — and verified the shared space functions. Whether
  nomic's dense arm actually beats BM25 on the hand-crafted qrels is the project's headline question
  and is not prejudged here. The map already accepts "sparse wins" as a plausible outcome.
- Only the nomic shared space was verified end-to-end; `jina-clip-v2`'s image tower was not
  benchmarked, only its text tower.

## Sources

Primary sources only — model cards, `config.json` files, the HF API, upstream repos, and papers.

Recommended models:
- https://huggingface.co/nomic-ai/nomic-embed-text-v1.5 (8192 tokens, Matryoshka dims, task prefixes)
- https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5 and its `config.json` (shared-space claim, 768 dim, Apache-2.0)
- https://arxiv.org/abs/2406.18587 — *Nomic Embed Vision: Expanding the Latent Space* (locked text embedder, LiT-style)
- https://huggingface.co/jinaai/jina-clip-v2 and its `config.json`; https://huggingface.co/api/models/jinaai/jina-clip-v2 (8192 tokens, 1024 dim, CC-BY-NC-4.0, 865,278,476 params)
- https://arxiv.org/abs/2412.08802 — *jina-clip-v2*
- https://huggingface.co/jinaai/jina-clip-v1 and `config.json`; https://arxiv.org/html/2405.20204v1

CLIP / SigLIP family:
- https://huggingface.co/openai/clip-vit-base-patch32/raw/main/config.json (`max_position_embeddings: 77`)
- https://huggingface.co/openai/clip-vit-large-patch14/raw/main/config.json; https://raw.githubusercontent.com/openai/CLIP/main/LICENSE
- https://raw.githubusercontent.com/mlfoundations/open_clip/main/docs/model_profile.csv and https://github.com/mlfoundations/open_clip/tree/main/src/open_clip/model_configs (`context_length: 77` across all mainstream checkpoints)
- https://huggingface.co/google/siglip-base-patch16-224 `config.json` + `tokenizer_config.json`
- https://huggingface.co/google/siglip-so400m-patch14-224/raw/main/config.json (`max_position_embeddings: 16`)
- https://huggingface.co/docs/transformers/en/model_doc/siglip (`SiglipTextConfig` default 64; `padding="max_length"`)
- https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/siglip2/configuration_siglip2.py (`max_position_embeddings: int = 64`)
- https://huggingface.co/google/siglip2-base-patch16-224 `config.json` + `tokenizer_config.json`; `siglip2-large-patch16-384`, `siglip2-so400m-patch14-384`, `siglip2-giant-opt-patch16-384`
- https://arxiv.org/abs/2502.14786 — *SigLIP 2* ("we set the text length to 64")

Apple / MLX:
- https://huggingface.co/apple/aimv2-large-patch14-224-lit/raw/main/config.json (`max_context_length: 77`)
- https://github.com/apple/ml-mobileclip (context77 / context256; Apple ML Research Model TOU)
- https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14/raw/main/open_clip_config.json (`apple-amlr`)
- https://github.com/ml-explore/mlx-examples/tree/main/clip (only CLIP B/32 and L/14)

LLM-backbone tier:
- https://huggingface.co/jinaai/jina-embeddings-v4 (32768 tokens, 2048 dim, Qwen Research License)
- https://huggingface.co/TIGER-Lab/VLM2Vec-Full and `config.json` (`hidden_size: 3072`)
- https://huggingface.co/royokong/e5-v (4096 dim, no licence declared)

Infrastructure:
- https://github.com/pgvector/pgvector — `vector` up to 16,000 dims; HNSW / IVFFlat index up to 2,000

Dataset:
- https://datasets-server.huggingface.co/first-rows and `/rows` for `Shengtao/recipe` (field lengths, `image` URL strings)
