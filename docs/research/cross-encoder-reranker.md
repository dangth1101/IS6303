# Research: cross-encoder reranker for the fourth arm (RRF + Rerank)

Resolves [#5](https://github.com/dangth1101/IS6303/issues/5). Part of [#1](https://github.com/dangth1101/IS6303/issues/1) — Milestone 1 (Plan + ADRs) / Milestone 5 (Evaluation).

Hardware: Apple M4, 16 GB unified memory, 10-core GPU, macOS 26.5.1, `torch 2.14.0`, `sentence-transformers 6.0.1`, `transformers 5.17.0`.

---

## Recommendation

**Use `cross-encoder/ms-marco-MiniLM-L6-v2` at rerank depth 100, with 200 as the hard ceiling.**

| | |
|---|---|
| Licence | **Apache-2.0** — no restriction, academic or otherwise |
| Size | 22.7 M params, 91 MB `model.safetensors` |
| Max input | 512 tokens; our longest query-document pair is **301** → **0 % truncation** |
| Measured latency (MPS) | **2.2 ms per query-document pair** steady state |
| Depth 100 | **222 ms** per query |
| Depth 200 | **448 ms** per query |
| Memory | 596 MiB process RSS, 87 MiB on the GPU |
| Integration | `CrossEncoder(...)`, one line, no `trust_remote_code`, no pinned deps |

Runner-up, if a ~1-point BEIR gain is judged worth extra fragility: `jinaai/jina-reranker-v1-turbo-en` (Apache-2.0, 37.8 M, 3.2 ms/pair MPS). It carries two real costs — `trust_remote_code=True`, and it **is currently broken on `transformers` 5.x** (see [Traps](#traps-found-the-hard-way)).

**Rejected**: both BGE rerankers, on measured latency (see below). `bge-reranker-v2-m3` at ~100 ms/pair supports a rerank depth of **three** inside an interactive budget. No mature MLX-native text cross-encoder exists, and Ollama cannot rerank at all.

---

## 1. The document, measured

Sampled 500 real rows from `Shengtao/recipe` via the HF datasets-server (5 offsets of 100 across the split: 0, 5 000, 12 000, 20 000, 30 000) and built the map's fixed document string `title + description + ingredients`.

| | median | p95 | max |
|---|---|---|---|
| Words | 81 | 152 | 219 |
| Tokens, document alone (BERT WordPiece) | 113.5 | 216 | 296 |
| Tokens, `(query, document)` pair incl. specials | 118.5 | 221 | **301** |
| Tokens, pair under XLM-R SentencePiece (BGE) | 135.5 | 252 | **339** |

Source: `https://datasets-server.huggingface.co/rows?dataset=Shengtao%2Frecipe&config=default&split=train`.

**The truncation trap does not bite here.** Every candidate reranker has a 512-token window, so `pct_over_512 = 0.00 %` for all of them — the p95 pair is under half the window and the worst case in 500 samples uses 66 % of it. Reranker choice is therefore *not* constrained by length, and that is a genuine asymmetry with the embedding arm.

**Cross-reference for the embedding ticket (map risk "CLIP-family text towers truncate hard"): 83.0 % of documents exceed CLIP's 77-token limit and 92.4 % exceed SigLIP's 64.** So the risk the map flags is real and severe on the embedding side, and simultaneously a non-issue on the rerank side. Do not let the reranker's comfort be read as reassurance about the embedder.

## 2. Measured latency on this M4

Method: `sentence-transformers` `CrossEncoder.predict` over real `(query, document)` pairs, `max_length=512`, `batch_size=32`, query `"easy baked chicken thighs"` (4 words, matching the map's short-lookup workload). One discarded warm-up batch, then median of 5 timed repeats per depth. Both `device="mps"` and `device="cpu"` measured.

Totals are **milliseconds per query** — that is, the whole rerank stage for that many candidates.

### MPS (10-core GPU)

| Model | d=10 | d=25 | d=50 | d=100 | d=200 | ms/pair @200 |
|---|---|---|---|---|---|---|
| `ms-marco-TinyBERT-L2-v2` | 11 | 15 | 28 | **48** | 90 | 0.45 |
| **`ms-marco-MiniLM-L6-v2`** | 28 | 79 | 151 | **222** | 448 | **2.24** |
| `jina-reranker-v1-tiny-en` | 35 | 105 | 197 | 275 | 503 | 2.51 |
| `jina-reranker-v1-turbo-en` | 36 | 134 | 244 | 357 | 647 | 3.23 |
| `ms-marco-MiniLM-L12-v2` | 51 | 153 | 286 | 403 | 774 | 3.87 |
| `bge-reranker-base` | 159 | 532 | 995 | 1 536 | 3 111 | 15.56 |
| `bge-reranker-v2-m3` | 1 020 | 2 561 | 4 877 | 9 579 | 21 480 | 107.40 |

### CPU

| Model | d=10 | d=25 | d=50 | d=100 | d=200 | ms/pair @200 |
|---|---|---|---|---|---|---|
| `ms-marco-TinyBERT-L2-v2` | 6 | 18 | 29 | 42 | 79 | 0.39 |
| `ms-marco-MiniLM-L6-v2` | 47 | 178 | 359 | 470 | 861 | 4.30 |
| `jina-reranker-v1-tiny-en` | 47 | 191 | 335 | 484 | 848 | 4.24 |
| `jina-reranker-v1-turbo-en` | 90 | 267 | 454 | 639 | 1 131 | 5.65 |
| `ms-marco-MiniLM-L12-v2` | 94 | 342 | 670 | 945 | 1 717 | 8.59 |
| `bge-reranker-base` | 298 | 1 042 | 1 955 | 3 044 | 6 199 | 30.99 |
| `bge-reranker-v2-m3` | 1 116 | 4 322 | 8 173 | 11 714 | 21 795 | 108.97 |

### Reading these numbers

- **MPS wins, roughly 2×**, for every model above TinyBERT. This contradicts the general MPS caution in [pytorch#148219](https://github.com/pytorch/pytorch/issues/148219) ("dispatch time of MPS ops trashes their performance for small to medium size inputs") — that issue benchmarks *elementwise ops*, not a batched transformer forward pass. At `batch_size=32` over 512-wide sequences the batch is large enough for MPS to pay off. Use `device="mps"`, but the CPU column is a working fallback, not a cliff.
- **Cost is sub-linear in depth** up to ~100 because fixed overhead (tokenisation, dispatch, host↔GPU copy) amortises across batches. `ms-marco-MiniLM-L6-v2` costs 3.0 ms/pair at depth 50 but 2.2 ms/pair at depth 200. Small depths get a worse per-pair rate — a reason not to set depth too low.
- **The 30-40× spread between MiniLM-L6 and `bge-reranker-v2-m3` is the whole decision.** The BGE models are not "slower"; at interactive depth they are a different product category.
- Load time and first-call warm-up are excluded deliberately: FastAPI must instantiate the `CrossEncoder` once at startup, not per request. Cold first inference was measured at 150-260 ms even for the small models.

### Third-party numbers, and why they are not used here

The `cross-encoder/*` model cards publish a Docs/Sec table (`TinyBERT-L2-v2` 9000, `MiniLM-L2-v2` 4100, `L4-v2` 2500, `L6-v2` 1800, `L12-v2` 960) with the note *"Runtime was computed on a V100 GPU"* and **no stated sequence length or batch size**. mixedbread's v2 card quotes latency *"measured on A100 GPU"*, likewise unparameterised. Neither transfers to an M4 and neither is used above. The relative ordering does hold: the cards put L6 at 1.9× L12, we measure 1.7×.

There is **no published credible latency benchmark for cross-encoder reranking on Apple Silicon.** `sentence-transformers`' own efficiency doc is *"RTX 3090 GPU, i7-13700K CPU… re-measured in July 2026 under WSL2"*; `mps` appears in it once, only as device selection. llama.cpp's `benches/mac-m2-ultra/` has no rerank entry. This is why the ticket was answered by measurement.

## 3. Maximum supported rerank depth

Budget framing. The UI doubles as the qrels annotation tool, so the relevant number is the added wall-clock an annotator absorbs on every judged query. Allowing the rerank stage **300 ms** keeps a full search request (BM25 + pgvector + RRF + rerank + serialisation) comfortably inside a ~1 s interactive feel; **500 ms** is the point where the arm starts to be noticeable but is still tolerable.

Depth supported by each model, derived from the measured MPS curves:

| Model | depth @ 300 ms | depth @ 500 ms | depth @ 1 s |
|---|---|---|---|
| `ms-marco-TinyBERT-L2-v2` | ~650 | ~1 100 | ~2 200 |
| **`ms-marco-MiniLM-L6-v2`** | **~135** | **~225** | **~450** |
| `jina-reranker-v1-turbo-en` | ~90 | ~155 | ~310 |
| `ms-marco-MiniLM-L12-v2` | ~75 | ~130 | ~260 |
| `bge-reranker-base` | ~20 | ~32 | ~64 |
| `bge-reranker-v2-m3` | **~3** | ~5 | ~9 |

**Answer for the fusion/rerank configuration ticket: rerank depth 100 (222 ms measured), with 200 (448 ms) as the maximum supported depth.**

Rationale for 100 rather than the 200 ceiling:

- It leaves ~80 ms of the 300 ms budget as headroom, so a slower query or a thermally throttled machine does not push the arm past the point an annotator notices.
- Reranking beyond 100 has diminishing recall value here. Depth caps what the reranker *can* fix — it can only reorder what fusion already retrieved — and the corpus is 32 722 documents of ~110 tokens each with short keyword queries. A candidate that RRF ranked 150th on a 3-8 word query is very unlikely to be a graded-3 relevant recipe.
- Depth 100 is a clean fit for RRF: take top-100 from each of BM25 and dense, fuse, rerank the top-100 of the fused list.
- 200 remains available at 448 ms if the evaluation shows recall being left on the table. Because the cost is near-linear past depth 100, doubling depth costs almost exactly 2×; there is no hidden cliff.

If annotation time turns out to be the binding constraint in practice, `ms-marco-TinyBERT-L2-v2` reranks 200 candidates in **90 ms** — 5× faster than MiniLM-L6 — at a published cost of 32.56 vs 39.01 MRR@10 on MS MARCO dev. That is the documented reduced-scope fallback the map asks for up front.

## 4. Candidates surveyed

Parameter counts are exact `safetensors` totals from the HF model API; disk sizes are actual blob bytes. Max input is `tokenizer_config.json`'s `model_max_length`.

### `cross-encoder/ms-marco-*` — Apache-2.0

| Model | Params | Disk | Layers × hidden | Max input | nDCG@10 TREC DL19 | MRR@10 MS MARCO dev |
|---|---|---|---|---|---|---|
| `ms-marco-TinyBERT-L2-v2` | 4.39 M | 17.6 MB | 2 × 128 | 512 | 69.84 | 32.56 |
| `ms-marco-MiniLM-L2-v2` | 15.6 M | 62.5 MB | 2 × 384 | 512 | 71.01 | 34.85 |
| `ms-marco-MiniLM-L4-v2` | 19.2 M | 76.7 MB | 4 × 384 | 512 | 73.04 | 37.70 |
| **`ms-marco-MiniLM-L6-v2`** | **22.7 M** | **90.9 MB** | 6 × 384 | 512 | **74.30** | **39.01** |
| `ms-marco-MiniLM-L12-v2` | 33.4 M | 133.5 MB | 12 × 384 | 512 | 74.31 | 39.02 |

Quality figures quoted from the model cards' Performance table. **L6 → L12 buys +0.01 nDCG and +0.01 MRR for 1.7× the measured latency** — L6 is the family's sweet spot and L12 should not be considered. Backbone is stock `BertForSequenceClassification`, `num_labels=1`; no remote code. Each repo also ships `onnx/model_qint8_arm64.onnx` (23.2 MB for L6) — a ready-made ARM64 int8 build if the CPU path is ever preferred.

**Canonical id note:** HF renamed these; `cross-encoder/ms-marco-MiniLM-L6-v2` (no hyphen before the digit) is canonical. The old `...-L-6-v2` id still resolves via HTTP 307, so old code keeps working, but pin the new id.

### BAAI BGE — permissive licence, unusable latency

| Model | Params | Disk | Backbone | Max input | Licence |
|---|---|---|---|---|---|
| `bge-reranker-base` | 278 M | 1.11 GB fp32 | XLM-R base | 512 | **MIT** |
| `bge-reranker-large` | 560 M | 2.24 GB fp32 | XLM-R large | 512 | **MIT** |
| `bge-reranker-v2-m3` | 568 M | 2.27 GB fp32 | `bge-m3` | 8192 | **Apache-2.0** |
| `bge-reranker-v2-gemma` | 2.51 B | 10.0 GB | Gemma-2B, causal LM | 8192 | Gemma terms |
| `bge-reranker-v2-minicpm-layerwise` | 2.72 B | 10.9 GB | MiniCPM, causal LM | 2048 | — |

Licences are fine; the base card states *"The released models can be used for commercial purposes free of charge."* The problem is entirely speed. Both `-base` and `-v2-m3` load and run correctly on MPS out of the box (no `trust_remote_code`, plain `CrossEncoder`) — they are simply 7× and 48× slower per pair than MiniLM-L6.

The two 10 GB variants are ruled out on memory alone: ~10 GB of fp32 weights before activations on a **16 GB unified** machine that also holds macOS, Postgres, Docker and the Python process. They are also decoder LLMs, not cross-encoders.

`bge-reranker-v2-m3`'s 8192-token window is the one thing it offers that MiniLM-L6 does not, and our p95 pair is 252 tokens — we have no use for it. Measured memory for `bge-reranker-base` was already 1 032 MiB RSS / 1 065 MiB on the GPU versus 596 / 87 for MiniLM-L6.

**Caveat on BGE quality claims: no BGE reranker card publishes machine-readable metrics.** The v2-m3 card's entire evaluation section is PNG images. Every BEIR figure in circulation for these models is third-party and the harnesses disagree — Jina's v2 card gives `bge-reranker-v2-m3` 53.65 BEIR nDCG@10, Jina's v3 card gives 56.51, mixedbread's card gives `bge-reranker-base` 41.6 over 11 BEIR datasets. Not mutually comparable; do not present them as a ladder.

### Jina — v1 is Apache-2.0, everything newer is non-commercial

| Model | Params | Disk | Backbone | Max input | Licence |
|---|---|---|---|---|---|
| `jina-reranker-v1-tiny-en` | 33.0 M | 66.1 MB | `JinaBertModel`, ALiBi | 512 | **Apache-2.0** |
| `jina-reranker-v1-turbo-en` | 37.8 M | 75.6 MB | `JinaBertModel`, ALiBi | 512 | **Apache-2.0** |
| `jina-reranker-v2-base-multilingual` | 278 M | 557 MB | XLM-R + custom Flash | 1024 | **CC-BY-NC-4.0** |
| `jina-reranker-v3` | 597 M | 1.19 GB | Qwen3-0.6B, listwise | 131 072 | **CC-BY-NC-4.0** |
| `jina-reranker-m0` | 2.44 B | 4.89 GB | Qwen2-VL, multimodal | 32 768 | **CC-BY-NC-4.0** |

The v1 licence is better than commonly assumed: `README.md` front-matter reads `license: apache-2.0` at revisions `2428e0d7` (2024-06-20) and `5ee87fc5` (2025-01-06), with no licence-changing commit since. From v2 onward the card is explicit: *"licenced for research and evaluation purposes under CC-BY-NC-4.0."* That carve-out does cover this graded academic project, but it would block any commercial demo or permissive re-publication of the code — a strategic cost, not just paperwork.

Jina's own card puts `jina-reranker-v1-turbo-en` at **49.60** nDCG@10 over 17 BEIR datasets versus **48.64** for `ms-marco-MiniLM-L-6-v2` and 47.89 for `bge-reranker-base` — about 1 point for 1.7 M extra params. Measured here, turbo costs 3.23 ms/pair versus MiniLM-L6's 2.24, i.e. 44 % more latency and a drop from depth ~135 to ~90 at a 300 ms budget. Note the "8192 tokens" headline for v1 is misleading: `max_position_embeddings` is 8192 (ALiBi) but `tokenizer_config.json` caps `model_max_length` at 512, and `CrossEncoder` inherits the tokenizer's value.

`jina-reranker-v3` is not a pairwise cross-encoder at all — it is a listwise "last but not late interaction" model that attends over up to 64 documents in one context ([arXiv:2509.25085](https://arxiv.org/abs/2509.25085)), so `CrossEncoder` is not its interface and it would not be an apples-to-apples fourth arm. It does have an official Apple Silicon port, `jinaai/jina-reranker-v3-mlx` (*"native MLX implementation with 100 % matching of rank scores"*), plus GGUFs down to 397 MB Q4_K_M — the most interesting MLX artefact found, but CC-BY-NC and architecturally off-spec. `jina-reranker-m0` is a visual-document reranker; wrong tool, and its text-only BEIR (58.95) is beaten by smaller models.

### mixedbread and Alibaba GTE — all Apache-2.0

| Model | Params | Disk | Backbone | Max input | BEIR nDCG@10 |
|---|---|---|---|---|---|
| `mxbai-rerank-xsmall-v1` | 70.8 M | 142 MB | DeBERTa-v2 | 512 | 43.9 (11 sets) |
| `mxbai-rerank-base-v1` | 184 M | 369 MB | DeBERTa-v2 | 512 | 46.9 |
| `mxbai-rerank-large-v1` | 435 M | 870 MB | DeBERTa-v2 | 512 | 48.8 |
| `mxbai-rerank-base-v2` | 494 M | 988 MB | **Qwen2 causal LM** | 32 768 | 55.57 (self-reported) |
| `mxbai-rerank-large-v2` | 1.54 B | 3.09 GB | **Qwen2 causal LM** | 32 768 | 57.49 (self-reported) |
| `gte-multilingual-reranker-base` | 306 M | 612 MB | `New` (RoPE) | 512 (8192 pos) | PNG only |

All Apache-2.0 with verbatim `LICENSE` files. Not benchmarked locally because none is competitive on the axis that matters: the v1 line is DeBERTa-v2 at 3-19× MiniLM-L6's parameter count, and **v2 is a Qwen2 decoder LLM despite the name** (`mxbai-rerank-base-v2` is subtitled "ProRank-0.5B") — same size class as `bge-reranker-v2-m3`, which we measured at 100 ms/pair. mxbai v2 also has **no technical report** ("coming soon"), so its 55.57/57.49 figures are self-reported and unreproducible. GTE requires `trust_remote_code` pointing at a *second* repo (`Alibaba-NLP/new-impl`), publishes results as a PNG, and its recommended acceleration path is xformers, which is CUDA-only.

### MLX-native — nothing mature exists

Checked thoroughly; the honest answer is no.

- **`mlx-community` hosts 13 `text-ranking` models and not one is a BERT-family cross-encoder.** All are decoder-LLM or VL rerankers: `Qwen3-Reranker-0.6B-4bit` (0.34 GB, Apache-2.0), `-0.6B/4B/8B-mxfp8`, `jina-reranker-v3-4bit-mxfp4` (0.32 GB, CC-BY-NC), `mxbai-rerank-large-v2` (3.09 GB, unquantised — no MLX size benefit), and eight `Qwen3-VL-Reranker-2B-*`. Searches for `cross-encoder`, `ms-marco` and `text-classification` under that author return empty; **no MLX conversion of any `bge-reranker` or `ms-marco` cross-encoder exists on the Hub.**
- **`ml-explore/mlx-examples` has no reranker and no request for one.** Its `bert/` example is a plain `BertModel` with no sequence-classification head. Issue/PR search for `rerank` in both `mlx-examples` and `mlx` returns `total_count: 0`. `mlx_lm/server.py` routes only `/v1/completions`, `/v1/chat/completions`, `/health`, `/v1/models` — no `/v1/rerank`. Apple ships no first-party MLX reranker.
- **`Blaizzy/mlx-embeddings` cannot run BGE rerankers.** Its `modernbert.py` has a genuine single-logit reranker head, but `bert.py` and `xlm_roberta.py` have no classification head at all — which rules out exactly the models one would want to port. The PR that would fix text reranking, [mlx-embeddings#62](https://github.com/Blaizzy/mlx-embeddings/pull/62), has been open and unreviewed since 2026-04-30 with 0 comments. The only real MLX BERT cross-encoder found anywhere, `afanjul/gte-reranker-modernbert-base-mlx`, has 55 downloads and 0 likes.

Given that MiniLM-L6 already reranks 100 candidates in 222 ms via MPS, there is no latency problem for MLX to solve. **Recommendation: do not pursue MLX for the reranker.**

### llama.cpp and Ollama

**llama.cpp supports reranking properly** — `--rerank` / `--reranking` plus `--pooling rank` exposes `POST /reranking` (aliases `/rerank`, `/v1/rerank`, `/v1/reranking`) taking `query`, `documents`, `top_n`. Added in [llama.cpp#9510](https://github.com/ggml-org/llama.cpp/pull/9510); the GGUF converter registers `BertForSequenceClassification`, `XLMRobertaForSequenceClassification` et al., and the regression test `tools/server/tests/unit/test_rerank.py` runs `jina-reranker-v1-tiny-en` — a BERT cross-encoder is what this endpoint is tested against. Ready-made GGUFs exist (`gpustack/bge-reranker-v2-m3-GGUF` down to 438 MB Q4_K_M; `ggml-org/jina-reranker-v1-turbo-en-GGUF`). One caution from [llama.cpp#19756](https://github.com/ggml-org/llama.cpp/issues/19756): on an M2 Max, `bge-reranker-v2-m3` reranking works, while `Qwen3-Reranker-0.6B` and `jina-reranker-v3` GGUFs crash with stack overflow — the BERT/XLM-R path is the reliable one on Metal, the decoder rerankers are not.

This is a viable route but not needed. It would add a second serving process next to FastAPI for a model we already run in-process at 2.2 ms/pair.

**Ollama cannot rerank. Do not plan around it,** despite it being installed. `docs/api.md` has zero occurrences of `rerank`; the installed client (0.34.0) has no rerank subcommand; only `/api/embed` exists. The canonical request [ollama#3368](https://github.com/ollama/ollama/issues/3368) has been **open since 2024-03-27** (109 comments, 287 👍) with 13 closed duplicates and 4 rejected PRs; the two open PRs are stalled. `ollama.com/library` lists no reranker among its 240 official models — the `dengcao/bge-reranker-v2-m3`-style hits are user-namespace uploads that give you a chat/embed model, not a scoring API.

## Traps found the hard way

1. **`jinaai/jina-reranker-v1-*` is broken on `transformers` 5.x.** Loading either v1 model fails with `ModuleNotFoundError: No module named 'transformers.onnx'` — the repo's remote code imports a module removed in transformers v5. It only ran here under `--with 'transformers<5'`. If Jina turbo is ever chosen, that pin becomes a project constraint, and it is a standing maintenance risk on a model whose code Jina no longer touches. This is a concrete argument for MiniLM-L6's zero-remote-code profile beyond the 1-point BEIR difference.
2. **`CrossEncoder` inherits `max_length` from the tokenizer, not from `max_position_embeddings`.** Always pass `max_length=512` explicitly so the truncation point is in our code and not implied by a third-party JSON file.
3. **The HF API silently ignores `library=mlx`** and will return `BAAI/bge-reranker-v2-m3` and `cross-encoder/ms-marco-MiniLM-L6-v2` as if they were MLX models. Use `?author=mlx-community&pipeline_tag=text-ranking` instead.
4. **Never time the first inference call.** Cold first-call cost was 150-260 ms even for TinyBERT — enough to distort a small-depth measurement by 5×. Warm up once at FastAPI startup.

## Limitations of this study

- Latency only; **no retrieval quality was measured on our corpus.** All nDCG/MRR figures are the publishers' own on MS MARCO / BEIR. Whether MiniLM-L6 actually improves over RRF on 32 722 recipes with short keyword queries is exactly what Milestone 5 exists to find out, and the honest prior is that the gain may be small — same caution the map records for "sparse wins".
- Single query string, single batch size (32), 500-document sample rather than the full corpus. Latency depends on sequence length, which we sampled representatively, but batch-size tuning was not swept.
- Thermals not controlled. Figures are from a cool machine; sustained annotation sessions may run slower, which is part of why depth 100 rather than 200 is recommended.
- `mxbai` and `gte` families were surveyed but not measured locally.

## Downstream inputs this produces

- **Fusion/rerank configuration ticket**: rerank depth **100**, ceiling **200**, model `cross-encoder/ms-marco-MiniLM-L6-v2` on `device="mps"`, `max_length=512`, `batch_size=32`, instantiated once at FastAPI startup.
- **API contract ticket**: the rerank arm adds ~222 ms; the response should carry the cross-encoder score alongside the fused rank so the qrels UI can show both without a second call.
- **Embedding-model ticket**: 83.0 % of documents exceed 77 tokens and 92.4 % exceed 64 — CLIP/SigLIP text towers would truncate almost the entire corpus. Confirms the map's stated risk with numbers.
- **Reduced-scope fallback**: `ms-marco-TinyBERT-L2-v2`, 200 candidates in 90 ms, if annotation throughput becomes the binding constraint.

## Licence summary

| Model | Licence | Academic use | Commercial use |
|---|---|---|---|
| `cross-encoder/ms-marco-*` (all) | Apache-2.0 | Yes | Yes |
| `BAAI/bge-reranker-base` / `-large` | MIT | Yes | Yes |
| `BAAI/bge-reranker-v2-m3` | Apache-2.0 | Yes | Yes |
| `jinaai/jina-reranker-v1-turbo-en` / `-tiny-en` | Apache-2.0 | Yes | Yes |
| `jinaai/jina-reranker-v2` / `v3` / `m0` | CC-BY-NC-4.0 | Yes (research/eval carve-out) | **No** |
| `mixedbread-ai/mxbai-rerank-*` | Apache-2.0 | Yes | Yes |
| `Alibaba-NLP/gte-multilingual-reranker-base` | Apache-2.0 | Yes | Yes |
| `mlx-community/Qwen3-Reranker-*` | Apache-2.0 | Yes | Yes |

The recommended model is Apache-2.0, so no licence constraint reaches the write-up.

## Reproducing the benchmark

```bash
uv run --with sentence-transformers --with torch python bench_rerank.py \
  cross-encoder/ms-marco-MiniLM-L6-v2,BAAI/bge-reranker-base mps,cpu
```

The harness samples documents from the datasets-server, reports the token-length distribution per tokenizer, discards a warm-up batch, and takes the median of 5 repeats at depths 10/25/50/100/200. It is not committed — it belongs in the evaluation harness (Milestone 5) once that ticket lands, so the numbers here can be re-measured against the real corpus rather than a 500-row sample.

## Sources

- HF datasets-server rows API for `Shengtao/recipe` — document length distribution.
- HF model API `?blobs=true` and each repo's raw `config.json` / `tokenizer_config.json` / `README.md` — exact parameter counts, blob sizes, `model_max_length`, licence front-matter.
- `cross-encoder/*` model cards — nDCG@10 TREC DL19 / MRR@10 MS MARCO dev / Docs-Sec table.
- `jinaai/jina-reranker-v1-turbo-en` card — 17-dataset BEIR comparison table.
- `mixedbread-ai/mxbai-rerank-*-v1` cards — 11-dataset BEIR via Pyserini.
- [llama.cpp `tools/server/README.md`](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) — rerank flags and endpoint; PRs #9510, #13858, #14029, #22553.
- [ollama `docs/api.md`](https://github.com/ollama/ollama/blob/main/docs/api.md) and [ollama#3368](https://github.com/ollama/ollama/issues/3368) — no rerank support.
- [ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples), `mlx_lm/server.py`, [mlx-embeddings#62](https://github.com/Blaizzy/mlx-embeddings/pull/62) — absence of an MLX cross-encoder.
- `sentence-transformers` `docs/cross_encoder/usage/efficiency.rst` — confirms no Apple Silicon reference figures exist.
- [pytorch#148219](https://github.com/pytorch/pytorch/issues/148219) — MPS dispatch-overhead caution (elementwise ops, not models).
- All latency, memory and token-length figures in §1-§3: measured on this machine, 2026-09-14.
