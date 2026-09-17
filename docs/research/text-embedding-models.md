# Text embedding models for the text→text Arm

Research for [#15](https://github.com/dangth1101/IS6303/issues/15), under map [#13](https://github.com/dangth1101/IS6303/issues/13). Dated 2026-09-17.

Premise 8 fixes embedding inference as **local**, and premise 9 says several models each
embed the Corpus and are compared on our own nDCG. So this is a shortlist with evidence,
not a pick. Every number below is either cited to a primary source or measured here; the
measured ones are marked **[measured]** and the method is in [Appendix A](#appendix-a--how-the-measurements-were-made).

---

## 1. The answer first

**Run these five.** They span the size/quality curve, cost under 2.5 h of CPU for one full
Corpus pass each, and none of them carries a reproducibility trap.

| # | Model | Why it earns an Arm |
|---|---|---|
| 1 | `BAAI/bge-small-en-v1.5` | The cheap floor that is still competitive — 51.68 BEIR-15 at 384 dims and **10.8 min** per field-Chunk pass. Smallest pgvector index of any serious candidate. |
| 2 | `BAAI/bge-base-en-v1.5` | Best-understood mid-size English retriever, 53.25 BEIR-15. Its query instruction is **optional**, so it cannot be silently misconfigured. |
| 3 | `Alibaba-NLP/gte-modernbert-base` | Highest BEIR-15 in the sub-150M class (**55.33**), 8192-token context so no Chunk can ever truncate, Apache-2.0, 298 MB. |
| 4 | `ibm-granite/granite-embedding-small-english-r2` | 8192-token context in a **47M** model at 384 dims, 50.9 BEIR-15, 114.8 Chunks/s. Strictly dominates `all-MiniLM-L6-v2` on quality and context at a third of the speed. |
| 5 | `intfloat/e5-base-v2` | Size-matched to bge-base (identical 109.5M/768/512) but from the **prefix-mandatory** family. It is the controlled test of the prefix hazard the ticket names. |

**Two more if the eval budget allows** — not core Arms:

| Model | Role |
|---|---|
| `BAAI/bge-large-en-v1.5` | Ceiling check. 54.29 BEIR-15, but 1024 dims and **91 min** per pass. Answers "does size still buy nDCG on recipe text?" |
| `sentence-transformers/all-MiniLM-L6-v2` | Floor / sanity Arm, and the live demonstration of truncation. 41.95 BEIR-15 is ~10 points below the field. |

### Truncation warnings

> **`all-MiniLM-L6-v2` (256 tokens) is the only shortlisted model that truncates.** **[measured]**
> At field-level chunking it silently cuts **9.31% of `directions` Chunks** and **0.04% of
> `ingredients` Chunks**. Under a whole-Recipe strategy it would cut **47.5%** of Chunks.

> **The ticket's worry about `ingredients` is unfounded, and the worry about `directions` is
> understated.** **[measured]** The 1,162-char max `ingredients` field is 318 BERT tokens —
> fine at 512, marginal at 256. But `directions` reaches **939 tokens**, so 256-token models
> lose the tail of one Chunk in eleven.

> **No 512-token model has a meaningful truncation problem.** **[measured]** At 512 tokens only
> **0.27%** of `directions` Chunks and **0%** of `ingredients` Chunks are cut. The 8192-token
> models (gte-modernbert, granite-r2, gte-v1.5, nomic, arctic) buy zero extra coverage at
> field-level chunking — their context is insurance for a whole-Recipe strategy, not a
> present-tense advantage.

### Two traps found while measuring

1. **The fp16-on-CPU trap — costs 5–8x.** **[measured]** Several current model repos ship
   fp16 weights. PyTorch has no fast fp16 CPU path, so loading them as shipped is catastrophic:
   `thenlper/gte-base` ran at **7.6 Chunks/s** as shipped versus **59.0** forced to fp32, and
   `gte-modernbert-base` at **5.3** versus **41.0**. Always pass
   `model_kwargs={"torch_dtype": torch.float32}` when embedding on CPU. Without this, gte looks
   like a terrible model when it is merely badly loaded.
2. **`trust_remote_code` models are version-fragile.** **[measured]** On `transformers` 5.17.0,
   `nomic-embed-text-v1.5` and `snowflake-arctic-embed-m-v2.0` both **fail to load** — their
   custom modeling files were written against `transformers` 4.x. For a project whose stated
   reason for local inference is *reproducibility* (premise 8), a model that needs a pinned
   `transformers` is a liability. This is why both are excluded despite good scores.

---

## 2. The evidence table

MTEB **Retrieval** = the 15-task BEIR subset, nDCG@10. This is the column the ticket asked
for, not the 56-task overall average.

| Model | Params | Dim | Max seq | Disk (safetensors) | Licence | Prefix | MTEB Retrieval (BEIR-15) | CPU Chunks/s **[measured]** |
|---|---:|---:|---:|---:|---|---|---:|---:|
| `all-MiniLM-L6-v2` | 22.7M | 384 | **256** | 90.9 MB | Apache-2.0 | none | 41.95 | **351.6** |
| `bge-small-en-v1.5` | 33.4M | 384 | 512 | 133.5 MB | MIT | query only, *optional* | 51.68 | 152.0 |
| `e5-small-v2` | 33.4M | 384 | 512 | 133.5 MB | MIT | **required**, both sides | 49.04 † | 153.2 |
| `gte-small` | 33.4M | 384 | 512 | 66.7 MB (fp16) | MIT | none | 49.46 | 154.5 (fp32) / 28.6 (as shipped) |
| `granite-embedding-small-english-r2` | 47.7M | 384 | 8192 | 95.3 MB | Apache-2.0 | none | 50.9 | 114.8 |
| `bge-base-en-v1.5` | 109.5M | 768 | 512 | 438.0 MB | MIT | query only, *optional* | **53.25** | 55.9 |
| `e5-base-v2` | 109.5M | 768 | 512 | 438.0 MB | MIT | **required**, both sides | 50.29 | 56.6 |
| `gte-base` | 109.5M | 768 | 512 | 219.0 MB (fp16) | MIT | none | 51.14 | 59.0 (fp32) / 7.6 (as shipped) |
| `gte-base-en-v1.5` | 136.8M | 768 | 8192 | 547.1 MB | Apache-2.0 | none, but `trust_remote_code` | 54.09 | not run |
| `nomic-embed-text-v1.5` | 136.7M | 768 (MRL 512/256/128/64) | 8192 | 546.9 MB | Apache-2.0 | **required**, 4 task prefixes | 52.8 (v1) | **load failed** |
| `gte-modernbert-base` | 149.0M | 768 | 8192 | 298.0 MB | Apache-2.0 | none | **55.33** | 41.0 (fp32) / 5.3 (as shipped) |
| `granite-embedding-english-r2` | 149.0M | 768 | 8192 | 298.0 MB | Apache-2.0 | none | 53.1 | not run |
| `snowflake-arctic-embed-m-v2.0` | 305.4M | 768 (MRL 256) | 8192 | 1221.5 MB | Apache-2.0 | `"query: "`, query only | 55.4 | **load failed** |
| `embeddinggemma-300m` | 302.9M | 768 (MRL 512/256/128) | 2048 | 1230.4 MB | **Gemma, gated** | **required**, both sides | n/a ‡ | not run |
| `bge-large-en-v1.5` | 335.1M | 1024 | 512 | 1340.6 MB | MIT | query only, *optional* | 54.29 | 17.9 |
| `mxbai-embed-large-v1` | 335.1M | 1024 (MRL) | 512 | 670.3 MB (fp16) | Apache-2.0 | **required** on query | 54.39 | 18.8 (fp32) |
| `gte-large` | 335.1M | 1024 | 512 | 670.3 MB (fp16) | MIT | none | 52.22 | not run |
| `e5-large-v2` | 335.1M | 1024 | 512 | 1340.6 MB | MIT | **required**, both sides | 50.56 † | not run |
| `snowflake-arctic-embed-l-v2.0` | 567.8M | 1024 (MRL 256) | 8192 | 2271.1 MB | Apache-2.0 | `"query: "`, query only | 55.6 | not run |
| `bge-m3` | ~568M | 1024 | 8192 | 2271.1 MB | MIT | none | 48.8 | not run |
| `jina-embeddings-v3` | 572.3M | 1024 (MRL) | 8192 | 1144.7 MB | **CC-BY-NC-4.0** | task LoRA adapters | n/a ‡ | not run |
| `Qwen3-Embedding-0.6B` | 595.8M | 32–1024 (MRL) | 32768 | 1191.6 MB | Apache-2.0 | instruction on query only | n/a ‡ | not run |

† Computed here from the model card's own `model-index` **[measured]**; the method reproduces
the published 50.29 / 51.68 / 53.25 exactly for e5-base-v2, bge-small and bge-base, so the
derived e5-small-v2 figure is trustworthy.
‡ Scored only on **MTEB(eng, v2)**, a different task set — see [§4](#4-a-benchmark-caveat-that-matters).

Params, on-disk safetensors bytes, licence strings and gating status are read from the
Hugging Face model API **[measured]**, not from prose.

---

## 3. The prefix rules, model by model

Getting this wrong degrades nDCG silently, which is exactly why the ticket asks. The families
genuinely differ:

- **BGE v1.5 — optional, query only.** The card says: *"For the `bge-*-v1.5`, we improve its
  retrieval ability when not using instruction. No instruction only has a slight degradation in
  retrieval performance compared with using instruction."* and *"In all cases, the
  documents/passages do not need to add the instruction."* The query instruction, if used, is
  `Represent this sentence for searching relevant passages: `. **Safest family — it cannot be
  catastrophically misconfigured.**
- **E5 — mandatory, both sides, asymmetric.** Retrieval Text gets `passage: `, the query gets
  `query: `. The card's FAQ on whether the prefix is required: *"Yes, this is how the model is
  trained, otherwise you will see a performance degradation."*
- **Nomic — mandatory, four task prefixes.** `search_document: ` for Retrieval Text,
  `search_query: ` for queries, plus `clustering: ` and `classification: `.
- **Arctic Embed v2.0 — query only, `"query: "`.** Confirmed from the repo's own
  `config_sentence_transformers.json`, which declares `"prompts": {"query": "query: "}`; the
  card's `transformers` example applies it *"just on the query"*. Note this differs from
  Arctic v1.5, which used the BGE-style sentence.
- **mxbai-embed-large-v1 — mandatory on the query**, and confusingly it reuses BGE's string
  `Represent this sentence for searching relevant passages: `. The card: *"you have to provide
  the prompt … for query if you want to use it for retrieval."*
- **EmbeddingGemma — mandatory, both sides, and unusually structured.** Query:
  `task: search result | query: {content}`. Document: `title: {title | "none"} | text: {content}`.
  The document template wants a title, which maps neatly onto a Recipe's `title` field.
- **No prefix at all:** `all-MiniLM-L6-v2`, `gte-*` (both generations), `granite-embedding-*-r2`,
  `bge-m3`.

**Operational consequence for premise 9.** Embeddings are keyed by `(chunk, model)`, so the
prefix is a property of the model, not of the Chunk. It must live in the model registry beside
the model name and be applied at encode time on both the ingest and the query path — otherwise
one Arm gets `passage: ` at ingest and nothing at query time, and the resulting nDCG drop looks
like a model quality result when it is a bug.

---

## 4. A benchmark caveat that matters

The ticket asks for "MTEB **retrieval** scores specifically". There are now **two incompatible
things with that name**:

- **MTEB v1 `Retrieval`** — the 15-task BEIR subset. This is what bge, e5, gte, mxbai, arctic
  and granite report, and what the comparison table above uses. Scores land in the **48–56** band.
- **MTEB(eng, v2)** — a rebuilt, differently-weighted English task set. Qwen3-Embedding and
  EmbeddingGemma report against this. Its Retrieval scores land in the **60s**.

Qwen3-Embedding-0.6B's **61.83** is *not* seven points better than gte-modernbert's **55.33**.
They are different benchmarks. Any table that ranks them together is wrong. Since premise 9
says the decision is made on our own nDCG against our own Qrels, this mostly argues for using
the published numbers only to pick *who gets an Arm*, and never to justify a final choice.

---

## 5. Cost of an ingest pass

Measured on the dev's own machine (Apple M4, 10 cores, `torch.set_num_threads(10)`, CPU,
fp32, batch 32, real Recipe text). Chunk counts are computed from the actual Corpus **[measured]**:

| Chunking Strategy | Chunks |
|---|---:|
| field (description / ingredients / directions) | 98,166 |
| step (one Chunk per direction step) | 166,360 |
| fixed window, 256 tokens / 64 overlap | 50,468 |

Minutes for one full Corpus pass:

| Model | field | step | fixed-window |
|---|---:|---:|---:|
| `bge-small-en-v1.5` | 10.8 | 18.2 | 5.5 |
| `granite-embedding-small-english-r2` | 14.3 | 24.2 | 7.3 |
| `e5-base-v2` | 28.9 | 49.0 | 14.9 |
| `bge-base-en-v1.5` | 29.3 | 49.6 | 15.0 |
| `gte-modernbert-base` | 39.9 | 67.6 | 20.5 |
| `bge-large-en-v1.5` | 91.4 | 154.9 | 47.0 |

**The five Tier-1 models cost ~2.1 h of CPU for one field-chunked pass over the whole Corpus.**
Across all three Chunking Strategies that is roughly **6 h**, which is an overnight run, not a
blocker. Adding `bge-large` as a sixth roughly doubles it.

Two levers, if that is still too slow:

- **MPS is free speed.** **[measured]** On the same M4, `device="mps"` gave bge-small
  **332.6 Chunks/s** (2.2x), bge-base **80.8** (1.4x), bge-large **27.9** (1.6x). Still local
  and still reproducible on the same machine, so premise 8 is satisfied.
- **Premise 6 already cuts this.** Chunking is settled on the text Arm first, so the full
  model × strategy grid never has to be run — settle the Chunking Strategy with one or two
  cheap models, then run the rest of the shortlist on the winning strategy only.

An incidental finding for the chunking ticket: **`instructions_list` parses on only 25,402 of
32,722 Recipes (77.6%)** **[measured]**, so a step Chunking Strategy has no step boundaries for
7,320 Recipes and must fall back to a single `directions` Chunk. That is the same ~22% cohort as
the null images noted in the map.

---

## 6. Excluded, and why

| Model | Reason |
|---|---|
| `nomic-embed-text-v1.5` | **Fails to load on `transformers` 5.17.0** **[measured]**. Needs a pinned 4.x. Reproducibility risk against premise 8. |
| `snowflake-arctic-embed-m-v2.0` / `-l-v2.0` | Same — remote code, and additionally demands `xformers` **[measured]**. Scores well (55.4 / 55.6) but is multilingual weight we don't need. |
| `embeddinggemma-300m` | Repo is **`gated: manual`** **[measured]** — needs manual approval plus an HF token before anyone can reproduce the run. Gemma terms are not OSI-open. Excellent model, wrong friction for a coursework artifact. |
| `jina-embeddings-v3` | **CC-BY-NC-4.0**. Coursework is non-commercial so it is *permitted*, but it poisons any later demo or publication, and it also needs remote code. |
| `Qwen3-Embedding-0.6B` | 596M params / 1.19 GB, decoder-based. Scored on the non-comparable MTEB(eng, v2). Cost per pass would be several hours on CPU for a benefit we cannot verify against the rest of the field. |
| `bge-m3` | 2.27 GB for **48.8** BEIR-15 — worse English retrieval than 33M `bge-small` at 17x the size. Its multi-vector and multilingual features are irrelevant to an English Corpus. |
| `gte-large`, `e5-large-v2` | Dominated by `bge-large-en-v1.5` at identical 335M/1024/512 (54.29 vs 52.22 / 50.56). |
| `all-mpnet-base-v2` | 43.81 BEIR-15 — below every v1.5-era model at the same size. The classic default, now clearly superseded. |
| `static-retrieval-mrl-en-v1` | Genuinely interesting as a speed floor: no active parameters, no sequence limit, 100–400x faster than mpnet on CPU, but only 0.5032 NanoBEIR mean. Worth a mention in the write-up as "how much nDCG does speed cost", not an Arm. |

---

## Appendix A — how the measurements were made

Everything marked **[measured]** was produced on the dev's own machine (Apple M4, 10 cores,
macOS 25.5.0) on 2026-09-17, against the real Corpus — `Shengtao/recipe`, `recipe.csv`,
32,722 rows, downloaded from the Hugging Face Hub.

- **Token lengths.** The raw `tokenizers` library, using each family's own `tokenizer.json`
  (`bge-base-en-v1.5` for BERT WordPiece, `gte-modernbert-base` for ModernBERT BPE,
  `Qwen3-Embedding-0.6B` for Qwen BPE, `bge-m3` for XLM-R). Every one of the 32,722 Recipes
  was tokenized per field; the thresholds are 254/510 tokens to leave room for `[CLS]`/`[SEP]`.
  Tokenizer choice barely matters — the four agree within ~13% on mean `directions` length
  (BERT 146.3, ModernBERT 142.7, Qwen 147.0, XLM-R 164.2 tokens).
- **CPU throughput.** `sentence-transformers` 6.0.1 / `torch` 2.14.0 / `transformers` 5.17.0.
  600 real Chunks (200 Recipes × description/ingredients/directions, seeded sample), batch 32,
  `max_seq_length` capped at 512, warm-up pass discarded, `torch.set_num_threads(10)`.
  Repeat runs of `bge-small` gave 154.3 and 152.0 Chunks/s, so run-to-run drift is under 2%
  and thermal throttling did not distort the ordering.
- **Sizes, licences, gating, parameter counts.** `https://huggingface.co/api/models/{id}?blobs=true`,
  summing `.safetensors` blob sizes. Note that the on-disk figure reflects the **shipped dtype** —
  the gte repos are half the size of their bge twins only because they are fp16.
- **Derived BEIR-15 averages (†).** Averaged `ndcg_at_10` over the 15 BEIR datasets from each
  model card's own `model-index`. Validated by reproducing three independently published
  figures exactly (e5-base-v2 50.29, bge-small 51.68, bge-base 53.25).

Caveat: throughput is one machine, one batch size, one dtype, PyTorch eager. ONNX or OpenVINO
int8 would change the absolute numbers substantially (sentence-transformers reports roughly
3–4x on CPU) but is unlikely to reorder models of the same architecture family.

## Appendix B — sources

Model cards (Hugging Face):
- https://huggingface.co/BAAI/bge-small-en-v1.5 · https://huggingface.co/BAAI/bge-base-en-v1.5 · https://huggingface.co/BAAI/bge-large-en-v1.5
- https://huggingface.co/intfloat/e5-small-v2 · https://huggingface.co/intfloat/e5-base-v2
- https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- https://huggingface.co/thenlper/gte-base (source of the 15-task Retrieval comparison table)
- https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5 · https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5 · https://huggingface.co/Alibaba-NLP/gte-modernbert-base
- https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
- https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1
- https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0 · https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0
- https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0/raw/main/config_sentence_transformers.json (the `"query: "` prefix, authoritative)
- https://huggingface.co/ibm-granite/granite-embedding-small-english-r2 · https://huggingface.co/ibm-granite/granite-embedding-english-r2
- https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- https://huggingface.co/google/embeddinggemma-300m (gated) · https://ai.google.dev/gemma/docs/embeddinggemma/model_card
- https://huggingface.co/BAAI/bge-m3
- https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1

Papers:
- Nomic Embed — https://arxiv.org/abs/2402.01613 (MTEB Retrieval 52.8; 137M params; 8192 ctx; Apache-2.0)
- Arctic-Embed 2.0 — https://arxiv.org/abs/2412.04506 (MTEB-R 0.554 / 0.556; MRL 256; base models)
- E5 — https://arxiv.org/abs/2212.03533
- MTEB — https://arxiv.org/abs/2210.07316

Other:
- https://huggingface.co/spaces/mteb/leaderboard
- https://sbert.net/docs/sentence_transformer/usage/efficiency.html and https://github.com/huggingface/sentence-transformers/releases/tag/v3.3.0 (CPU backend speedups: ~3x ONNX int8, ~4–5x OpenVINO int8)
- https://developers.googleblog.com/en/introducing-embeddinggemma/ (<200 MB RAM quantized, <15 ms/256 tokens on EdgeTPU)
- https://jina.ai/models/jina-embeddings-v3/ (CC-BY-NC-4.0, 570M, 8192 ctx, 1024 dims)
