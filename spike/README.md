# Spike: end-to-end dense retrieval demo

Throwaway code for issue #12. **Not** the real ingest pipeline or backend — no Postgres,
no BM25, no RRF, no reranking, no ANN index. It exists to prove the pieces compose and
that results look sensible to a human.

## Run it

```sh
cd spike
uv run --project . python build_index.py --sample 2000   # ~35s after model download
uv run --project . uvicorn api:app --port 8711           # ~30s to load the model
```

Then:

```sh
curl -s localhost:8711/health
curl -s -G localhost:8711/search --data-urlencode "q=vegetarian chickpea curry" --data-urlencode "k=5"
```

`build_index.py --sample 0` would index the full 32,722-row corpus (~9 min, see below).

## What it does

1. Downloads `recipe.csv` (~64 MB) from the Hugging Face hub.
2. Takes a **uniform random** sample (fixed seed). Not evenly-spaced offsets — issue #3
   found null-image rows cluster in blocks, so systematic sampling misrepresents this corpus.
3. Builds the document string settled while charting: `title + description + ingredients`,
   with `directions` excluded. Built in exactly one place so every future arm sees the
   identical text.
4. Embeds with `nomic-embed-text-v1.5` (768-dim, L2-normalised) and writes
   `index/vectors.npy` + `index/docs.jsonl`.
5. Serves `GET /search?q=&k=` — brute-force cosine similarity over the whole sample.

## Measured on an Apple M4 / 16 GB

| | |
|---|---|
| Encoding throughput | **58.9 docs/s** (2,000 docs in 34.0 s), batch size 32, MPS |
| Projected full corpus | **~9 min** for all 32,722 |
| Query latency | 13 ms warm, 115–280 ms on early calls |
| Index size | 2,000 × 768 float32 = 5.9 MB |

## Findings worth carrying forward

- **`transformers` must be pinned `<5`.** `nomic-embed-text-v1.5`'s remote code calls
  `get_extended_attention_mask`, removed in transformers 5.x; without the pin it fails at
  `forward()` with an `AttributeError`. Research had flagged this for the *runner-up*
  models — it applies to the primary one too.
- **Encoding is ~5× faster than the research estimate** (58.9/s here vs 11.1/s in #2),
  probably batch size. The full-corpus embed is ~9 minutes, not ~49 — so re-embedding
  the whole corpus is cheap enough to do casually.
- **Query encoding dominates query latency**, not the similarity search. The brute-force
  scan over 2,000 × 768 is negligible; at full corpus scale the scan is still likely
  smaller than the ~10 ms encode. Relevant to whether an ANN index earns its complexity.
- **Dense retrieval visibly works.** `vegetarian chickpea curry` → *Vegan Sweet Potato
  Chickpea Curry* (0.820), *Curry Chickpea Salad*, *Vegetarian Bean Curry*.
  `spicy noodle soup` → *Khao Soi Soup* — a semantic hit no lexical match would find,
  since "Khao Soi" shares no term with the query.
- **A limitation the demo makes concrete:** `quick chicken dinner` returned *Rapid Chicken
  Stock* — semantically "quick" and "chicken", but stock is not a dinner, and the model has
  no access to the structured `total_time` field. Embedding text alone cannot honour
  numeric constraints. Worth raising in the schema ticket (#8) and in the write-up.
