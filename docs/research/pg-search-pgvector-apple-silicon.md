# `pg_search` + `pgvector` together on Apple Silicon

Research for [#4](https://github.com/dangth1101/IS6303/issues/4) (Milestone 1, part of [#1](https://github.com/dangth1101/IS6303/issues/1)).
Date: 2026-09-14.

**Verification status.** Everything in the "Verified locally" column was executed on this
machine — Apple M4, 16 GB, macOS (Darwin 25.5.0), Docker 29.6.2 (`Server.Arch = arm64`) —
against a live `paradedb/paradedb:0.25.9-pg17` container. Claims marked *docs only* come from
primary sources (ParadeDB repo/docs at tag `v0.25.9`, pgvector README) and were **not**
executed. Where the two disagreed, the live server wins and I say so.

---

## 1. Answer in one screen

```yaml
# docker-compose.yml — the pinned, arm64-verified set
services:
  db:
    image: paradedb/paradedb:0.25.9-pg17@sha256:8da5d202fe31875af32a49e802ce17cfbc146b0672306f4109809efee1e6f916
    platform: linux/arm64
    shm_size: 1g            # REQUIRED — see §4.2, parallel HNSW build fails on the 64 MB default
    environment:
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: recipes
    ports: ["5432:5432"]
    volumes: ["pgdata:/var/lib/postgresql/data"]
volumes: { pgdata: }
```

| Component | Version | Source |
| --- | --- | --- |
| ParadeDB / `pg_search` | **0.25.9** (single version; no separate `pg_search` number) | verified: `SELECT * FROM paradedb.version_info()` → `0.25.9 / release` |
| PostgreSQL | **17.11** (Debian 17.11-1.pgdg13+2), `aarch64-unknown-linux-gnu` | verified: `SELECT version()` |
| `pgvector` | **0.8.4** — preinstalled, and a *required dependency* of `pg_search` since 0.25.0 | verified: `pg_extension.extversion` |
| Image digest (index) | `sha256:8da5d202…1e6f916` | `docker buildx imagetools inspect` |
| Image digest (arm64 manifest) | `sha256:e1cf12e0f31645ec9a40f60cc5c621aca4ae144d8ba0c7c0f02eaa772c63bfaf` | same |
| Image size | 355 MB, `Architecture: arm64` | `docker image inspect` |

**Coexistence: settled, and better than the ticket assumed.** Both extensions are not merely
compatible — they arrive **already created** in the image's default database. No
`CREATE EXTENSION` is strictly needed, and no Postgres version constraint pins them against
each other (ParadeDB supports PG 15–18; pgvector supports PG 13+).

```
        extname         | extversion            -- observed, unmodified container
------------------------+------------
 fuzzystrmatch          | 1.2
 pg_ivm                 | 1.13
 pg_search              | 0.25.9
 pg_stat_statements     | 1.11
 plpgsql                | 1.0
 postgis                | 3.6.4
 postgis_tiger_geocoder | 3.6.4
 postgis_topology       | 3.6.4
 vector                 | 0.8.4
```

`shared_preload_libraries = 'pg_search,pg_cron,pg_stat_statements'` is preset. `pgvector`
needs no preload. Also bundled: PostGIS 3.6.4, pg_cron 1.6.7, pg_ivm 1.13 — dead weight for
us, but harmless.

**arm64: native, no emulation.** Every one of the 100 most-recently-updated Docker Hub tags
publishes both `linux/amd64` and `linux/arm64` manifests. The repo Dockerfile hard-fails on any
third architecture (per-arch `.deb` checksums for exactly `amd64` and `arm64`), so arm64 is a
first-class target, not a courtesy build. Confirmed at runtime: `uname -m` → `aarch64`,
`Architecture: arm64`, and `version()` reports `aarch64-unknown-linux-gnu`. **No Rosetta/QEMU
emulation, therefore no emulation tax on indexing** — see the measured build times in §4.

> Use Docker Hub, **not** GHCR. `ghcr.io/paradedb/paradedb` does not exist —
> `https://ghcr.io/v2/paradedb/paradedb/tags/list` returns `NAME_UNKNOWN`.

---

## 2. `pg_search` BM25: DDL, query, scores

### 2.1 The API changed twice; use the 0.25.x form

The ticket's framing and most material on the web are stale. Three generations exist:

| Generation | Form | Status at 0.25.9 |
| --- | --- | --- |
| ≤ 0.12 | `CALL paradedb.create_bm25(...)` procedure | **removed in 0.13.0** ([PR #1915](https://github.com/paradedb/paradedb/pull/1915)) |
| 0.13 – 0.24 | `USING bm25 (...) WITH (key_field, text_fields='{json}')` | works, undocumented legacy; reloptions still registered |
| **0.25.x** | **`USING paradedb (...)` + cast-based tokenizer typmods** | **current — use this** |

Both access-method names resolve today (`USING bm25` is a back-compat alias; verified —
`USING paradedb` yields `pg_am.amname = 'paradedb'`). Both `pdb.score()` and
`paradedb.score()` resolve and return **byte-identical** scores (verified). Prefer
`USING paradedb` and `pdb.*`: they are what the current docs describe.

### 2.2 Index DDL — the shape the schema ticket should build on

```sql
CREATE INDEX recipes_search_idx ON recipes USING paradedb (
  id,                                                          -- key_field, MUST be first
  (title::pdb.simple('stemmer=english','stopwords_language=english')),
  (description::pdb.simple('stemmer=english','stopwords_language=english')),
  (ingredients::pdb.simple('stemmer=english','stopwords_language=english')),
  (doc_text::pdb.simple('stemmer=english','stopwords_language=english')),
  category,          -- non-text fields are columnar-indexed automatically
  rating
) WITH (key_field = 'id');
```

Hard rules, all verified or from `docs/documentation/indexing/create-index.mdx`:

- **The cast must be parenthesised.** `USING paradedb (id, doc_text::pdb.simple('...'))` is a
  **syntax error** (`syntax error at or near "::"`). It must be
  `(doc_text::pdb.simple('...'))`. This cost me a cycle; it is not in the docs' examples
  prominently.
- `key_field` must be first in the column list, must be `UNIQUE` (normally the PK), and must be
  untokenized if text.
- **Only one ParadeDB index per table.** Verified: a second one errors with
  `a relation may only have one ParadeDB index`. **Design consequence:** every column the
  sparse arm needs — for matching, filtering, sorting, *or* faceting — must go in this single
  index. There is no "add another index later" escape hatch.
- Use an **integer/bigint** key. `NUMERIC` key fields and columns are a live minefield: seven
  open bugs filed 2026-08-26 (#6100–#6108) covering rejected typmods, rounded pushdown
  literals, and silently-empty result sets.

### 2.3 Query syntax — five operators

At least one ParadeDB operator must appear or the index is not used.

| Operator | Meaning | Tokenizes the query? |
| --- | --- | --- |
| `@@@` | general search predicate (query string, or a `pdb.*` builder) | yes |
| `\|\|\|` | match **disjunction** — any of the terms | yes |
| `&&&` | match **conjunction** — all of the terms | yes |
| `===` | **term** — exact token match | **NO** |
| `###` | **phrase** — tokens in order | yes |

> **Sharp edge I found empirically, and it is not documented.** `===` bypasses the field
> tokenizer, so on a stemmed field you must supply the *already-stemmed* form. On a field
> indexed with `stemmer=english` over 32,722 rows:
>
> ```
> WHERE doc_text === 'chocolate'   -->     0 rows   -- raw term, never matches
> WHERE doc_text === 'chocol'      -->  2181 rows   -- stemmed form matches
> WHERE doc_text ||| 'chocolate'   -->  2181 rows   -- tokenizes, so it just works
> ```
>
> **For the sparse arm, use `@@@` or `|||`, never `===`.** A `===` arm on a stemmed field would
> silently return near-nothing and look like a catastrophic BM25 result.

### 2.4 How scores are returned

`pdb.score(<key_field>)` returns the BM25 score as a `real`. There is no `rank_bm25` in the
current codebase (zero hits in source or docs) — ignore any material that mentions it.

```sql
SELECT id, title, pdb.score(id) AS bm25
FROM recipes
WHERE doc_text @@@ 'chocolate cake'
ORDER BY pdb.score(id) DESC, id ASC     -- id tiebreak keeps ordering stable AND keeps Top-K
LIMIT 10;
```

Verified plan — a dedicated Top-K executor, not a sort-everything scan:

```
 Limit
   ->  Custom Scan (ParadeDB Base Scan) on recipes
         Index: recipes_bm25
         Exec Method: TopKScanExecState
         Scores: true
            TopK Order By: pdb.score() desc
            TopK Limit: 5
         Tantivy Query: {"with_index":{"query":{"parse_with_field":{...}}}}
```

Two caveats that matter for measurement:

- A field must be **in the index** to be factored into the score.
- **`VACUUM` before you measure anything.** Per `docs/documentation/sorting/score.mdx`, dead
  rows still influence BM25 until vacuumed — IDF drifts. For a graded, reproducible eval, run
  `VACUUM ANALYZE` after ingest and after any bulk update, and record that you did.

Other retrieval affordances, all verified working: `pdb.snippet(field)` returns `<b>`-tagged
highlights; `paradedb.boolean(should => ARRAY[paradedb.boost(2.0, ...), ...])` gives per-field
weighting (a title-2× boost reordered my test set as expected); `pdb.parse('doc_text:(chocolate
cake) AND rating:>3')` supports Lucene-ish field syntax with numeric filters.

### 2.5 `k1` and `b` — **configurable, per field, and I proved it**

This was the ticket's critical question. The docs are nearly silent: `k1`/`b` appear in
**exactly one line** of the entire docs tree (the 0.23.0 changelog, "Per-field tunable BM25
`k1` and `b` parameters via typmod") with **no reference page**. So I established it live.

**Defaults: `k1 = 1.2`, `b = 0.75`.** Confirmed two independent ways:

1. Source — ParadeDB's pinned Tantivy fork, `src/query/bm25.rs`: *"Defaults to
   `Bm25Params::DEFAULT` (`k1 = 1.2`, `b = 0.75`)"*.
2. **Behaviourally**: an index built with explicit `'k1=1.2','b=0.75'` and one built with no
   `k1`/`b` at all produced **byte-identical scores** (`1.3378724`, `1.2232435`) on the same
   query. That is what "the defaults are 1.2/0.75" actually means, demonstrated.

**They are settable, and they change ranking.** Same query, same data, only `k1`/`b` differing:

| Index config | doc 1 score | doc 5 score |
| --- | --- | --- |
| default (≡ `k1=1.2, b=0.75`) | 1.3378724 | 1.2232435 |
| `'k1=0.1','b=0.0'` | 0.9319507 | 0.9171578 |

Introspection confirms the values land in the schema — `paradedb.index_fields('...')` shows
`"b": 0.75, "k1": 1.2000000476837158` (stored as `f32`, hence the float wobble), versus
`"b": null, "k1": null` when unset.

Syntax (verified):

```sql
CREATE INDEX ... USING paradedb (
  id, (doc_text::pdb.simple('stemmer=english', 'k1=1.2', 'b=0.75'))
) WITH (key_field='id');
```

**Valid ranges** (`pg_search/src/api/tokenizers/typmod/validation.rs`): `k1` ∈ [0.0, 100.0],
`b` ∈ [0.0, 1.0].

**They are index-time, not query-time.** Baked into the Tantivy schema; there is no per-query
override. Changing them counts as "changing a field's tokenizer" and therefore **requires a
full index rebuild** via `CREATE INDEX CONCURRENTLY` + swap. Budget for that if index tuning
(Milestone 5) wants to sweep `k1`/`b` — each point in the sweep is a rebuild, not a `SET`.

The complete set of accepted per-field typmod options, obtained by deliberately passing a bogus
one (the most useful error message in the system):

```
ERROR:  Invalid option: 'nonsense_option'. Allowed options: alias, alpha_num_only,
ascii_folding, b, columnar, fieldnorms, k1, lowercase, normalizer, remove_long,
remove_short, stemmer, stopwords, stopwords_language, trim.
```

### 2.6 English tokenizer / stemmer configuration

Default tokenizer is `pdb.unicode_words` (UAX #29 word boundaries, lowercased) with
**no stemmer and no stopwords** — verified: a fresh index reports
`"tokenizer": {"UnicodeWords": {...}}` with `"stemmer": null, "stopwords": null`. **Sparse
retrieval over recipes without stemming would be a self-inflicted wound**: `cookie` would not
match a recipe titled "Cookies".

Demonstrated, on a field with vs. without `stemmer=english`:

| Query | no stemmer | `stemmer=english` |
| --- | --- | --- |
| `cookies` | 1 hit | 1 hit |
| `cookie` | **0 hits** | **1 hit** |

Inspect tokenization directly without building an index — the single best debugging affordance
here (verified output):

```sql
SELECT 'I am running cookies'::pdb.simple::text[],                        -- {i,am,running,cookies}
       'I am running cookies'::pdb.simple('stemmer=english')::text[],      -- {i,am,run,cooki}
       'The cat in the hat'::pdb.simple('stopwords_language=english')::text[]; -- {cat,hat}
```

Note `cookies → cooki`: Snowball English is aggressive and produces non-words. That is normal
and harmless (query and document pass through the same filter) — but it is why `===` fails
(§2.3) and why you should never eyeball the index for readable terms.

**Recommended config for this project:**

```sql
(doc_text::pdb.simple('stemmer=english','stopwords_language=english'))
```

- `stemmer=english` — Snowball. Valid languages include arabic, czech, danish, dutch, english,
  finnish, french, german, greek, hungarian, italian, norwegian, polish, portuguese, romanian,
  russian, spanish, swedish, tamil, turkish.
- `stopwords_language=english` — verified applied (`"stopwords_language": ["English"]`). A
  separate `stopwords` key takes an explicit list.
- `lowercase` is **on by default**; pass `lowercase=false` to disable.
- Consider `ascii_folding=true` if the corpus has accented ingredient names (*sauté*, *crème*).
- `pdb.alias(...)` allows **multiple tokenizers on one field** (e.g. raw *and* stemmed), and
  `search_tokenizer` allows a different tokenizer at query time. Both are escape hatches if the
  stemmed-only arm underperforms; neither is needed to start.

---

## 3. `pgvector` 0.8.4: HNSW vs IVFFlat

Image ships **0.8.4**; upstream latest is **0.8.6** (2026-07-29). The 0.8.x line is pure
bug-fix — no feature or default changed since 0.8.0 — so all tuning guidance below applies
unchanged. Two of the fixes we skip are worth knowing: 0.8.4 fixed an HNSW
`graph not repaired` vacuum bug and IVFFlat builds exceeding `maintenance_work_mem`; 0.8.6
fixed an IVFFlat 32-bit build buffer overflow. None blocks us at 32,722 rows, but if a vacuum
or build misbehaves, an image bump is the first thing to try.

### 3.1 Dimensionality ceilings

The column ceiling and the *indexable* ceiling are different numbers — the trap.

| Type | Column max | HNSW-indexable | IVFFlat-indexable | Bytes each |
| --- | --- | --- | --- | --- |
| `vector` (float32) | 16,000 dims | **2,000** | **2,000** | `4·d + 8` |
| `halfvec` (float16) | 16,000 dims | **4,000** | **4,000** | `2·d + 8` |
| `bit` | — | 64,000 | 64,000 | `d/8 + 8` |
| `sparsevec` | 16,000 nnz | 1,000 nnz | *unsupported* | `8·nnz + 16` |

Confirmed in source: `HNSW_MAX_DIM 2000`, `IVFFLAT_MAX_DIM 2000`.

**Relevance to us: no ceiling problem.** Every candidate text/multimodal encoder — CLIP (512),
SigLIP (768), `bge`/`e5` (768/1024) — sits far under 2,000. Plain `vector(d)` +
`vector_cosine_ops` works with no `halfvec` gymnastics. If a future model exceeded 2,000, the
escape is index-on-cast (column stays full precision, index is halved):

```sql
CREATE INDEX ON items USING hnsw ((embedding::halfvec(3072)) halfvec_l2_ops);
SELECT * FROM items ORDER BY embedding::halfvec(3072) <-> '[...]' LIMIT 10;  -- cast must repeat
```

### 3.2 Parameters and defaults

**HNSW** — `CREATE INDEX ... USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64);`

| Parameter | Default | Range | When |
| --- | --- | --- | --- |
| `m` — max connections per layer | **16** | 2–100 | build |
| `ef_construction` — build candidate list | **64** | 4–1000 | build |
| `hnsw.ef_search` — search candidate list | **40** | 1–1000 | query (`SET`/`SET LOCAL`) |
| `hnsw.iterative_scan` | `off` | `strict_order` \| `relaxed_order` | query, 0.8.0+ |
| `hnsw.max_scan_tuples` | 20,000 | — | query, iterative scans |

**IVFFlat** — `CREATE INDEX ... USING ivfflat (embedding vector_cosine_ops) WITH (lists=32);`

| Parameter | Default | Recommendation |
| --- | --- | --- |
| `lists` | 100 | `rows/1000` up to 1M rows; `sqrt(rows)` above |
| `ivfflat.probes` | **1** | `sqrt(lists)` — the default of 1 is far too low |

**IVFFlat requires a populated table** (k-means training step). Building it on an empty or
sparse table silently costs recall; the README's own troubleshooting entry is *"Why are there
less results for a query after adding an IVFFlat index? … Drop the index until the table has
more data."* HNSW has no such requirement — an important operational difference given we will
re-ingest during development.

### 3.3 The trade-off, in the README's own words

> **HNSW** — *"has better query performance than IVFFlat (in terms of speed-recall tradeoff),
> but has slower build times and uses more memory. Also, an index can be created without any
> data in the table since there isn't a training step like IVFFlat."*
>
> **IVFFlat** — *"faster build times and uses less memory than HNSW, but has lower query
> performance (in terms of speed-recall tradeoff)."*

### 3.4 Distance operators

| Operator | Distance | Op class |
| --- | --- | --- |
| `<->` | L2 / Euclidean | `vector_l2_ops` |
| `<=>` | **cosine** | `vector_cosine_ops` |
| `<#>` | negative inner product | `vector_ip_ops` |
| `<+>` | L1 / taxicab | `vector_l1_ops` (HNSW only) |
| `<~>` / `<%>` | Hamming / Jaccard | `bit_*_ops` |

For **normalized** embeddings the README prefers inner product: *"If vectors are normalized to
length 1 (like OpenAI embeddings), use inner product for best performance"* — `<#>` with
`vector_ip_ops`. `<#>` returns the *negative* inner product, so ascending order is still
"nearest". Cosine (`<=>` + `vector_cosine_ops`) is the safer default and what I verified; on
unit-length vectors the two rank identically, so this is a performance choice, not a
correctness one. Note **zero vectors are not indexed for cosine distance**, and `NULL` vectors
are never indexed — worth an ingest assertion.

**Index-usage rule that bites hybrid queries:** the `ORDER BY` must be a bare distance operator
in ascending order. `ORDER BY 1 - (embedding <=> '...') DESC` gets **no index**. Compute
similarity in the `SELECT` list, sort by raw distance.

### 3.5 Sizing for 32,722 rows

| dims | `4·d+8` bytes | × 32,722 | as `halfvec` |
| --- | --- | --- | --- |
| 512 | 2,056 | ~64 MiB | ~32 MiB |
| 768 | 3,080 | ~96 MiB | ~48 MiB |
| 1024 | 4,104 | ~128 MiB | ~64 MiB |

IVFFlat: `lists = 32722/1000 ≈ 32`, `probes = sqrt(32) ≈ 6`. **Do not leave `probes` at 1** —
that visits ~1/32 of the corpus and will make the dense arm look far worse than it is. This is
the single most likely way to accidentally sandbag the dense arm.

HNSW: defaults are fine at this scale (the README: *"use the defaults unless seeing low
recall"*). Everything fits comfortably in 16 GB.

---

## 4. Measured on this machine (M4, 16 GB, native arm64)

32,722 synthetic recipe-shaped documents (~35 tokens each) plus 32,722 × 768-dim vectors —
matching the real corpus row count. Table 140 MB.

| Operation | `shm_size=1g` | default 64 MB shm |
| --- | --- | --- |
| Insert 32,722 rows | 5.8 s | 8.2 s |
| **BM25 index build** (`stemmer=english`) | **0.34 s** | 0.61 s |
| **HNSW build** (768-dim, cosine, defaults) | **2.1 s** | **FAILS** → 4.8 s serial |
| **IVFFlat build** (`lists=32`) | **1.4 s** | 1.6 s |
| BM25 query, Top-10 | **0.8 – 2.9 ms** | 1.3 – 3.4 ms |
| RRF fusion query (both arms, k=60) | **12.5 ms** | — |

Index sizes: BM25 4.4 MB; IVFFlat 128 MB; HNSW 17 MB.

**Read the build times as the arm64 verdict: seconds, not minutes.** There is no emulation and
no indexing tax. The ticket's worry about "what emulation costs in indexing time" is moot —
indexing the full corpus is a rounding error, and index tuning in Milestone 5 can iterate
freely. The real Milestone 2 cost will be **embedding generation and image fetching**, not
Postgres.

> Caveat on the two vector index sizes: my synthetic vectors are highly repetitive, so the
> 17 MB HNSW figure is implausibly low against the ~96 MiB of raw 768-dim data and I do not
> trust it. IVFFlat's 128 MB matches expectation. Re-measure both with real embeddings via
> `pg_size_pretty(pg_relation_size('idx'))`. Build *times* are unaffected by this caveat.

### 4.1 One-ParadeDB-index-per-table, demonstrated

```
ERROR:  a relation may only have one ParadeDB index
```

### 4.2 `shm_size` is mandatory for parallel HNSW builds

On a default container, `CREATE INDEX ... USING hnsw` **fails outright**:

```
ERROR:  could not resize shared memory segment "/PostgreSQL.3084209676"
        to 533811360 bytes: No space left on device
```

Cause, confirmed: `df -h /dev/shm` → **64 MB**, Docker's default. pgvector's parallel build
wants ~509 MB of shared memory. Two fixes, both verified:

- **`shm_size: 1g` in compose** (recommended) — `/dev/shm` becomes 1.0 GB, parallel build
  succeeds in **2.1 s**.
- `SET max_parallel_maintenance_workers = 0` — serial build succeeds in **4.8 s**, 2.3× slower.

This is the single most likely thing to break a marker's reproduction: the error is opaque, it
mentions neither Docker nor `shm`, and it only appears once vectors exist. **`shm_size: 1g`
must be in the committed compose file.**

---

## 5. Hybrid / RRF: the query shape that works

Verified end to end at 32,722 rows in **12.5 ms**. Both arms in one statement, ranks fused by
RRF (`k = 60`):

```sql
WITH sparse AS (
  SELECT id, ROW_NUMBER() OVER (ORDER BY s DESC, id) AS rnk
  FROM (SELECT b.id, pdb.score(b.id) AS s
        FROM bench b
        WHERE b.doc_text @@@ 'chocolate cake'
        ORDER BY s DESC, b.id
        LIMIT 60) t
),
dense AS (
  SELECT id, ROW_NUMBER() OVER (ORDER BY dist, id) AS rnk
  FROM (SELECT b.id, b.embedding <=> $1::vector AS dist
        FROM bench b
        ORDER BY dist, b.id
        LIMIT 60) t
)
SELECT COALESCE(s.id, d.id) AS id,
       s.rnk AS sparse_rank, d.rnk AS dense_rank,
       COALESCE(1.0/(60 + s.rnk), 0) + COALESCE(1.0/(60 + d.rnk), 0) AS rrf
FROM sparse s FULL OUTER JOIN dense d ON s.id = d.id
ORDER BY rrf DESC, id
LIMIT 10;
```

`FULL OUTER JOIN` is the right join: a document found by only one arm must still be ranked
(with `COALESCE(..., 0)` for the missing side). That is also exactly the pooling the qrels
protocol needs.

> **Gotcha, found the hard way.** Threading the query *through a CTE* and referencing it as a
> column breaks the planner:
>
> ```sql
> WITH q AS (SELECT 'chocolate cake' AS qtext, ... AS qvec)
> SELECT ... FROM bench b, q WHERE b.doc_text @@@ q.qtext   -- ERROR
> ```
> ```
> ERROR:  Unsupported query shape. Please report at https://github.com/paradedb/paradedb/issues/new/choose
> ```
>
> The `@@@` right-hand side must be a **literal or a bind parameter**, not a column reference
> from a join. Verified working: a literal, and a `PREPARE sp(text) AS ... WHERE doc_text @@@ $1`
> bind parameter. **For the FastAPI layer this is good news** — pass the query as a normal
> `$1` parameter (which is what any driver does anyway) and it works; just don't get clever
> with CTE-based query plumbing.

Also verified: `pdb.score()` inside a `ROW_NUMBER() OVER (...)` works, but the shape above
(score in an inner subquery, rank outside) is the one I'd commit to — it is the most robust and
keeps the `LIMIT 60` pushed into the Top-K executor.

pgvector itself provides **no** fusion primitive; the README's hybrid section just says *"use
together with Postgres full-text search"* and links Python RRF examples. Fusion is our SQL or
our Python — the above shows SQL is entirely viable.

---

## 6. Licence and maturity

- **`pg_search` is AGPL-3.0-or-later.** One `LICENSE` at the repo root (full AGPLv3), `license
  = "AGPL-3.0"` in `Cargo.toml`, and the shipped Debian package's copyright file confirms
  `AGPL-3.0-or-later`. **No BSL, no Elastic licence, no per-directory relicensing.**
- A proprietary **ParadeDB Enterprise** exists but is not in this repo. It waives copyleft and
  adds closed features. Community-vs-Enterprise gaps: **no HA, no read replicas, max cluster
  size 1**; ParadeDB indexes exist only on the primary since Community lacks physical
  replication. BM25 scoring, query builder, Top-K, highlighting and MVCC-safety are all in
  Community.
- **For this project the AGPL is a non-issue**: we run it locally, unmodified, and distribute
  nothing. Worth one sentence in the write-up's limitations, not an ADR.
- **Maturity: active but pre-1.0.** 9,260 stars, created 2023-06-30, ~weekly patch releases
  (0.25.0 → 0.25.9 across 2026-07-28 → 2026-09-11), minors every 4–8 weeks. Still `0.25.x`.
  **Pin the exact version** — at this cadence, `latest` will move under us mid-project, and
  `latest` also silently tracks the newest PG major (18 today).
- `pgvector` 0.8.x is mature and stable; note it publishes **no GitHub Releases**, only git
  tags plus `CHANGELOG.md`.
- **Beta subsystems to avoid.** `pg_search` 0.25.0 added *native* vector search inside the
  ParadeDB index (`USING paradedb (id, description, embedding vector_cosine_ops)`), explicitly
  **beta**: *"future releases may change the vector storage format and require a reindex."*
  There is an open panic against it (#6076, vector segment merge in superkmeans). **Use
  `pgvector`'s own HNSW/IVFFlat indexes, not ParadeDB's native vector search.** This also keeps
  the dense arm's index independently tunable, which the one-index-per-table rule would
  otherwise prevent.

---

## 7. Sharp edges checklist

Operational, in rough order of how likely each is to cost us time:

1. **`shm_size: 1g`** or HNSW builds fail with an opaque `No space left on device` (§4.2).
2. **`===` on a stemmed field returns ~nothing** — use `@@@` or `|||` (§2.3).
3. **`VACUUM ANALYZE` before every measurement.** Dead rows perturb BM25 scores and stale
   visibility maps slow reads. A `Parallel Custom Scan` with many `Heap Fetches` means vacuum.
   Consider `ALTER TABLE recipes SET (autovacuum_vacuum_threshold = 500)`.
4. **One ParadeDB index per table** — index every column the sparse arm will ever touch, up
   front (§2.2).
5. **`ivfflat.probes` defaults to 1** — must be raised to ~6, or the dense arm is sandbagged.
6. **Changing `k1`/`b`/tokenizer requires a full rebuild** via `CREATE INDEX CONCURRENTLY` +
   drop. Rebuild triggers: adding/removing an index field, renaming an indexed column,
   changing a tokenizer. Plain `REINDEX` takes an exclusive lock (blocks writes, not reads).
7. **Build the indexes *after* bulk ingest.** ParadeDB packs a tighter on-disk representation
   and builds faster; IVFFlat *requires* populated data for k-means.
8. **Avoid `NUMERIC`** columns, especially as `key_field` — seven open correctness bugs
   (#6100–#6108). Use `bigint` keys and `real`/`double precision` for ratings.
9. **Approximate-index filtering happens *after* the index scan.** With `hnsw.ef_search = 40`
   and a `WHERE` matching 10% of rows, you get ~4 results. If the UI adds category filters,
   enable `SET hnsw.iterative_scan = strict_order` (0.8.0+) or use a partial index.
10. **Write amplification.** Every `INSERT`/`UPDATE`/`COPY`/`VACUUM` runs segment compaction.
    Default background layer sizes `100KB, 1MB, 100MB, 1GB, 10GB`, tunable via
    `ALTER INDEX ... SET (background_layer_sizes = '...')`. `mutable_segment_rows` (default
    1000) trades write throughput against read RAM. Mostly irrelevant for a
    bulk-load-once corpus — relevant if we re-ingest repeatedly.
11. **Memory for large builds.** `maintenance_work_mem` ≥ 64 MB *per parallel worker* (≥15 MB
    hard minimum, else error); `work_mem` below 15 MB is ignored and forced to 15 MB. Immaterial
    at our scale.
12. **The image auto-tunes `postgresql.conf`** from detected CPU/RAM (`shared_buffers` = 25% of
    RAM, `maintenance_work_mem` = RAM/16 capped 2 GB). Disable with `-e PDB_TUNE=false` **at
    first run only**. For reproducibility on a marker's different machine, either set
    `PDB_TUNE=false` and pin settings explicitly, or note that config is host-dependent.
13. `CREATE INDEX CONCURRENTLY` needs the session to stay open — a pooler's idle timeout will
    cancel it and leave an invalid index to drop by hand.
14. **The repo's checked-in Dockerfile at tag `v0.25.9` installs the 0.25.8 `.deb`** (a
    tagging-order artifact; Dockerfiles are regenerated at publish time). The *published image*
    is correct — verified `0.25.9` at runtime. Don't build from the repo tree expecting 0.25.9.

---

## 8. Recommendations for dependent tickets

**Schema ticket.** Take §2.2 verbatim: `bigint` PK as `key_field`, one `USING paradedb` index
covering `id, title, description, ingredients, doc_text, category, rating`, each text field
`::pdb.simple('stemmer=english','stopwords_language=english')`; leave `k1`/`b` at defaults.
Separate `vector(d)` column with its own HNSW index (`vector_cosine_ops`, defaults) — not
ParadeDB's beta native vector search. Build all indexes after ingest, then `VACUUM ANALYZE`.

**Backend ticket.** Sparse arm: `@@@` with the query as a bind parameter, `ORDER BY
pdb.score(id) DESC, id ASC LIMIT k`. Dense arm: `ORDER BY embedding <=> $1 LIMIT k` with
`SET LOCAL hnsw.ef_search = 100`. RRF arm: §5. Never build the query via a CTE column.

**Evaluation ticket.** Index rebuilds are seconds at this scale, so sweeping `k1`/`b`,
`ef_search`, `m`/`ef_construction` and `probes` is cheap — but `k1`/`b` sweeps mean a rebuild
per point, not a `SET`. `VACUUM ANALYZE` between runs and record it.

**ADR candidates.** (a) `pgvector` HNSW over ParadeDB native vector search — beta storage
format vs. independent tunability, plus the one-index-per-table constraint. (b) English
stemming on by default for the sparse arm — it changes what "BM25" means in our comparison and
`===` behaves surprisingly because of it. Both are non-obvious and hard to reverse.

---

## 9. Primary sources

- ParadeDB repo, tag `v0.25.9` — https://github.com/paradedb/paradedb
  - `docs/documentation/indexing/create-index.mdx`, `.../reindexing.mdx`, `.../columnar.mdx`
  - `docs/documentation/sorting/score.mdx`, `docs/documentation/filtering.mdx`
  - `docs/documentation/tokenizers/overview.mdx`, `docs/documentation/token-filters/*.mdx`
  - `docs/documentation/performance-tuning/{create-index,reads,writes,overview}.mdx`
  - `docs/changelog/0.25.0.mdx` (pgvector required; vector search beta), `0.23.0.mdx` (k1/b),
    `0.19.0.mdx` / `0.19.5.mdx` (`paradedb.*` → `pdb.*`)
  - `pg_search/src/api/tokenizers/typmod/validation.rs` (k1/b ranges),
    `pg_search/src/schema/config.rs` (`apply_bm25`),
    `pg_search/src/api/operator/atatat.rs` (`@@@` overloads)
  - `docker/Dockerfile.paradedb-17` (bundled extension pins), `docs/deploy/enterprise.mdx`
  - Tantivy fork `src/query/bm25.rs` — `k1 = 1.2`, `b = 0.75` defaults
- ParadeDB releases — https://github.com/paradedb/paradedb/releases
- Docker Hub tags API — `https://hub.docker.com/v2/repositories/paradedb/paradedb/tags`
- pgvector README (canonical docs) — https://github.com/pgvector/pgvector/blob/master/README.md
  - `CHANGELOG.md`; `src/hnsw.h`, `src/ivfflat.h`, `src/hnsw.c`, `src/ivfflat.c` @ `v0.8.6`
- Live verification: `paradedb/paradedb:0.25.9-pg17` on Apple M4 / Docker 29.6.2 arm64,
  2026-09-14.
