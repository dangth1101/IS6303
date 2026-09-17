# Research: hybrid search in ParadeDB — capabilities and Fusion algorithms

Resolves [#17](https://github.com/dangth1101/IS6303/issues/17). Part of [#13](https://github.com/dangth1101/IS6303/issues/13).
Date: 2026-09-17.

**Verification status: documentary, not executed.** Every claim below is sourced to a primary
document — the ParadeDB repo at `main` (post-`v0.25.9`), the pgvector README and source at
`master`, the PostgreSQL 17 manual, or the original IR papers. Nothing here was run against a
live server in this session.

**Baseline.** [`docs/research/pg-search-pgvector-apple-silicon.md`](pg-search-pgvector-apple-silicon.md)
(branch `research/pg-search-pgvector-apple-silicon`, commit `33f1e3f`) was *executed* against
`paradedb/paradedb:0.25.9-pg17` on the dev's M4. It already establishes: exact versions,
index DDL, the five query operators, how BM25 scores are returned, that `k1`/`b` are per-field
configurable, tokenizer/stemmer config, HNSW vs IVFFlat parameters and dimensionality ceilings,
distance operators, corpus sizing, a working RRF query shape (12.5 ms at 32,722 rows), the
mandatory `shm_size: 1g`, and a 14-item sharp-edges checklist. **That document is the baseline
and wins on anything it measured.** This one cites it rather than repeating it, and spends its
length on the four things it does not cover: many vectors per Chunk at differing dimensions,
pre- vs post-filtering against an HNSW scan, the Fusion algorithms themselves, and what the
0.25.x native vector index changes.

**Corpus scale used throughout: 32,722 Recipes** (the datasets-server `/size` endpoint's exact
figure; `32,700` in the ticket bodies is the dataset viewer's rounded display) **producing
~98,000+ Chunks**, each embedded by *several* models.

---

## Answers in one screen

| # | Question | Answer |
|---|---|---|
| A1 | Raw BM25 scores? | Yes — `pdb.score(<key_field>)`. |
| A1 | Native Fusion/RRF helper? | **No.** There is no RRF function, aggregate or operator. ParadeDB *documents* the hand-written CTE and calls it the recommended approach. |
| A2 | HNSW vs IVFFlat at ~98k Chunks | HNSW, defaults. IVFFlat's only wins (build time, memory) are irrelevant at this size; its `probes = 1` default is a live footgun. |
| A3 | Many vectors per Chunk, differing dims | One row per `(chunk, model)` with a **dimensionless `vector` column**, plus **one partial expression index per model**. pgvector forces this: the type is dimension-typed, and an index can only cover rows of one dimension. |
| A4 | One SQL statement for both Arms + Fusion? | **Yes** — one statement, two CTEs, `UNION ALL`/`FULL OUTER JOIN`, fuse in the outer query. Both Arms stay index-accelerated. |
| A5 | `WHERE` + HNSW | **Post-filter by default** — pgvector applies the filter *after* the index scan, so a predicate matching 10% of rows returns ~4 of 40 candidates. Three fixes: iterative scans (0.8.x), a partial index, or partitioning. |
| B | Which Fusion wins | RRF is the safe, untuned default and what ParadeDB documents. The best current evidence (Bruch et al., TOIS 2023) says a **tuned convex combination of normalized scores beats RRF** in- and out-of-domain, and needs few labelled queries to tune. Run both as Arms — we have Qrels. |

---

# PART A — what the stack actually supports

Everything in Part A is for `paradedb/paradedb:pg17`, which today resolves to `0.25.9-pg17`
(Docker Hub tag `pg17`, `latest-pg17`, `0.25.9-pg17` and `17-v0.25.9` share digest
`sha256:8da5d202…`, pushed 2026-09-11 — [Docker Hub tags API](https://hub.docker.com/v2/repositories/paradedb/paradedb/tags)),
shipping pgvector 0.8.4 ([verified live in the baseline](pg-search-pgvector-apple-silicon.md)).
`main`'s `docker/Dockerfile.paradedb-17` now pins `postgresql-17-pgvector=0.8.6`, so the next
patch bump moves us to 0.8.6. Both are ≥ 0.8.0, which is the only version boundary that matters
below (iterative index scans).

## A1. BM25 scores, and whether Fusion is built in

**Scores: yes, first class.** `pdb.score(<key_field>)` returns the BM25 score for any query
containing a ParadeDB operator, and can be selected, ordered by, and wrapped in a window
function:

```sql
SELECT id, pdb.score(id)
FROM mock_items
WHERE description ||| 'shoes'
ORDER BY pdb.score(id) DESC
LIMIT 5;
```

— [`docs/reference/full-text/score.mdx`](https://github.com/paradedb/paradedb/blob/main/docs/reference/full-text/score.mdx).
Two caveats from that same page: a field only contributes to the score if it is *in* the
ParadeDB index, and **dead rows perturb scores** until `VACUUM` runs ("Running `VACUUM` on the
underlying table will remove all dead rows from the index and ensures that only rows visible to
the current transaction are factored into the BM25 score"). The baseline's sharp-edge #3 says
the same thing from measurement: `VACUUM ANALYZE` before every measurement.

**Fusion: no helper exists.** There is no `paradedb.rank_hybrid`, no RRF function, no fusion
aggregate. ParadeDB's own reference page for Reciprocal Rank Fusion is a *page of hand-written
SQL*, and the BM25 scoring page says outright that "The recommended approach for combining
scores from multiple tables is to use Reciprocal Rank Fusion (RRF)" — written by hand. The
canonical shape, verbatim from
[`docs/reference/hybrid/rrf.mdx`](https://github.com/paradedb/paradedb/blob/main/docs/reference/hybrid/rrf.mdx):

```sql
WITH text AS (
    SELECT id, RANK() OVER (ORDER BY pdb.score(id) DESC, id) AS rank
    FROM mock_items
    WHERE description ||| 'running shoes'
    ORDER BY pdb.score(id) DESC, id
    LIMIT 20
),
vector AS (
    SELECT id, RANK() OVER (ORDER BY embedding <=> '[1,2,3,4,5,6,7,8]', id) AS rank
    FROM mock_items
    WHERE id @@@ pdb.all()
    ORDER BY embedding <=> '[1,2,3,4,5,6,7,8]', id
    LIMIT 20
),
fused AS (
    SELECT id, sum(weight) AS score
    FROM (
        SELECT id, 1.0 / (60 + rank) AS weight FROM text
        UNION ALL
        SELECT id, 0.7 / (60 + rank) AS weight FROM vector
    ) u
    GROUP BY id
)
SELECT m.id, m.description, f.score
FROM fused f
JOIN mock_items m USING (id)
ORDER BY f.score DESC, m.id
LIMIT 5;
```

Note what ParadeDB itself says about the parameters on that page, because it is the closest
thing to vendor guidance we will get:

- `k = 60` is "the widely used default and a good starting point"; **lower `k` (20–40)** sharpens
  the curve so one branch's top hits dominate; **higher `k` (80–100)** flattens it so agreement
  across branches matters more.
- Weights: "Tune weights against a labelled query set rather than by intuition." We have Qrels,
  so this is actionable.
- Branch `LIMIT`: "The fused result can only contain rows that at least one branch returned. A
  row missing from both candidate lists cannot be recovered later" — they suggest **100 to 200
  candidates per branch for a top-10 result**. The baseline's verified query used `LIMIT 60`;
  raise it.
- Determinism: add a tiebreaker (`, id`) after *both* sort keys, or a `LIMIT` cutting through
  tied scores makes the same query return different candidates run to run. All tiebreaker
  columns must be in the ParadeDB index to keep the Top-K optimization.

pgvector supplies no fusion primitive either — the baseline confirms this from the pgvector
README. **Fusion is our SQL. That is the finding, and it is fine: it means every Fusion
variant in Part B is equally available to us, since we are writing the arithmetic anyway.**

## A2. pgvector index types at ~98,000 Chunk rows

The baseline covers parameters, defaults, ranges, dimensionality ceilings, distance operators
and per-model byte sizing; see its §3. Not repeated. What matters *at this row count*:

**Use HNSW.** The README's trade-off is fixed and one-directional: HNSW "has better query
performance than IVFFlat (in terms of speed-recall tradeoff), but has slower build times and
uses more memory", while IVFFlat has "faster build times and uses less memory … but has lower
query performance". At 98k rows the baseline *measured* an HNSW build at 2.1 s (768-dim, 32,722
rows, `shm_size=1g`) — so IVFFlat's advantage is a couple of seconds we do not need, and its
disadvantage is permanent recall loss on every query of the study.

Three further reasons specific to us:

1. **IVFFlat requires a populated table** (k-means training). We will re-ingest repeatedly while
   comparing Chunking Strategies; each re-ingest must re-train. HNSW "can be created without any
   data in the table since there isn't a training step".
2. **`ivfflat.probes` defaults to 1** — visiting ~1/32 of the Corpus at `lists = 98`. The
   baseline already flags this as "the single most likely way to accidentally sandbag the dense
   arm". Recomputed for Chunks rather than Recipes: `lists = 98000/1000 ≈ 98`,
   `probes = sqrt(98) ≈ 10`.
3. **An index per model is coming** (A3). Anything that multiplies build cost by the number of
   models argues for the cheaper *query*, not the cheaper build.

**Sizing, recomputed for ~98,000 Chunks** (arithmetic from the README's `4·d + 8` bytes per
`vector`, plus pgvector's own HNSW neighbour-tuple size —
`HNSW_NEIGHBOR_TUPLE_SIZE(level, m)` allocates `(level+2)·m` `ItemPointerData` of 6 bytes each
([`src/hnsw.h`](https://github.com/pgvector/pgvector/blob/master/src/hnsw.h)), i.e. ~192 B for a
level-0 element at the default `m = 16`):

| dims | bytes/vector | 98,000 vectors | ≈ HNSW index | ×4 models |
|---|---|---|---|---|
| 512 | 2,056 | ~192 MiB | ~210 MiB | ~840 MiB |
| 768 | 3,080 | ~288 MiB | ~306 MiB | ~1.2 GiB |
| 1024 | 4,104 | ~384 MiB | ~402 MiB | ~1.6 GiB |

Still comfortable on 16 GB, but no longer free: "Indexes build significantly faster when the
graph fits into `maintenance_work_mem`", and the README emits an explicit `NOTICE: hnsw graph no
longer fits into maintenance_work_mem after N tuples … Building will take significantly more
time.` Set `maintenance_work_mem` before a multi-model build pass, and keep `shm_size: 1g` for
parallel builds (baseline §4.2, measured: without it the parallel HNSW build *fails*).

`HNSW_MAX_DIM` is 2000 for `vector`, 4000 for `halfvec` — the baseline confirms this from source
and confirms no candidate encoder is near it.

## A3. Many vectors per Chunk, at differing dimensions — the headline

**What pgvector forces.** Two facts, both from the pgvector README, combine into the whole
answer:

1. A column may be declared **`vector` with no dimension**, and then rows in it may have
   different dimensions:
   > "You can use `vector` as the type (instead of `vector(n)`)."
   > ```sql
   > CREATE TABLE embeddings (model_id bigint, item_id bigint, embedding vector, PRIMARY KEY (model_id, item_id));
   > ```
2. But an **index cannot span dimensions**:
   > "However, you can only create indexes on rows with the same number of dimensions (using
   > [expression](https://www.postgresql.org/docs/current/indexes-expressional.html) and
   > [partial](https://www.postgresql.org/docs/current/indexes-partial.html) indexing):"
   > ```sql
   > CREATE INDEX ON embeddings USING hnsw ((embedding::vector(3)) vector_l2_ops) WHERE (model_id = 123);
   > ```
   > and query with:
   > ```sql
   > SELECT * FROM embeddings WHERE model_id = 123 ORDER BY embedding::vector(3) <-> '[3,1,2]' LIMIT 5;
   > ```

— [pgvector README, "Can I store vectors with different dimensions in the same column?"](https://github.com/pgvector/pgvector/blob/master/README.md#can-i-store-vectors-with-different-dimensions-in-the-same-column)

So the dimension-typing does **not** force one column per model. It forces **one index per
model**, whichever layout we choose. That reframes the ticket's question: the layouts differ in
what *else* they cost, not in index count.

### Option A — one row per (chunk, model), one dimensionless `vector` column

This is [map #13's premise 9](https://github.com/dangth1101/IS6303/issues/13) ("embeddings are
keyed by `(chunk, model)`, never fixed columns on a chunk row") and it is the right call.

```sql
CREATE TABLE chunk_embedding (
  chunk_id   bigint  NOT NULL REFERENCES chunk(id) ON DELETE CASCADE,
  model_id   smallint NOT NULL REFERENCES embedding_model(id),
  embedding  vector  NOT NULL,            -- no typmod: dimensions vary by model
  PRIMARY KEY (model_id, chunk_id)        -- model first: clusters one model's rows together
);

-- integrity the missing typmod would otherwise have given us, one per model
ALTER TABLE chunk_embedding ADD CONSTRAINT emb_dims_ok
  CHECK (vector_dims(embedding) = CASE model_id WHEN 1 THEN 768 WHEN 2 THEN 1024 ... END);

-- one index per model
CREATE INDEX chunk_emb_m1 ON chunk_embedding
  USING hnsw ((embedding::vector(768)) vector_cosine_ops) WHERE (model_id = 1);
CREATE INDEX chunk_emb_m2 ON chunk_embedding
  USING hnsw ((embedding::vector(1024)) vector_cosine_ops) WHERE (model_id = 2);
```

The check constraint is the README's own suggestion in the adjacent FAQ entry
(`ALTER TABLE items ADD CHECK (vector_dims(embedding::vector) = 3)`), and it matters more here
than usual: without a typmod, nothing stops an ingest bug from writing a 768-dim vector under
`model_id = 2` and silently excluding it from the index.

**Index-maintenance cost.** Each partial index contains only its own model's rows — 98,000
entries, sized per A2. Adding a model is `INSERT` + one `CREATE INDEX`; nothing existing is
rewritten. An `INSERT` evaluates every partial index's predicate (a `smallint` equality — free)
and writes into exactly one graph. Re-embedding one model is `DELETE … WHERE model_id = n` +
re-insert + `REINDEX INDEX chunk_emb_mn`, touching one index. Dropping a model is
`DROP INDEX` + `DELETE`. **This is the cheapest layout for the thing we will actually do most
often: add, re-run and compare models.**

**The two sharp edges, and they are sharp.**

1. **A partial index is not usable from a parameterized query.** PostgreSQL:
   > "a partial index can be used in a query only if the system can recognize that the `WHERE`
   > condition of the query mathematically implies the predicate of the index."

   and, decisively:

   > "Matching takes place at query planning time, not at run time. As a result, parameterized
   > query clauses do not work with a partial index. For example a prepared query with a
   > parameter might specify `x < ?` which will never imply `x < 2` for all possible values of
   > the parameter."

   — [PostgreSQL 17, Partial Indexes](https://www.postgresql.org/docs/17/indexes-partial.html)

   So `WHERE model_id = $2` **cannot** use `WHERE (model_id = 1)` under a *generic* plan. Under a
   *custom* plan the parameter value is substituted before planning and the implication holds —
   and Postgres uses custom plans for the first five executions, then may switch:
   > "the first five executions are done with custom plans and the average estimated cost of
   > those plans is calculated. Then a generic plan is created and its estimated cost is compared
   > to the average custom-plan cost."
   > "This heuristic can be overridden … by setting `plan_cache_mode` to `force_generic_plan` or
   > `force_custom_plan`."

   — [PostgreSQL 17, PREPARE](https://www.postgresql.org/docs/17/sql-prepare.html)

   **Consequence for the FastAPI layer:** the `model_id` must reach the planner as a *literal*,
   not a bind parameter — interpolate the model id into the SQL string (it is an internal
   smallint from our own registry, not user input) — or the dense Arm will silently fall off its
   index on the sixth execution and every one after. This is exactly the class of bug that makes
   a comparison study produce a wrong answer rather than an error: the query still *works*, it
   just seq-scans 392,000 rows and the Arm looks slow. The query's cast must also match the
   index expression character for character.

2. **The `WHERE model_id = n` predicate is itself the post-filter trap of A5** — unless the
   partial index handles it, which it does *by construction* (the index contains only that
   model's rows, so there is nothing left to filter). This is the strongest argument for the
   partial index over any "one big index plus a filter" scheme, and it is the point where A3 and
   A5 meet.

**Variant A′ — LIST-partition by `model_id`.** Partitioning sidesteps sharp edge 1 entirely,
because partition pruning is not the implication prover:

> "Partition pruning can be performed not only during the planning of a given query, but also
> during its execution." … "Partition pruning may also be performed here to remove partitions
> using values which are only known during actual query execution."

— [PostgreSQL 17, Partitioning](https://www.postgresql.org/docs/17/ddl-partitioning.html)

Each partition then carries a plain (non-partial) expression index on its own dimension, and a
bind parameter works. pgvector's README independently recommends partitioning for this shape:
"If filtering by many different values, consider partitioning", and its Multitenancy section
warns that "sharing an approximate index between tenants means vectors from one tenant can
affect recall (and speed) for other tenants" — models are tenants here. **Cost:** DDL per model
is now `CREATE TABLE … PARTITION OF` plus an index, and every query plan carries an Append. At
our scale this is a wash; take it if the backend wants clean parameter binding, take plain
Option A if it wants the simplest schema.

### Option B — one row per Chunk, several typed vector columns

```sql
ALTER TABLE chunk ADD COLUMN emb_bge  vector(1024);
ALTER TABLE chunk ADD COLUMN emb_e5   vector(768);
```

Cheaper to read when you want two models' vectors for the *same* Chunk in one row (we never do —
the Arms are compared, not combined), and `NULL` vectors are simply not indexed, so a
half-filled column is harmless.

**Why it loses.**

- **Adding a model is a schema migration**, not a data load: `ALTER TABLE ADD COLUMN` then a
  full-table `UPDATE` to backfill 98,000 rows. Each backfilling `UPDATE` writes a **new heap
  tuple** for the whole Chunk row and a new index entry in every index on that table — including
  the ParadeDB BM25 index, which is a covering index over all the Chunk's text. Ingesting model
  *n+1* therefore churns model *n*'s indexes and the BM25 index for no reason, then needs a
  `VACUUM` (which, per A1, BM25 scores are sensitive to).
- **Wide rows.** pgvector declares all vector types `STORAGE = external`
  ([`sql/vector.sql`](https://github.com/pgvector/pgvector/blob/master/sql/vector.sql)), so
  vectors above the ~2 kB tuple threshold live out of line, uncompressed, in TOAST. That keeps
  the main tuple narrow, but it means each extra model column adds a TOAST chain per row, and
  the README warns the planner mishandles this: "The planner doesn't consider out-of-line storage
  in cost estimates, which can make a serial scan look cheaper."
- **It forecloses the comparison we are running.** The number of models is the independent
  variable of premise 9. Encoding it in the *schema* means every model added or dropped is a
  migration, and the ingest pipeline that is supposed to differ "by nothing else" now differs by
  DDL.
- It buys nothing on index count or index size: still one HNSW index per model, same bytes.

**Verdict: Option A (or A′). One row per `(chunk, model)`, dimensionless `vector` column, one
partial expression HNSW index per model, `model_id` as a literal in the dense Arm's SQL.**

### One coupling to note

ParadeDB's own native vector index derives dimensionality from the column's typmod:
`let dims = if typmod > 0 { typmod as usize } else { 0 };`
([`pg_search/src/schema/mod.rs`](https://github.com/paradedb/paradedb/blob/main/pg_search/src/schema/mod.rs)).
A dimensionless `vector` column yields `0`, and ParadeDB publishes no expression/partial-index
recipe for vector fields the way pgvector does. **Choosing Option A therefore also chooses
pgvector's HNSW over ParadeDB's native vector index** — which is where the baseline already
landed for independent reasons (§6: beta storage format, open panic #6076, and the
one-index-per-table rule removing independent tunability). The two decisions reinforce each
other; see A6.

## A4. One statement, or two queries joined in the application?

**One statement.** The RRF query in A1 runs both Arms and fuses them in a single
`SELECT`, and ParadeDB documents how to prove both Arms stayed index-accelerated: `EXPLAIN` must
show a `Custom Scan (ParadeDB Base Scan)` with `Exec Method: TopKScanExecState` **for each
branch**, with the branch `LIMIT` pushed in as `TopK Limit`.

```text
->  WindowAgg
      Window: w1 AS (ORDER BY pdb.score(id), id ROWS UNBOUNDED PRECEDING)
      ->  Custom Scan (ParadeDB Base Scan) on mock_items
            Exec Method: TopKScanExecState
               TopK Order By: pdb.score() desc, id asc
               TopK Limit: 20
```

"If you do not see `TopKScanExecState`, the branch is not taking the optimized path."

The baseline *measured* this shape end to end at 12.5 ms over 32,722 rows, using a
`FULL OUTER JOIN` rather than `UNION ALL … GROUP BY`. Both are correct;
`FULL OUTER JOIN` + `COALESCE(1.0/(60+rnk), 0)` makes the "found by only one Arm" case explicit,
which is precisely the **Pool** the Qrel protocol needs. Prefer it for that reason.

Two constraints on the statement, both from the baseline and both verified there:

- The `@@@` right-hand side **must be a literal or a bind parameter**, never a column threaded
  through a CTE — that raises `ERROR: Unsupported query shape`. A normal `$1` from the driver is
  fine.
- `ORDER BY` for the dense Arm must be a bare distance operator ascending.
  `ORDER BY 1 - (embedding <=> $1) DESC` gets no index at all. Compute the similarity in the
  select list if you want to display it.

Note the asymmetry in ParadeDB's example that is easy to miss: the vector branch's `WHERE` is
`id @@@ pdb.all()`, which exists only so a ParadeDB predicate is present at the same level as
the `ORDER BY … LIMIT`. That is a *ParadeDB-native* vector-search construct. With pgvector's
HNSW driving the dense Arm (our choice), the dense branch needs no `@@@` at all — which is what
the baseline's verified query does.

**When two queries are better anyway.** If we implement any Part B fusion that needs a
*population statistic* over one Arm's scores — z-score normalization needs mean and standard
deviation; min-max needs the min and max of the retrieved set — SQL can still do it in one
statement (a window `avg()`/`stddev_samp()`/`min()`/`max()` over the branch CTE). Nothing in
Part B forces the application round trip. The only reason to split is if we want to cache one
Arm's results across several fusion settings while sweeping, which is an evaluation-harness
decision, not a database one.

## A5. `WHERE` against an HNSW scan — pre- vs post-filter, and the recall hit (for #22)

**Default behaviour is post-filtering, and the README states both the mechanism and the damage
in one sentence:**

> "With approximate indexes, filtering is applied *after* the index is scanned. If a condition
> matches 10% of rows, with HNSW and the default `hnsw.ef_search` of 40, only 4 rows will match
> on average."

— [pgvector README, Filtering](https://github.com/pgvector/pgvector/blob/master/README.md#filtering)

That is the whole problem in miniature. The HNSW scan returns a fixed-size candidate list
(`hnsw.ef_search`, default 40) chosen *without knowledge of the predicate*; the predicate then
deletes from it. The degradation is roughly multiplicative in selectivity: a filter passing
fraction *s* leaves ≈ `s · ef_search` rows. At `LIMIT 10` you need `s ≳ 0.25` just to fill the
page, and "filling the page" is not the same as "returning the true top 10" — the survivors are
the top-*s*-fraction of an unfiltered neighbourhood, not the true nearest neighbours *among
matching rows*. The related troubleshooting entry is blunt: "Why are there less results for a
query after adding an HNSW index? Results are limited by the size of the dynamic candidate list
(`hnsw.ef_search`), which is 40 by default."

**Four remedies, in the order I would reach for them.**

1. **Partial index per filter value — a true pre-filter.** "If filtering by only a few distinct
   values, consider partial indexing": `CREATE INDEX … USING hnsw (…) WHERE (category_id = 123);`
   The graph then contains only matching rows, so the scan is exact with respect to that
   predicate and recall is unaffected. **This is what A3's per-model partial index already gives
   us for free on `model_id`** — the highest-cardinality, most-certain filter in the system
   costs us nothing. Subject to the parameterization trap in A3.
2. **Partitioning — a true pre-filter, parameter-safe.** "If filtering by many different values,
   consider partitioning." See A3 variant A′.
3. **Iterative index scans (pgvector 0.8.0+, so present in both 0.8.4 and 0.8.6).** This is the
   part that *changes the picture* versus pre-0.8 advice:
   > "Starting with 0.8.0, you can enable iterative index scans, which will automatically scan
   > more of the index until enough results are found (or it reaches `hnsw.max_scan_tuples` or
   > `ivfflat.max_probes`)."

   ```sql
   SET hnsw.iterative_scan = strict_order;   -- exact order by distance
   SET hnsw.iterative_scan = relaxed_order;  -- "slightly out of order … but provides better recall"
   SET hnsw.max_scan_tuples = 20000;         -- default; "approximate and does not affect the initial scan"
   SET hnsw.scan_mem_multiplier = 2;         -- "Try increasing this if increasing hnsw.max_scan_tuples does not improve recall"
   ```

   Two things to be precise about. **It fixes result *count*, and mostly fixes recall, but it is
   bounded** — the scan stops at `max_scan_tuples` (20,000, i.e. ~20% of our 98,000 Chunks) or at
   `scan_mem_multiplier × work_mem`, whichever comes first, and returns whatever it has. For a
   filter selective enough that its matches are scattered thinly through the graph, iterative
   scanning degenerates toward scanning a fifth of the index for a partial answer. And
   `relaxed_order` recovers recall at the cost of ordering; if strict order is needed, the README
   gives the materialized-CTE trick:
   ```sql
   WITH relaxed_results AS MATERIALIZED (
       SELECT id, embedding <-> '[1,2,3]' AS distance FROM items WHERE category_id = 123 ORDER BY distance LIMIT 5
   ) SELECT * FROM relaxed_results ORDER BY distance + 0;
   ```
   ("`+ 0` is needed for Postgres 17+" — which is us.)
4. **Raise `hnsw.ef_search`.** Crude — cost is paid on every query including unfiltered ones —
   but valid: `SET LOCAL hnsw.ef_search = 100` (max 1000, `HNSW_MAX_EF_SEARCH`). Combine with
   over-fetching the branch `LIMIT`, which the Fusion design wants anyway (A1: 100–200
   candidates per branch).

   And for genuinely selective predicates, the README's first suggestion is not an ANN index at
   all: "A good place to start is creating an index on the filter column. This can provide fast,
   exact nearest neighbor search in many cases." At 98,000 Chunks an exact scan over a narrow
   filtered subset is entirely affordable, and it has *perfect* recall. Do not rule it out.

**The ParadeDB alternative, which #22 should know about.** As of 0.25.0, the ParadeDB index can
index pgvector's `vector` type *inside the same index as the text and the filter columns*, and
push the filter into the same scan — genuine pre-filtering with no recall cliff:

> "The ParadeDB index can index pgvector's `vector` type alongside your text and other columns.
> This lets you combine vector search with full text search and filters in a single index, which
> can significantly improve latency/recall for selective queries."

```sql
SELECT id, description
FROM mock_items
WHERE category === 'footwear'
ORDER BY embedding <=> '[1, 2, 3, 4, 5, 6, 7, 8]'
LIMIT 5;
```
```text
->  Custom Scan (ParadeDB Base Scan) on mock_items
      Exec Method: TopKScanExecState
         TopK Order By: embedding <=> vector asc
         TopK Limit: 5
      Tantivy Query: {"with_index":{"query":{"term":{"field":"category","value":"footwear"}}}}
```
"Notice that the `category === 'footwear'` filter was pushed down into the same index scan that
performs the vector search. Without pushdown, the filter would run as a separate step after the
nearest neighbors were fetched."
— [`docs/reference/vector/querying.mdx`](https://github.com/paradedb/paradedb/blob/main/docs/reference/vector/querying.mdx)

It is a SPANN-style (IVF-like, clustered) index, not HNSW, tuned by `centroid_ratio`,
`training_samples_per_centroid` and `cluster_replication` at build time and
`paradedb.vector_cluster_max_probe` (default `0.02`) at query time — and note that
`cluster_replication` exists specifically because "Higher values can improve recall for
**filtered** queries at the cost of a larger index"
([`docs/reference/indexing/indexing-vectors.mdx`](https://github.com/paradedb/paradedb/blob/main/docs/reference/indexing/indexing-vectors.mdx),
[`docs/reference/vector/tuning.mdx`](https://github.com/paradedb/paradedb/blob/main/docs/reference/vector/tuning.mdx)).

**I do not recommend switching to it**, for the baseline's reasons plus one of my own — see A6.

**Answer for #22, stated plainly.** With pgvector HNSW, a `WHERE` predicate is a *post*-filter
and costs recall in proportion to how much it excludes. Our own design forces one such predicate
(`model_id`), and the per-model partial index converts it into a pre-filter at zero cost. Any
*additional* UI filter (category, rating, "has an image") will hit the post-filter path, and the
right response is, in order: partial index if the filter has few values; partition if many;
iterative scan with `relaxed_order` + materialized CTE for the general case; raised `ef_search`
plus over-fetch as the always-available blunt instrument; exact scan when the filter is very
selective. **If filtering turns out to be central to the product rather than incidental, that is
the argument that would justify revisiting ParadeDB's native vector index** — which is precisely
what #22 should decide.

## A6. Where this updates or contradicts the baseline

Nothing here contradicts anything the baseline *measured*. Three places where the documentary
record has moved or where I would qualify a recommendation:

| Point | Baseline (executed 2026-09-14) | Current docs (`main`, post-0.25.9) | Which to trust |
|---|---|---|---|
| pgvector version | 0.8.4 in the image, verified at runtime | `main`'s `Dockerfile.paradedb-17` pins 0.8.6 | Baseline, for today's image. Both ≥ 0.8.0; nothing in A5 changes. Re-verify after any `docker compose pull`, as the ADR warns the patch floats. |
| ParadeDB native vector search | "Beta subsystems to avoid … Use pgvector's own HNSW/IVFFlat indexes" | Now has a full reference section (indexing, querying, tuning, hybrid) and drives the official RRF example; still labelled beta in every page header | **Baseline, but the reasoning needs updating.** The docs are no longer thin — but the 0.25.0 changelog still says "While vector search is marked as beta, future releases may change the vector storage format and require a reindex", and A3 adds a new, independent reason: it needs a typmod, so it cannot index the dimensionless `(chunk, model)` column at all. Two independent reasons now point the same way. |
| RRF branch `LIMIT` | verified at `LIMIT 60` | ParadeDB recommends 100–200 candidates per branch for a top-10 | Docs. The baseline's 60 was a working demonstration, not a tuned choice; the ceiling argument in A1 is sound and the measured 12.5 ms leaves room. |

One nuance worth recording rather than a contradiction: the baseline's timings are over **32,722
rows** (one row per Recipe). Our real dense index is over **~98,000 Chunks** — 3× the rows —
multiplied by the number of models. Treat its 2.1 s HNSW build and 12.5 ms fusion query as a
*lower bound*, not a prediction.

---

# PART B — the Fusion algorithms

The vocabulary: each **Arm** produces a ranked list; **Fusion** combines two or more into one.
The question is what the combiner consumes (ranks or scores), what it is robust to, what has to
be tuned, and what the evidence says.

## B1. Reciprocal Rank Fusion

**Origin.** G. V. Cormack, C. L. A. Clarke, S. Büttcher, *"Reciprocal Rank Fusion outperforms
Condorcet and individual Rank Learning Methods"*, SIGIR '09, Boston, pp. 758–759
([PDF](http://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf),
[ACM](https://dl.acm.org/doi/10.1145/1571941.1572114)).

**The formula, verbatim from the paper.** Given a set *D* of documents and a set of rankings *R*,
each a permutation on 1..|D|:

> RRFscore(d ∈ D) = Σ<sub>r ∈ R</sub> 1 / (k + r(d))

**Where `k = 60` comes from — the answer is "a pilot experiment", and the paper says so.**

> "where *k* = 60 was fixed during a pilot investigation and not altered during subsequent
> validation. Our intuition in choosing this formula derived from fact that while highly-ranked
> documents are more important, the importance of lower-ranked documents does not vanish as it
> would were, say, an exponential function used. The constant *k* mitigates the impact of high
> rankings by outlier systems."

The pilot was "four pilot experiments, each combining the results of 30 configurations of Wumpus
Search applied to four different TREC collections", and its result is the single most useful
number in the paper, because it bounds how much `k` can possibly matter — MAP over TREC topics
351–400:

| k | 0 | 10 | 20 | 30 | 40 | 50 | **60** | 70 | 80 | 90 | 100 | 500 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MAP | .2072 | .2123 | .2134 | .2139 | .2138 | .2144 | **.2145** | .2146 | .2147 | .2145 | .2142 | .2098 |

> "the results of the first, shown in table 1, indicated that *k* = 60 was near-optimal, but that
> the choice was not critical."

Read that table carefully: the *whole* range 20–100 spans 0.0013 MAP (0.6% relative), while
`k = 0` costs 3.4% and `k = 500` costs 2.2%. **`k` is a shallow parameter over its sane range and
a real one outside it.** ParadeDB's guidance (`k` 20–40 sharpens, 80–100 flattens) is a
reasonable description of the *shape*; the paper's evidence says do not expect the choice to move
the number much.

**What it consumes:** ranks only. Scores are discarded. ParadeDB states the motivation exactly:
"BM25 scores and vector distances operate on different scales, so they cannot be added together
directly … a row that ranks first contributes the same amount whether its BM25 score was `1` or
`100`."

**What it is robust to:** score scale, score distribution, monotone score transformations,
outlier scores, and an Arm whose scores are badly calibrated. Elasticsearch's framing is the
industry's: RRF "requires no tuning, and the different relevance indicators do not have to be
related to each other to achieve high-quality results", and it "removes the need to figure out
what the appropriate weighting is using linear combination"
([Elasticsearch RRF reference](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion),
`rank_constant` default **60**, confirming the constant's status as a de facto standard).

**What it is *not* robust to, and this matters for us:**

- **It cannot see margins.** A Chunk that wins its Arm by a mile and one that wins by a hair
  contribute identically. When one Arm returns nothing genuinely relevant, its rank-1 result
  still gets full weight — the exact failure mode of a vector Arm on a query with no semantic
  match, and of a BM25 Arm on an out-of-vocabulary query.
- **The candidate cut-off is load-bearing.** A document absent from both lists scores 0 and
  cannot be recovered (ParadeDB says this explicitly). RRF's behaviour therefore depends on the
  branch `LIMIT` as much as on `k`.
- **Ties.** With `RANK()` and tied BM25 scores, tied rows get identical contributions; with
  `ROW_NUMBER()` the tie is broken arbitrarily unless a tiebreaker column is in the `ORDER BY`.
  ParadeDB warns that "Duplicate vectors make this common in the vector branch".

**Tuning surface:** `k`, per-Arm weights (`weight / (k + rank)`), and branch depth. Weights are
where the real gains are, and they need Qrels.

**Lineage worth one line.** RRF's ancestors are Fox & Shaw's score-combination family —
`CombSUM` = "SUM(Individual Similarities)", `CombMNZ` = "SUM(Individual Similarities) × Number of
Nonzero Similarities", plus `CombMAX`, `CombMIN`, `CombANZ` (E. A. Fox and J. A. Shaw,
*Combination of Multiple Searches*, [TREC-2, 1994](https://trec.nist.gov/pubs/trec2/papers/txt/23.txt))
— and Montague & Aslam's Condorcet Fuse (CIKM '02), which RRF's own title claims to beat. In the
paper's TREC table RRF's MAP exceeds Condorcet in all four collections and CombMNZ in three of
four, with RRF outperforming "Condorcet, CombMNZ and the best system by 4% to 5% on average". Our
**Roll-up by max** is, notably, `CombMAX` applied across a Recipe's Chunks — the same family.

## B2. Normalized weighted score fusion

The alternative: put both Arms' raw scores on a common scale, then take a weighted sum. The
combiner is trivial; **all the difficulty is in the normalization**, which is why the ticket
asks about it by name. Writing *α* for the weight on the semantic Arm, the fusion function is a
convex combination:

> f<sub>Convex</sub>(q,d) = α · φ<sub>Sem</sub>(f<sub>Sem</sub>(q,d)) + (1 − α) · φ<sub>Lex</sub>(f<sub>Lex</sub>(q,d))

with φ the per-Arm normalizer (Bruch, Gai & Ingber, below). The three normalizers:

### Min-max ("standard" normalization)

> φ<sub>mm</sub>(f(q,d)) = (f(q,d) − m<sub>q</sub>) / (M<sub>q</sub> − m<sub>q</sub>)

where m<sub>q</sub> and M<sub>q</sub> are the minimum and maximum scores **in the retrieved set
for that query**.

- **Needs:** raw scores, plus the min and max of the retrieved set — i.e. it is computed *per
  query*, over the candidate list. In SQL: `min()`/`max()` as window functions over the branch
  CTE.
- **Robust to:** differing scales and offsets between Arms. Guarantees output in [0, 1].
- **Fragile to:** the extremes. A single outlier sets the scale for everything else; and because
  m and M come from the *retrieved* set, the top result always normalizes to exactly 1.0 and the
  bottom to 0.0 **no matter how good or bad they are**. A query where the dense Arm found nothing
  relevant still gets a 1.0. Changing the branch depth changes every normalized score. This is
  the same information loss RRF has, in a subtler place.
- **Tuning:** *α* only. Weaviate's `relativeScoreFusion` is exactly this — "The highest value
  becomes 1, the lowest value becomes 0, and others end up in between according to this scale" —
  and it is **their default since v1.24**, on the argument that it "retains more information from
  the original searches than rankedFusion, which only retains the rankings"
  ([Weaviate hybrid search docs](https://docs.weaviate.io/weaviate/search/hybrid),
  [Weaviate fusion-algorithms blog](https://weaviate.io/blog/hybrid-search-fusion-algorithms)).

### Z-score

> φ<sub>z</sub>(f(q,d)) = (f(q,d) − μ) / σ

with μ, σ the mean and standard deviation of that Arm's scores for that query.

- **Needs:** raw scores plus two moments — `avg()` and `stddev_samp()` over the branch CTE.
- **Robust to:** single outliers, far better than min-max: one extreme value moves μ and σ a
  little instead of redefining the endpoints. Handles differing *spreads*, not just differing
  ranges.
- **Fragile to:** unbounded output (no [0, 1] guarantee, so weights are harder to reason about),
  σ → 0 when an Arm returns near-identical scores (divide-by-zero; must be special-cased), and an
  implicit assumption that the two Arms' score distributions are similar in *shape*.
- **Tuning:** *α*.

### Theoretical min-max (TM-norm) — the one to prefer

> φ<sub>tmm</sub>(f(q,d)) = (f(q,d) − inf f(q,·)) / (M<sub>q</sub> − inf f(q,·))

The minimum is the metric's **theoretical** infimum rather than the observed minimum. From the
paper: "when f<sub>Lex</sub> is BM25, then its infimum is 0. When f<sub>Sem</sub> is cosine
similarity, then that quantity is −1."

- **Needs:** raw scores plus the observed max (and a known theoretical floor for each metric).
- **Robust to:** the bottom-of-list artefact that spoils min-max. A weak Arm's scores now stay
  *low* instead of being stretched to fill [0, 1], so "this Arm found nothing" survives
  normalization — exactly the signal RRF throws away. It also makes normalized scores comparable
  *across queries*, which plain min-max is not.
- **Fragile to:** still anchored on the observed max, so a single high outlier compresses
  everything below it; and it requires knowing each metric's floor (trivial for cosine and BM25,
  awkward for a raw inner product).
- **Tuning:** *α*.

**Our metrics make this easy.** The dense Arm is cosine distance (`<=>`), so similarity is
`1 - distance` ∈ [0, 2] with a known floor; the sparse Arm is BM25, floor 0. Both theoretical
anchors are known constants, so TM-norm is implementable in the same SQL statement as the
Arms, with a single `max()` window per branch and no round trip.

## B3. Other things in current practice

- **Distribution-Based Score Fusion (DBSF).** Normalize each Arm by its own 3-sigma range —
  `ŝ = (s − (μ − 3σ)) / 6σ` — then sum. A z-score in disguise, rescaled into a nominal [0, 1].
  Shipped by Qdrant alongside RRF, with honest caveats: scores are not clipped, the statistics
  come from a small prefetch sample and "outliers can skew normalization", it "works best when
  you trust retrievers' raw scores to carry magnitude information", and "Neither consistently
  dominates — evaluation on your specific dataset is recommended"
  ([Qdrant hybrid queries](https://qdrant.tech/documentation/concepts/hybrid-queries/)).
- **Weighted RRF.** `weight / (k + rank)` per branch, which ParadeDB's own example uses
  (`1.0` text, `0.7` vector). A middle road: keeps RRF's scale-robustness, adds the one knob
  that matters, still ignores margins.
- **Per-Arm score normalization pipelines.** OpenSearch's normalization processor is a
  production implementation of exactly B2: it normalizes each query clause's scores
  (`min_max`, `z_score`, and others), then combines them (`arithmetic_mean` and others) with a
  per-clause `weights` array — because "BM25 and k-NN search use different scales … it is
  beneficial to normalize them so that they are on the same scale". Evidence that "normalize
  then weight" is mainstream production practice, not a research curiosity
  ([normalization processor](https://docs.opensearch.org/latest/search-plugins/search-pipelines/normalization-processor/),
  [z-score announcement](https://opensearch.org/blog/introducing-the-z-score-normalization-technique-for-hybrid-search/)).
- **Reranking, which is the real competitor.** A cross-encoder over the fused candidate list beats
  tuning the fusion arithmetic, and the repo already has that research: `research/cross-encoder-reranker`
  recommends `cross-encoder/ms-marco-MiniLM-L6-v2` at rerank depth 100 (222 ms on the M4). Fusion
  and reranking are complements — Fusion builds the candidate list, the reranker orders it — and
  the map already treats "RRF + Rerank" as a fourth Arm. Nothing in Part B changes that.

## B4. What the evidence says about which wins

**The strongest single source is** Sebastian Bruch, Siyu Gai, Amir Ingber, *"An Analysis of Fusion
Functions for Hybrid Retrieval"*, ACM Transactions on Information Systems, 2023
([arXiv:2210.11934](https://arxiv.org/abs/2210.11934), [DOI 10.1145/3596512](https://doi.org/10.1145/3596512)).
It evaluates on MS MARCO Passage v1 and eight BEIR datasets (NQ, Quora, NFCorpus, HotpotQA, Fever,
SciFact, DBPedia, FiQA). Abstract, verbatim:

> "We study hybrid search in text retrieval where lexical and semantic search are fused together
> with the intuition that the two are complementary in how they model relevance. In particular, we
> examine fusion by a convex combination (CC) of lexical and semantic scores, as well as the
> Reciprocal Rank Fusion (RRF) method, and identify their advantages and potential pitfalls.
> Contrary to existing studies, we find RRF to be sensitive to its parameters; that the learning
> of a CC fusion is generally agnostic to the choice of score normalization; that CC outperforms
> RRF in in-domain and out-of-domain settings; and finally, that CC is sample efficient, requiring
> only a small set of training examples to tune its only parameter to a target domain."

Four claims, each of which bears directly on this project:

1. **RRF *is* sensitive to its parameters**, contrary to the folklore (and contrary to
   Elasticsearch's "requires no tuning"). Note this is not in conflict with Cormack's table:
   Cormack fused 30 *homogeneous* TREC runs, where `k` was shallow; Bruch fuses two
   *heterogeneous* retrievers, which is our case.
2. **The choice of normalization barely matters once *α* is learned.** So the min-max /
   z-score / TM-norm question is *less* important than the ticket's framing implies — pick
   TM-norm for its theoretical-floor property and spend the effort on *α*.
3. **Convex combination beats RRF**, in-domain and out-of-domain.
4. **It is sample efficient** — "only a small set of training examples to tune its only
   parameter". We are hand-building Qrels over a Pool; a small labelled set is exactly what we
   will have.

**Corroboration from practice, weaker but independent:** Weaviate switched its *default* from
rank fusion to score fusion in v1.24 and reports "a ~6% improvement in recall over the
`rankedFusion` method" on FIQA in internal benchmarks — one dataset, one vendor, self-reported,
so treat it as directional. Qdrant declines to pick a winner at all. Elasticsearch's position
("no tuning") is a defensible *default* for a general-purpose engine that has no labels, which is
precisely the situation we are not in.

**The honest summary:** RRF is what you use when you cannot measure. We can measure — that is
the entire point of the project — so RRF is the *baseline*, not the answer.

## B5. What this forces on our design

1. **Make Fusion a parameter of the Fusion Arm, not a hardcoded formula.** RRF (`k`, weights) and
   weighted TM-normalized convex combination (*α*) differ by about ten lines of SQL over the same
   two branch CTEs. Running both is nearly free and directly answers the interesting question.
   That is an Arm-level comparison, which is the unit the project already compares.
2. **Fuse Chunks, then Roll-up.** Both Arms retrieve **Chunks**; **Roll-up** collapses Chunk hits
   into a Recipe ranking by max (premise 7). The order matters and should be stated in the ADR:
   fusing Chunk rankings first, then rolling up, keeps each Arm's ranks defined over the unit each
   Arm actually scored. Rolling up first and fusing Recipe rankings is also defensible but changes
   what `rank` means — and with RRF, "rank" is the entire input. Do not leave this implicit.
3. **Over-fetch each branch.** 100–200 Chunks per Arm for a top-10 Recipe result (ParadeDB's
   guidance), and remember Roll-up compresses further: 200 Chunks may be far fewer than 200
   distinct Recipes, since a Recipe has many Chunks. The branch `LIMIT` should be set against the
   *post-Roll-up* count we need, not the Chunk count.
4. **The `FULL OUTER JOIN` shape is the Pool.** A Chunk found by exactly one Arm must still be
   ranked (`COALESCE(…, 0)`), and the union of both branches' candidates is precisely the **Pool**
   the Qrel protocol requires. One query serves retrieval and judgment-set construction.
5. **Fix `VACUUM ANALYZE` in the measurement protocol** — BM25 scores move with dead rows (A1),
   and every score-based Fusion in Part B consumes those scores directly, where RRF only consumes
   their order. Score fusion is *more* exposed to this than RRF is.

---

## Recommendations for dependent tickets

**Schema ticket.** `chunk_embedding(chunk_id, model_id, embedding vector)` — dimensionless column,
PK `(model_id, chunk_id)`, a `vector_dims()` check constraint per model, and one partial
expression HNSW index per model: `USING hnsw ((embedding::vector(D)) vector_cosine_ops) WHERE
(model_id = N)`. Not multiple typed columns; not ParadeDB's native vector index (A3, A6).
Consider LIST-partitioning by `model_id` if the backend wants to bind `model_id` as a parameter.

**Backend ticket.** The dense Arm's `model_id` must be a **literal in the SQL text**, and the
`::vector(D)` cast must match the index expression exactly, or the partial index is silently
unused from the sixth execution onward (A3). Keep the text Arm's query as a bind parameter
(`$1`) — that is verified working in the baseline. Set `hnsw.ef_search` per request via
`SET LOCAL`.

**Filtering ticket (#22).** A5 is your answer: post-filter by default, quantified as
`≈ s · ef_search` surviving candidates; remedies ranked; iterative scans (0.8.x) help but are
bounded by `max_scan_tuples = 20000`. If product-level filtering turns out to be central rather
than incidental, that is the one argument strong enough to reopen ParadeDB's native vector index.

**Fusion ADR.** Two decisions to record: (a) RRF as baseline **and** a TM-normalized weighted
convex combination as a second Fusion Arm, tuned on Qrels — justified by Bruch et al. over
Cormack et al.; (b) Fusion happens at Chunk level, Roll-up after.

---

## Sources

**Primary — ParadeDB** (repo `main`, post-`v0.25.9`; docs are versioned in-repo under `docs/`):
- `docs/reference/hybrid/rrf.mdx` — the hand-written RRF query, `k` guidance, weights, branch limit, determinism, pushdown verification
- `docs/reference/hybrid/overview.mdx`, `docs/concepts/hybrid/overview.mdx` — why rank, not score
- `docs/reference/full-text/score.mdx` — `pdb.score()`, joined scores via RRF, `VACUUM` and score freshness
- `docs/reference/vector/{overview,querying,tuning}.mdx`, `docs/reference/indexing/indexing-vectors.mdx` — native vector search (beta), SPANN index, filter pushdown, `centroid_ratio`/`cluster_replication`/`paradedb.vector_cluster_max_probe`
- `docs/reference/indexing/{indexing-partial,indexing-expressions}.mdx`, `docs/reference/filtering/{overview,external-indexes}.mdx`
- `docs/reference/operators-and-functions.mdx` — `@@@ ||| &&& === ### ## ##>`
- `docs/project/changelog/0.25.0.mdx` — pgvector now a required dependency; native vector search beta ("future releases may change the vector storage format and require a reindex")
- `pg_search/src/schema/mod.rs` — vector dimensions taken from the column typmod
- `docker/Dockerfile.paradedb-17`; Docker Hub tags API for the `pg17` digest

**Primary — pgvector** (README and source at `master`):
- README: HNSW vs IVFFlat, index options, build time and `maintenance_work_mem`, Filtering, Multitenancy, Iterative Index Scans, FAQ ("Can I store vectors with different dimensions in the same column?"), Troubleshooting
- `src/hnsw.h` (`HNSW_MAX_DIM`, `HNSW_DEFAULT_*`, neighbour tuple sizing), `sql/vector.sql` (`STORAGE = external`)

**Primary — PostgreSQL 17 manual:**
- [Partial Indexes](https://www.postgresql.org/docs/17/indexes-partial.html) — implication, and "parameterized query clauses do not work with a partial index"
- [PREPARE](https://www.postgresql.org/docs/17/sql-prepare.html) — custom vs generic plans, `plan_cache_mode`
- [Table Partitioning](https://www.postgresql.org/docs/17/ddl-partitioning.html) — planning-time vs execution-time pruning

**Primary — IR literature:**
- G. V. Cormack, C. L. A. Clarke, S. Büttcher. *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods.* SIGIR '09. http://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf · https://dl.acm.org/doi/10.1145/1571941.1572114
- E. A. Fox, J. A. Shaw. *Combination of Multiple Searches.* TREC-2, 1994. https://trec.nist.gov/pubs/trec2/papers/txt/23.txt
- S. Bruch, S. Gai, A. Ingber. *An Analysis of Fusion Functions for Hybrid Retrieval.* ACM TOIS, 2023. https://arxiv.org/abs/2210.11934 · https://doi.org/10.1145/3596512

**Secondary — current practice (vendor docs, treated as evidence of practice, not of truth):**
- Elasticsearch, [Reciprocal rank fusion](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion) — `rank_constant` default 60
- Weaviate, [Hybrid search](https://docs.weaviate.io/weaviate/search/hybrid) and [fusion algorithms](https://weaviate.io/blog/hybrid-search-fusion-algorithms) — `relativeScoreFusion` default since v1.24, ~6% recall gain on FIQA
- Qdrant, [Hybrid queries](https://qdrant.tech/documentation/concepts/hybrid-queries/) — RRF and DBSF
- OpenSearch, [normalization processor](https://docs.opensearch.org/latest/search-plugins/search-pipelines/normalization-processor/) and [z-score announcement](https://opensearch.org/blog/introducing-the-z-score-normalization-technique-for-hybrid-search/) — `min_max` / `z_score`, `arithmetic_mean`, per-clause `weights`

**In-repo:**
- [`docs/research/pg-search-pgvector-apple-silicon.md`](pg-search-pgvector-apple-silicon.md) (branch `research/pg-search-pgvector-apple-silicon`, `33f1e3f`) — the executed baseline
- `docs/research/cross-encoder-reranker.md` (branch `research/cross-encoder-reranker`)
- [`docs/adr/0001-paradedb-for-hybrid-search.md`](../adr/0001-paradedb-for-hybrid-search.md)
