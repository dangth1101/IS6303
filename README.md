# IS6303

Hybrid search (BM25 + vector) over the
[`Shengtao/recipe`](https://huggingface.co/datasets/Shengtao/recipe) corpus.

## Database

A local Postgres with `pg_search` (BM25 full-text) and `pgvector`, run via Docker
Compose. Data persists in a named volume across `docker compose down`.

### Start it

```sh
cp .env.example .env     # then set POSTGRES_PASSWORD
docker compose up -d
```

The database is published on the port in `.env` (`5432` by default). If another
container or a local Postgres already owns that port, `up` fails with
`port is already allocated` — stop the other one, or change `POSTGRES_PORT`.

### Connect

```sh
docker compose exec db psql -U recipes -d recipes
```

Or from any client: `postgresql://recipes:<password>@localhost:5432/recipes`

### Verify the extensions are loaded

```sql
SELECT extname, extversion FROM pg_extension ORDER BY extname;
```

`pg_search` and `vector` should both appear. A BM25 smoke test, needing no
schema:

```sql
SELECT paradedb.tokenize(paradedb.tokenizer('default'), 'simple macaroni and cheese');
SELECT '[1,2,3]'::vector <-> '[3,2,1]'::vector AS distance;
```

### Stop it

```sh
docker compose down            # keeps the data
docker compose down -v         # deletes the volume, and the corpus with it
```

There is no schema yet — the table layout, chunking, and embedding model are
still being designed. See
[docs/adr/0001-paradedb-for-hybrid-search.md](docs/adr/0001-paradedb-for-hybrid-search.md)
for why this image and this version pin.
