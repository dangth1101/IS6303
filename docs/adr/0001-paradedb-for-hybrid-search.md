# ParadeDB for hybrid search

We need BM25 keyword search and vector similarity search over the same recipe
corpus in one database. We use the prebuilt `paradedb/paradedb` image, which
ships `pg_search` and `pgvector` already compiled and preloaded, rather than
Postgres' built-in `tsvector` search or a hand-rolled image.

## Considered options

- **Stock Postgres with `tsvector` + `pgvector`.** No third-party dependency,
  but `ts_rank` is not BM25: it has no document-length normalisation or corpus
  term statistics, which is most of what makes keyword ranking competitive in a
  hybrid setup.
- **A custom Dockerfile installing `pg_search` onto `pgvector/pgvector`.**
  Works — ParadeDB publishes per-major `.deb`s for amd64 and arm64 — but it
  costs a build step and ongoing version maintenance to end up with what the
  official image already provides.
- **ParadeDB's prebuilt image (chosen).** `pg_search` is a Rust extension that
  must sit in `shared_preload_libraries`, so it cannot simply be `CREATE
  EXTENSION`-ed onto a vanilla image; the official image handles that, bundles
  pgvector, and publishes native arm64.

## Consequences

- **The Postgres major is pinned to `pg17`, not `latest`.** `PGDATA` is
  version-scoped (`/var/lib/postgresql/data` on pg17 and below,
  `/var/lib/postgresql/18/docker` on pg18+). If `latest` were to move to a new
  major, the server would find no cluster at the new path and run `initdb` into
  a fresh, empty one — the corpus would still be on disk but invisible, with no
  error. Pinning the major keeps the path stable. The ParadeDB patch version
  still floats, so extension behaviour can change on `docker compose pull`.
- **The volume is mounted at `/var/lib/postgresql/`, not `.../data`.** The
  parent is correct for every tag; the conventional `/data` suffix would break
  persistence silently on pg18+.
- **ParadeDB Community is AGPL-3.0.** Every search feature we need (BM25,
  vector, hybrid, filtering, faceting) is in Community; only high availability
  and read replicas are gated. Running it behind our own code is unaffected;
  distributing a modified `pg_search` would be.
- **Extensions are created by the image, not by us.** Its bootstrap seeds
  `template1` as well as `POSTGRES_DB`, so any database created later inherits
  `pg_search` and `vector`. We own no extension DDL, but that also means a
  volume restored from elsewhere needs `CREATE EXTENSION` run by hand.
