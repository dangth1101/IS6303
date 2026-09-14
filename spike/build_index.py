"""Build the spike index: sample the corpus, embed it, write vectors + metadata to disk.

Throwaway. See issue #12. Not the real ingest pipeline.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

CSV_URL = "https://huggingface.co/datasets/Shengtao/recipe/resolve/main/recipe.csv"
MODEL = "nomic-ai/nomic-embed-text-v1.5"
OUT = Path(__file__).parent / "index"

# The document definition settled while charting the map: title + description +
# ingredients, with `directions` deliberately excluded. Every retrieval arm must
# see this identical string, so it is built in exactly one place.
DOC_FIELDS = ("title", "description", "ingredients")


def build_document(row: pd.Series) -> str:
    parts = [str(row[f]).strip() for f in DOC_FIELDS if pd.notna(row[f])]
    return "\n".join(p for p in parts if p and p.lower() != "nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=6303)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)

    print(f"reading {CSV_URL}")
    df = pd.read_csv(CSV_URL)
    print(f"  {len(df):,} rows x {len(df.columns)} columns")

    # Rows with no usable title are unretrievable; drop before sampling so the
    # sample size is the number of documents actually indexed.
    df = df[df["title"].notna() & (df["title"].astype(str).str.strip() != "")]

    # Uniform random, NOT evenly-spaced offsets: issue #3 found that null-image
    # rows cluster in blocks, so systematic sampling misrepresents this corpus.
    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=args.seed)
    df = df.reset_index(drop=True)

    docs = [build_document(r) for _, r in df.iterrows()]
    print(f"  {len(docs):,} documents, mean {np.mean([len(d) for d in docs]):.0f} chars")

    print(f"loading {MODEL}")
    model = SentenceTransformer(MODEL, trust_remote_code=True)
    print(f"  device={model.device}")

    # nomic-embed is prefix-conditioned: documents and queries MUST carry
    # different task prefixes or retrieval quality degrades badly.
    t0 = time.perf_counter()
    vectors = model.encode(
        [f"search_document: {d}" for d in docs],
        batch_size=args.batch_size,
        normalize_embeddings=True,  # so cosine similarity is a plain dot product
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    elapsed = time.perf_counter() - t0
    print(f"  encoded {len(docs):,} docs in {elapsed:.1f}s ({len(docs) / elapsed:.1f}/s)")
    print(f"  shape {vectors.shape}")

    np.save(OUT / "vectors.npy", vectors)
    keep = ["title", "description", "ingredients", "category", "url", "rating", "total_time"]
    with (OUT / "docs.jsonl").open("w") as fh:
        for i, (_, row) in enumerate(df.iterrows()):
            record = {"id": i, "document": docs[i]}
            record.update({k: (None if pd.isna(row.get(k)) else row.get(k)) for k in keep})
            fh.write(json.dumps(record) + "\n")

    print(f"wrote {OUT}/vectors.npy and {OUT}/docs.jsonl")


if __name__ == "__main__":
    main()
