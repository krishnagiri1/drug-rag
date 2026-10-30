"""Build and load the two indexes for one chunk size.

data/index/cs1000/
    chunks.jsonl     the chunks (text + metadata), one per line
    dense.faiss      FAISS inner-product index over normalized embeddings (= cosine)
    meta.json        which embedder built it, sizes, build time

BM25 is rebuilt in memory at load time from chunks.jsonl (a few seconds), which keeps
the index portable across Python and library versions.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .chunking import Chunk, chunk_labels
from .config import INDEX_DIR
from .embeddings import Embedder


def index_dir(chunk_size: int, root: str | Path = INDEX_DIR) -> Path:
    return Path(root) / f"cs{chunk_size}"


def build_index(labels: list[dict], chunk_size: int, embedder: Embedder, root: str | Path = INDEX_DIR,
                progress: bool = True) -> Path:
    import faiss

    t0 = time.time()
    chunks = chunk_labels(labels, chunk_size)
    if progress:
        print(f"[cs{chunk_size}] {len(chunks)} chunks from {len(labels)} labels; embedding with {embedder.name} ...")
    texts = [c.text for c in chunks]
    vecs = []
    step = 2048
    for i in range(0, len(texts), step):
        vecs.append(embedder.embed_documents(texts[i : i + step]))
        if progress:
            print(f"  embedded {min(i + step, len(texts))}/{len(texts)}")
    import numpy as np

    mat = np.vstack(vecs).astype("float32")
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)

    out = index_dir(chunk_size, root)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    faiss.write_index(index, str(out / "dense.faiss"))
    (out / "meta.json").write_text(json.dumps({
        "embedder": embedder.name, "chunk_size": chunk_size, "n_chunks": len(chunks),
        "n_labels": len(labels), "dim": int(mat.shape[1]), "build_seconds": round(time.time() - t0, 1),
    }, indent=2))
    if progress:
        print(f"[cs{chunk_size}] saved to {out} in {time.time() - t0:.0f}s")
    return out


def load_chunks(path: str | Path) -> list[Chunk]:
    with (Path(path) / "chunks.jsonl").open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]


def load_meta(path: str | Path) -> dict:
    return json.loads((Path(path) / "meta.json").read_text())
