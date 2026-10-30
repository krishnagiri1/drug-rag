"""Section-aware chunking.

Chunks never cross section boundaries, so every chunk belongs to exactly one
(drug, section). Each chunk's text starts with a short header naming the drug and the
section. That header matters a lot for retrieval: a passage like "Do not exceed 4 g per
day" is useless if the retriever can't tell which drug it is about.

Splitting within a section uses LangChain's RecursiveCharacterTextSplitter, which prefers
paragraph, then sentence, then word boundaries.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .config import CHUNK_OVERLAP_RATIO, SECTIONS


@dataclass
class Chunk:
    chunk_id: str
    set_id: str
    generic_name: str
    brand_name: str
    section: str
    text: str  # header + passage; this is what gets embedded, searched, and shown

    def to_dict(self) -> dict:
        return asdict(self)


def drug_title(label: dict) -> str:
    g, b = label["generic_name"], label.get("brand_name") or ""
    return f"{g} ({b})" if b and b.lower() != g.lower() else g


def chunk_label(label: dict, chunk_size: int) -> list[Chunk]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * CHUNK_OVERLAP_RATIO),
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )
    title = drug_title(label)
    out: list[Chunk] = []
    for section, text in label["sections"].items():
        header = f"{title} | {SECTIONS.get(section, section)}\n"
        for i, piece in enumerate(splitter.split_text(text)):
            out.append(Chunk(
                chunk_id=f"{label['set_id']}:{section}:{i}",
                set_id=label["set_id"],
                generic_name=label["generic_name"],
                brand_name=label.get("brand_name") or "",
                section=section,
                text=header + piece.strip(),
            ))
    return out


def chunk_labels(labels: list[dict], chunk_size: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for lab in labels:
        chunks.extend(chunk_label(lab, chunk_size))
    return chunks
