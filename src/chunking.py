"""Chunking parametrizável — o objeto de estudo da otimização.

chunk_size e overlap_ratio NÃO são detalhes de implementação:
são hiperparâmetros que afetam diretamente recall, custo e latência.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    source: str


def load_corpus(corpus_dir: Path) -> dict[str, str]:
    """Carrega todos os .md do corpus como {nome_arquivo: texto}."""
    docs = {}
    for path in sorted(corpus_dir.glob("*.md")):
        docs[path.name] = path.read_text(encoding="utf-8")
    if not docs:
        raise FileNotFoundError(f"Nenhum .md encontrado em {corpus_dir}")
    return docs


def chunk_document(
    text: str,
    source: str,
    chunk_size: int,
    overlap_ratio: float,
) -> list[Chunk]:
    """Janela deslizante por caracteres com sobreposição.

    overlap_ratio controla quanto contexto é repetido entre chunks
    consecutivos. Overlap alto reduz o risco de cortar uma resposta
    ao meio, mas aumenta o número de chunks (custo de indexação).
    """
    if not 0.0 <= overlap_ratio < 1.0:
        raise ValueError("overlap_ratio deve estar em [0, 1)")
    step = max(int(chunk_size * (1.0 - overlap_ratio)), 32)

    chunks: list[Chunk] = []
    n = 0
    for start in range(0, len(text), step):
        segment = text[start : start + chunk_size].strip()
        if segment:
            chunks.append(Chunk(f"{source}::{n}", segment, source))
            n += 1
        if start + chunk_size >= len(text):
            break
    return chunks


def build_chunks(
    corpus: dict[str, str],
    chunk_size: int,
    overlap_ratio: float,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for source, text in corpus.items():
        chunks.extend(chunk_document(text, source, chunk_size, overlap_ratio))
    return chunks
