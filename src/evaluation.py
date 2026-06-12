"""Avaliação de retrieval contra um golden dataset.

Um item do golden dataset tem:
- question: pergunta como o usuário faria
- answer_span: trecho literal do corpus que responde a pergunta
- source: documento onde a resposta vive

Um chunk é considerado relevante se contém o answer_span completo.
Isso captura o efeito real do chunking: chunks pequenos demais
CORTAM a resposta ao meio e zeram o recall, mesmo com bom retrieval.
"""

import json
import unicodedata
from pathlib import Path

from chunking import Chunk
from retriever import TfidfRetriever


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def load_golden(path: Path) -> list[dict]:
    items = json.loads(path.read_text(encoding="utf-8"))
    required = {"question", "answer_span", "source"}
    for i, item in enumerate(items):
        missing = required - item.keys()
        if missing:
            raise ValueError(f"Item {i} do golden dataset sem campos {missing}")
    return items


def is_relevant(chunk: Chunk, item: dict) -> bool:
    return _normalize(item["answer_span"]) in _normalize(chunk.text)


def evaluate(
    retriever: TfidfRetriever,
    golden: list[dict],
    top_k: int,
) -> dict[str, float]:
    """Calcula recall@k e MRR sobre o golden dataset."""
    hits, reciprocal_ranks = [], []
    for item in golden:
        results = retriever.search(item["question"], top_k)
        rank = next(
            (r for r, (chunk, _) in enumerate(results, start=1) if is_relevant(chunk, item)),
            None,
        )
        hits.append(1.0 if rank else 0.0)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
    n = len(golden)
    return {
        "recall_at_k": sum(hits) / n,
        "mrr": sum(reciprocal_ranks) / n,
    }
