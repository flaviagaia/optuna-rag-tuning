"""Testes do pipeline de tuning. Rode com: pytest tests/ -v"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from chunking import build_chunks, chunk_document, load_corpus  # noqa: E402
from evaluation import evaluate, is_relevant, load_golden  # noqa: E402
from retriever import TfidfRetriever  # noqa: E402


def test_chunking_respeita_tamanho():
    chunks = chunk_document("a" * 1000, "doc.md", chunk_size=200, overlap_ratio=0.0)
    assert all(len(c.text) <= 200 for c in chunks)
    assert len(chunks) == 5


def test_overlap_gera_mais_chunks():
    text = "palavra " * 500
    sem = chunk_document(text, "d.md", 300, 0.0)
    com = chunk_document(text, "d.md", 300, 0.4)
    assert len(com) > len(sem)


def test_overlap_invalido_levanta_erro():
    import pytest

    with pytest.raises(ValueError):
        chunk_document("texto", "d.md", 200, 1.5)


def test_golden_spans_existem_no_corpus():
    """Garante a integridade do golden dataset: todo answer_span
    precisa existir literalmente no documento de origem."""
    corpus = load_corpus(ROOT / "data" / "corpus")
    golden = load_golden(ROOT / "data" / "golden_dataset.json")
    for item in golden:
        assert item["source"] in corpus, f"{item['source']} não existe"
        assert item["answer_span"] in corpus[item["source"]], (
            f"Span não encontrado em {item['source']}: {item['answer_span']!r}"
        )


def test_retrieval_com_config_razoavel_tem_recall_alto():
    corpus = load_corpus(ROOT / "data" / "corpus")
    golden = load_golden(ROOT / "data" / "golden_dataset.json")
    chunks = build_chunks(corpus, chunk_size=600, overlap_ratio=0.3)
    retriever = TfidfRetriever(chunks, ngram_max=1)
    metrics = evaluate(retriever, golden, top_k=5)
    assert metrics["recall_at_k"] >= 0.7, metrics
    assert 0.0 <= metrics["mrr"] <= 1.0


def test_chunk_pequeno_demais_corta_respostas():
    """O fenômeno central do projeto: chunk minúsculo quebra spans."""
    corpus = load_corpus(ROOT / "data" / "corpus")
    golden = load_golden(ROOT / "data" / "golden_dataset.json")
    pequeno = build_chunks(corpus, chunk_size=64, overlap_ratio=0.0)
    adequado = build_chunks(corpus, chunk_size=600, overlap_ratio=0.3)
    cobertura = lambda cs: sum(  # noqa: E731
        any(is_relevant(c, item) for c in cs) for item in golden
    )
    assert cobertura(pequeno) < cobertura(adequado)
