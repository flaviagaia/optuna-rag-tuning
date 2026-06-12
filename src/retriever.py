"""Retriever lexical TF-IDF.

Por que TF-IDF e não embeddings neurais?
1. A metodologia de tuning é idêntica — só muda o vetorizador.
2. Roda 100% offline, sem API key e sem custo, o que torna o
   experimento reprodutível por qualquer pessoa.
3. Permite centenas de trials do Optuna em segundos.

Para usar embeddings reais, basta substituir esta classe por uma
implementação com o mesmo contrato (build + search).
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from chunking import Chunk


class TfidfRetriever:
    def __init__(self, chunks: list[Chunk], ngram_max: int = 1) -> None:
        if not chunks:
            raise ValueError("Lista de chunks vazia")
        self.chunks = chunks
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, ngram_max),
            strip_accents="unicode",
        )
        self._matrix = self._vectorizer.fit_transform(c.text for c in chunks)

    def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        """Retorna os top_k chunks por similaridade de cosseno."""
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix).ravel()
        order = scores.argsort()[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in order]
