# RAG as a Bayesian Optimization Problem — Optuna + Cost-Aware Objective

[🇧🇷 Português](#-português) · [🇺🇸 English](#-english)

Python 3.10+ · Optuna (TPE) · scikit-learn · 100% offline, sem API key · MIT License

---

## 🇧🇷 Português

### A tese

chunk_size, overlap, top_k não são "detalhes de implementação" de um RAG.
São **hiperparâmetros** — e podem ser otimizados como qualquer hiperparâmetro de ML: com otimização bayesiana, função objetivo explícita e golden dataset.

Este projeto demonstra o método de ponta a ponta, executável em segundos na sua máquina, sem nenhuma API key.

### Espaço de busca

| Hiperparâmetro | Faixa | Observação |
|---|---|---|
| `chunk_size` | 128–1200 (escala log) | efeito multiplicativo → busca em log |
| `overlap_ratio` | 0.0–0.5 | overlap evita cortar respostas ao meio |
| `top_k` | 1–10 | cada chunk a mais custa tokens |
| `ngram_max` | 1–2 | unigrams vs bigrams no vetorizador |

### Função objetivo com consciência de custo

```
score = MRR + 0.5 · recall@k − λ · (top_k · chunk_size)
```

Sem o termo de penalidade, o otimizador "trapaceia": `top_k=10` com chunks gigantes maximiza recall — e explode o custo por pergunta. A penalidade força a pergunta certa: **qual o melhor retrieval pelo MENOR contexto?**

Resultado real deste repositório (40 trials, seed fixa): recall@k = 1.0 e MRR = 1.0 com **top_k = 1** e chunks de ~850 caracteres. Mesma qualidade que uma config ingênua de `top_k=10`, com ~10× menos contexto enviado ao LLM.

### Metodologia de avaliação

O golden dataset (`data/golden_dataset.json`) tem pares pergunta → `answer_span`, onde o span é um trecho **literal** do corpus. Um chunk é relevante se contém o span completo. Isso captura o fenômeno central do chunking: chunks pequenos demais cortam a resposta ao meio e zeram o recall, mesmo com retrieval "bom". O teste `test_golden_spans_existem_no_corpus` garante a integridade do dataset.

### Por que TF-IDF e não embeddings neurais?

A metodologia é idêntica — só muda o vetorizador. TF-IDF roda offline, sem custo, e permite centenas de trials em segundos, o que torna o experimento reprodutível por qualquer pessoa. Para usar embeddings reais, substitua `TfidfRetriever` por uma classe com o mesmo contrato (`build` + `search`) e reduza `n_trials`.

### Execução

```bash
pip install -r requirements.txt
pytest tests/ -v          # 6 testes, inclusive integridade do golden dataset
python src/tune.py 60     # otimização (60 trials, ~10s)
```

### Estrutura

```
data/
├── corpus/               # 4 políticas internas (corpus de exemplo)
└── golden_dataset.json   # 15 perguntas com answer_span literal
src/
├── chunking.py           # chunking parametrizável (o objeto de estudo)
├── retriever.py          # TF-IDF retriever (contrato substituível)
├── evaluation.py         # recall@k e MRR contra o golden dataset
└── tune.py               # estudo Optuna com TPESampler e seed fixa
tests/
└── test_pipeline.py      # testes de unidade + integridade de dados
```

### Limitações honestas

Corpus pequeno e lexical (TF-IDF favorece match de vocabulário). Em produção: use embeddings reais, valide em corpus maior, separe golden dataset de tuning e de teste (overfitting do conjunto de validação existe aqui também) e re-otimize quando a base de conhecimento mudar.

---

## 🇺🇸 English

### The thesis

chunk_size, overlap and top_k are not RAG "implementation details".
They are **hyperparameters** — and can be optimized like any ML hyperparameter: Bayesian optimization, an explicit objective function and a golden dataset.

This project demonstrates the method end to end, running in seconds on your machine with no API key.

### Search space

| Hyperparameter | Range | Note |
|---|---|---|
| `chunk_size` | 128–1200 (log scale) | multiplicative effect → log search |
| `overlap_ratio` | 0.0–0.5 | overlap avoids splitting answers |
| `top_k` | 1–10 | every extra chunk costs tokens |
| `ngram_max` | 1–2 | unigrams vs bigrams in the vectorizer |

### Cost-aware objective

```
score = MRR + 0.5 · recall@k − λ · (top_k · chunk_size)
```

Without the penalty term the optimizer cheats: `top_k=10` with huge chunks maximizes recall — and explodes per-question cost. The penalty forces the right question: **what is the best retrieval for the SMALLEST context?**

Actual result in this repo (40 trials, fixed seed): recall@k = 1.0 and MRR = 1.0 with **top_k = 1** and ~850-char chunks. Same quality as a naive `top_k=10` config with ~10× less context sent to the LLM.

### Evaluation methodology

The golden dataset maps question → `answer_span`, where the span is a **literal** excerpt of the corpus. A chunk is relevant iff it contains the full span. This captures the core chunking failure mode: too-small chunks split the answer and zero out recall even with "good" retrieval. A dedicated test guarantees dataset integrity.

### Why TF-IDF instead of neural embeddings?

The methodology is identical — only the vectorizer changes. TF-IDF runs offline, free, enabling hundreds of trials in seconds and full reproducibility. To use real embeddings, swap `TfidfRetriever` for any class with the same contract and lower `n_trials`.

### Running

```bash
pip install -r requirements.txt
pytest tests/ -v
python src/tune.py 60
```

### Honest limitations

Small, lexical corpus (TF-IDF favors vocabulary overlap). In production: real embeddings, larger corpora, separate tuning vs test golden sets (validation-set overfitting applies here too), and re-optimization whenever the knowledge base changes.

---

Part of my LinkedIn series on RAG efficiency → [Flávia Gaia](https://www.linkedin.com/in/flavia-gaia/)
