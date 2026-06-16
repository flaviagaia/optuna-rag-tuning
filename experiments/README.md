# Experiments — pruning & benchmark de otimizadores

[🇧🇷 Português](#-português) · [🇺🇸 English](#-english)

Extensão do `optuna-rag-tuning`. Mesma base de código (`src/`), mesmo corpus e
golden dataset (`data/`), mesma função objetivo com consciência de custo.
Dois experimentos offline, testados, que viram conteúdo dos posts **B20** e **B21**.

---

## 🇧🇷 Português

### B20 — Pruning multi-fidelidade (ASHA / Successive Halving)

`exp_pruning.py`

**A ideia:** nem todo trial merece o orçamento de avaliação completo. Um trial
obviamente ruim pode ser interrompido cedo, liberando compute para os
promissores. A dimensão de fidelidade aqui é o **número de perguntas** do golden
dataset: avaliar uma config em 3 perguntas é barato e já dá um sinal; avaliar nas
15 é o orçamento cheio. O pruner observa o score parcial a cada rung (3 → 6 → 9 →
12 → 15 perguntas) e mata os trials sem futuro.

**Detalhe que muda tudo:** o golden vem ordenado por documento. Avaliar só as
3 primeiras perguntas testaria apenas um documento — sinal enviesado, que mataria
"late bloomers" bons em documentos ainda não vistos. Por isso embaralhamos o
golden uma vez com seed fixa, e cada rung passa a ser representativo.

**Resultado real (40 trials, seed 42):**

| Estratégia | Avaliações | Melhor score | Trials podados |
| ---------- | ---------- | ------------ | -------------- |
| Sem pruning (baseline) | 1800 | 1.4661 | 0 |
| Com ASHA | 894 | 1.4663 | 31/40 |

**50% menos avaliações, com a qualidade do melhor trial preservada** (1.4663 vs
1.4661, diferença dentro do ruído). A lição do post: pruning não é "rodar menos
trials", é rodar o MESMO número de trials gastando fundo só nos que importam.

```
python experiments/exp_pruning.py 40
```

### B21 — Hyperopt vs Optuna no mesmo problema

`exp_benchmark.py`

**A ideia:** "qual biblioteca de tuning usar?" é menos importante do que ter uma
função objetivo honesta. As duas líderes usam a mesma família de algoritmo (TPE).
Aqui rodamos as duas no mesmo espaço de busca, com a mesma função objetivo e o
mesmo orçamento.

**Resultado real (60 trials, seed 42):**

| Biblioteca | Melhor score | Melhor config |
| ---------- | ------------ | ------------- |
| Optuna | 1.4661 | chunk_size≈847, top_k=1, overlap≈0.29 |
| Hyperopt | 1.4659 | chunk_size≈853, top_k=1, overlap≈0.15 |

Convergência (melhor score até o trial N):

| trial | Optuna | Hyperopt |
| ----- | ------ | -------- |
| 5  | 1.4341 | 1.2907 |
| 10 | 1.4341 | 1.3232 |
| 20 | 1.4341 | 1.4566 |
| 30 | 1.4622 | 1.4659 |
| 60 | 1.4661 | 1.4659 |

**Empate técnico no ótimo.** As duas concordam na lição central — `top_k=1` com
chunks de ~850 caracteres. O Optuna estabiliza num bom platô mais cedo; o Hyperopt
arranca mais devagar e alcança. Conclusão do post: troque de biblioteca pela
ergonomia da API e pelo ecossistema (dashboard, storage, pruners), não esperando
um salto de qualidade — **o objetivo importa mais que o otimizador.**

```
python experiments/exp_benchmark.py 60
```

### Testes

```
pytest tests/test_experiments.py -v   # 4 testes
```

Cada lição é um invariante: o pruning economiza (≥25%), o pruning preserva a
qualidade (queda < 0.05), as duas bibliotecas convergem ao mesmo ótimo (|Δ| < 0.02)
e a função objetivo é determinística.

---

## 🇺🇸 English

### B20 — Multi-fidelity pruning (ASHA / Successive Halving)

`exp_pruning.py`

Not every trial deserves the full evaluation budget. The fidelity dimension is the
**number of golden questions**: scoring a config on 3 questions is cheap and
already informative; scoring on all 15 is the full budget. The pruner watches the
partial score at each rung (3 → 6 → 9 → 12 → 15) and kills hopeless trials early.

**The detail that matters:** the golden set is ordered by document, so the first 3
questions cover a single document — a biased low-fidelity signal that would kill
good "late bloomers". We shuffle the golden set once with a fixed seed so every
rung is representative.

**Actual result (40 trials, seed 42):** ASHA used **894 vs 1800 evaluations (50%
fewer)** while preserving the best score (1.4663 vs 1.4661, within noise); 31/40
trials pruned. The lesson: pruning is not "run fewer trials", it is running the
same number of trials but spending deeply only on the ones that matter.

### B21 — Hyperopt vs Optuna on the same problem

`exp_benchmark.py`

Both leading libraries use the same algorithm family (TPE). Run on the same search
space, objective and budget, they tie at the optimum (1.4661 vs 1.4659) and agree
on the core lesson — `top_k=1` with ~850-char chunks. Optuna plateaus earlier;
Hyperopt starts slower and catches up. Choose your library for API ergonomics and
ecosystem, not for a quality jump — **the objective matters more than the optimizer.**

### Tests

```
pytest tests/test_experiments.py -v   # 4 tests
```

### Referências científicas (crédito aos autores)

- Akiba et al. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. KDD.
- Bergstra et al. (2011). Algorithms for Hyper-Parameter Optimization. NeurIPS — TPE (base do Hyperopt).
- Li et al. (2020). A System for Massively Parallel Hyperparameter Tuning. MLSys — ASHA.
- Jamieson & Talwalkar (2016). Non-stochastic Best Arm Identification. AISTATS — Successive Halving.
