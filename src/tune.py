"""Otimização bayesiana dos hiperparâmetros do RAG com Optuna.

Espaço de busca:
- chunk_size     [128, 1200] (escala log: o efeito é multiplicativo)
- overlap_ratio  [0.0, 0.5]
- top_k          [1, 10]
- ngram_max      {1, 2}

Função objetivo COM CONSCIÊNCIA DE CUSTO:
    score = MRR + 0.5 * recall@k - lambda * (top_k * chunk_size)

O termo de penalidade (top_k * chunk_size é um proxy do nº de
caracteres enviados ao LLM por pergunta) impede que o otimizador
"trapaceie" maximizando contexto: recall@k=1.0 com top_k=10 e
chunks gigantes é fácil — e caro. Queremos o melhor retrieval
pelo MENOR contexto.

Uso:
    python src/tune.py [n_trials]   # default: 60
"""

import sys
from pathlib import Path

import optuna

from chunking import build_chunks, load_corpus
from evaluation import evaluate, load_golden
from retriever import TfidfRetriever

ROOT = Path(__file__).parent.parent
COST_LAMBDA = 0.00004  # peso da penalidade por caractere de contexto
SEED = 42

corpus = load_corpus(ROOT / "data" / "corpus")
golden = load_golden(ROOT / "data" / "golden_dataset.json")


def objective(trial: optuna.Trial) -> float:
    chunk_size = trial.suggest_int("chunk_size", 128, 1200, log=True)
    overlap_ratio = trial.suggest_float("overlap_ratio", 0.0, 0.5)
    top_k = trial.suggest_int("top_k", 1, 10)
    ngram_max = trial.suggest_int("ngram_max", 1, 2)

    chunks = build_chunks(corpus, chunk_size, overlap_ratio)
    retriever = TfidfRetriever(chunks, ngram_max=ngram_max)
    metrics = evaluate(retriever, golden, top_k)

    context_cost = top_k * chunk_size
    score = metrics["mrr"] + 0.5 * metrics["recall_at_k"] - COST_LAMBDA * context_cost

    # Métricas brutas ficam auditáveis em cada trial
    trial.set_user_attr("recall_at_k", round(metrics["recall_at_k"], 4))
    trial.set_user_attr("mrr", round(metrics["mrr"], 4))
    trial.set_user_attr("context_chars", context_cost)
    trial.set_user_attr("n_chunks", len(chunks))
    return score


def main() -> None:
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 60

    study = optuna.create_study(
        study_name="rag-retrieval-tuning",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),  # reprodutível
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    print("\n" + "=" * 64)
    print(f"Melhor trial (#{best.number}) — score {best.value:.4f}")
    print("=" * 64)
    for k, v in best.params.items():
        print(f"  {k:<14} = {v}")
    print("  ---")
    for k, v in best.user_attrs.items():
        print(f"  {k:<14} = {v}")

    # Top 5 para discussão de trade-offs
    print("\nTop 5 configurações:")
    ranked = sorted(study.trials, key=lambda t: t.value or -1, reverse=True)[:5]
    for t in ranked:
        print(
            f"  #{t.number:<3} score={t.value:.4f} "
            f"recall={t.user_attrs['recall_at_k']:.2f} mrr={t.user_attrs['mrr']:.2f} "
            f"contexto={t.user_attrs['context_chars']} chars | {t.params}"
        )

    out = ROOT / "results.csv"
    study.trials_dataframe().to_csv(out, index=False)
    print(f"\nTodos os trials salvos em {out.name}")


if __name__ == "__main__":
    main()
