"""B20 — Pruning multi-fidelidade (ASHA / Successive Halving).

Tese do post: nem todo trial merece o orçamento completo de avaliação.
Um trial obviamente ruim pode ser MORTO cedo, liberando compute para
trials promissores. É o princípio de Successive Halving / ASHA.

A dimensão de fidelidade aqui é o NÚMERO DE PERGUNTAS do golden dataset.
Avaliar uma config em 3 perguntas é barato e já dá um sinal; avaliar nas
15 é o "orçamento cheio". O pruner observa o score parcial após cada
rung (3, 6, 9, 12, 15 perguntas) e interrompe trials sem futuro.

Métrica de custo: número de avaliações pergunta-config (proxy do que,
em produção, seria uma chamada de LLM por pergunta). Comparamos:
  - estudo SEM pruning  (NopPruner)   — todo trial roda os 5 rungs
  - estudo COM pruning  (ASHA)        — trials ruins param antes

Os dois usam o MESMO sampler (TPE) com a MESMA seed, então a diferença
é puramente o efeito do pruning.

Uso:
    python experiments/exp_pruning.py [n_trials]   # default: 40
"""

import random
import sys
from pathlib import Path

import optuna

SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))

from chunking import build_chunks, load_corpus  # noqa: E402
from evaluation import evaluate, load_golden  # noqa: E402
from retriever import TfidfRetriever  # noqa: E402

ROOT = Path(__file__).parent.parent
SEED = 42
COST_LAMBDA = 0.00004
RUNGS = [3, 6, 9, 12, 15]  # nº de perguntas avaliadas em cada rung

corpus = load_corpus(ROOT / "data" / "corpus")
golden = load_golden(ROOT / "data" / "golden_dataset.json")
# O golden vem ordenado por documento. Embaralhamos UMA vez com seed fixa
# para que cada rung (3, 6, 9... perguntas) seja representativo de todos os
# documentos — sem isso, o sinal de baixa fidelidade fica enviesado e o
# pruner mata "late bloomers" bons em documentos ainda não avaliados.
random.Random(SEED).shuffle(golden)


class EvalCounter:
    """Conta avaliações pergunta-config consumidas (proxy de custo)."""

    def __init__(self) -> None:
        self.count = 0


def make_objective(counter: EvalCounter):
    def objective(trial: optuna.Trial) -> float:
        chunk_size = trial.suggest_int("chunk_size", 128, 1200, log=True)
        overlap_ratio = trial.suggest_float("overlap_ratio", 0.0, 0.5)
        top_k = trial.suggest_int("top_k", 1, 10)
        ngram_max = trial.suggest_int("ngram_max", 1, 2)

        chunks = build_chunks(corpus, chunk_size, overlap_ratio)
        retriever = TfidfRetriever(chunks, ngram_max=ngram_max)
        context_cost = top_k * chunk_size

        score = 0.0
        for step, n_q in enumerate(RUNGS):
            subset = golden[:n_q]
            metrics = evaluate(retriever, subset, top_k)
            counter.count += n_q  # custo: n_q avaliações neste rung
            score = (
                metrics["mrr"]
                + 0.5 * metrics["recall_at_k"]
                - COST_LAMBDA * context_cost
            )
            trial.report(score, step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trial.set_user_attr("recall_at_k", round(metrics["recall_at_k"], 4))
        trial.set_user_attr("context_chars", context_cost)
        return score

    return objective


def run(pruner: optuna.pruners.BasePruner, n_trials: int) -> tuple[optuna.Study, int]:
    counter = EvalCounter()
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=pruner,
    )
    study.optimize(make_objective(counter), n_trials=n_trials)
    return study, counter.count


def main() -> None:
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 40

    baseline, base_cost = run(optuna.pruners.NopPruner(), n_trials)
    asha, asha_cost = run(
        optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=3),
        n_trials,
    )

    n_pruned = len([t for t in asha.trials if t.state == optuna.trial.TrialState.PRUNED])
    saving = 100 * (1 - asha_cost / base_cost)

    print("\n" + "=" * 64)
    print(f"PRUNING MULTI-FIDELIDADE (ASHA) — {n_trials} trials, seed {SEED}")
    print("=" * 64)
    print(f"  Sem pruning : {base_cost:5d} avaliações | melhor score {baseline.best_value:.4f}")
    print(f"  Com ASHA    : {asha_cost:5d} avaliações | melhor score {asha.best_value:.4f}")
    print(f"  Trials podados: {n_pruned}/{n_trials}")
    print(f"  Economia de avaliações: {saving:.1f}%")
    print(f"  Queda de qualidade: {baseline.best_value - asha.best_value:+.4f}")
    print("\n  Melhor config (ASHA):")
    for k, v in asha.best_params.items():
        print(f"    {k:<14} = {v}")


if __name__ == "__main__":
    main()
