"""B21 — Hyperopt vs Optuna no mesmo problema.

Tese do post: "qual biblioteca de tuning usar?" é menos importante do que
ter uma função objetivo honesta. As duas líderes — Hyperopt e Optuna —
usam a MESMA família de algoritmo (TPE, Tree-structured Parzen Estimator).
Aqui rodamos as duas no MESMO espaço de busca, com a MESMA função objetivo
com consciência de custo e o MESMO orçamento, e comparamos:
  - melhor score alcançado
  - velocidade de convergência (melhor score até o trial N)
  - ergonomia de API (ver o código)

Uso:
    python experiments/exp_benchmark.py [n_trials]   # default: 60
"""

import math
import sys
from pathlib import Path

import numpy as np
import optuna
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))

from chunking import build_chunks, load_corpus  # noqa: E402
from evaluation import evaluate, load_golden  # noqa: E402
from retriever import TfidfRetriever  # noqa: E402

ROOT = Path(__file__).parent.parent
SEED = 42
COST_LAMBDA = 0.00004

corpus = load_corpus(ROOT / "data" / "corpus")
golden = load_golden(ROOT / "data" / "golden_dataset.json")


def score_config(chunk_size: int, overlap_ratio: float, top_k: int, ngram_max: int) -> float:
    """Função objetivo única, compartilhada pelas duas bibliotecas."""
    chunks = build_chunks(corpus, chunk_size, overlap_ratio)
    retriever = TfidfRetriever(chunks, ngram_max=ngram_max)
    metrics = evaluate(retriever, golden, top_k)
    context_cost = top_k * chunk_size
    return metrics["mrr"] + 0.5 * metrics["recall_at_k"] - COST_LAMBDA * context_cost


# ---------------------------------------------------------------- Optuna
def run_optuna(n_trials: int) -> tuple[float, dict, list[float]]:
    def objective(trial: optuna.Trial) -> float:
        return score_config(
            trial.suggest_int("chunk_size", 128, 1200, log=True),
            trial.suggest_float("overlap_ratio", 0.0, 0.5),
            trial.suggest_int("top_k", 1, 10),
            trial.suggest_int("ngram_max", 1, 2),
        )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=n_trials)
    best_so_far, running = [], -math.inf
    for t in study.trials:
        running = max(running, t.value)
        best_so_far.append(running)
    return study.best_value, study.best_params, best_so_far


# -------------------------------------------------------------- Hyperopt
def run_hyperopt(n_trials: int) -> tuple[float, dict, list[float]]:
    space = {
        "chunk_size": hp.qloguniform("chunk_size", math.log(128), math.log(1200), 1),
        "overlap_ratio": hp.uniform("overlap_ratio", 0.0, 0.5),
        "top_k": hp.quniform("top_k", 1, 10, 1),
        "ngram_max": hp.choice("ngram_max", [1, 2]),
    }

    # Hyperopt MINIMIZA, então devolvemos -score (objetivo idêntico).
    def objective(params: dict) -> dict:
        score = score_config(
            int(params["chunk_size"]),
            float(params["overlap_ratio"]),
            int(params["top_k"]),
            int(params["ngram_max"]),
        )
        return {"loss": -score, "status": STATUS_OK}

    trials = Trials()
    fmin(
        objective,
        space,
        algo=tpe.suggest,
        max_evals=n_trials,
        trials=trials,
        rstate=np.random.default_rng(SEED),
        show_progressbar=False,
    )
    losses = [r["loss"] for r in trials.results]
    best_so_far, running = [], math.inf
    for loss in losses:
        running = min(running, loss)
        best_so_far.append(-running)
    best_idx = int(np.argmin(losses))
    best_params = trials.trials[best_idx]["misc"]["vals"]
    flat = {k: (v[0] if v else None) for k, v in best_params.items()}
    flat["ngram_max"] = [1, 2][int(flat["ngram_max"])]  # hp.choice -> índice
    flat["chunk_size"] = int(flat["chunk_size"])
    flat["top_k"] = int(flat["top_k"])
    flat["overlap_ratio"] = round(float(flat["overlap_ratio"]), 4)
    return -min(losses), flat, best_so_far


def main() -> None:
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 60

    o_best, o_params, o_curve = run_optuna(n_trials)
    h_best, h_params, h_curve = run_hyperopt(n_trials)

    print("\n" + "=" * 64)
    print(f"HYPEROPT vs OPTUNA — {n_trials} trials, seed {SEED}, mesmo objetivo")
    print("=" * 64)
    print(f"  Optuna   : melhor score {o_best:.4f} | {o_params}")
    print(f"  Hyperopt : melhor score {h_best:.4f} | {h_params}")
    print("\n  Convergência (melhor score até o trial N):")
    print(f"  {'trial':>6} | {'Optuna':>8} | {'Hyperopt':>8}")
    for n in [5, 10, 20, 30, 40, 60]:
        if n <= n_trials:
            print(f"  {n:>6} | {o_curve[n-1]:>8.4f} | {h_curve[n-1]:>8.4f}")


if __name__ == "__main__":
    main()
