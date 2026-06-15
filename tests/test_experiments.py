"""Testes dos experimentos da extensão (B20 pruning, B21 benchmark).

Cada lição vira um invariante garantido por teste:
- pruning ASHA gasta menos avaliações que o baseline (B20)
- pruning não destrói a qualidade do melhor trial (B20)
- Hyperopt e Optuna chegam ao mesmo ótimo no mesmo objetivo (B21)
- a função objetivo compartilhada é determinística
"""

import sys
from pathlib import Path

import optuna

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))

optuna.logging.set_verbosity(optuna.logging.WARNING)

import exp_benchmark  # noqa: E402
import exp_pruning  # noqa: E402


def test_pruning_economiza_avaliacoes():
    """ASHA deve consumir menos avaliações que o baseline sem pruning."""
    _, base_cost = exp_pruning.run(optuna.pruners.NopPruner(), n_trials=30)
    _, asha_cost = exp_pruning.run(
        optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=3),
        n_trials=30,
    )
    assert asha_cost < base_cost
    assert asha_cost <= 0.75 * base_cost  # economia relevante (>= 25%)


def test_pruning_preserva_qualidade():
    """O pruning não pode derrubar o melhor score de forma significativa."""
    baseline, _ = exp_pruning.run(optuna.pruners.NopPruner(), n_trials=30)
    asha, _ = exp_pruning.run(
        optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=3),
        n_trials=30,
    )
    assert baseline.best_value - asha.best_value < 0.05


def test_benchmark_mesmo_otimo():
    """Hyperopt e Optuna, no mesmo objetivo, devem convergir ao mesmo ótimo."""
    o_best, o_params, _ = exp_benchmark.run_optuna(n_trials=40)
    h_best, h_params, _ = exp_benchmark.run_hyperopt(n_trials=40)
    assert o_best > 1.4 and h_best > 1.4
    assert abs(o_best - h_best) < 0.02
    # As duas bibliotecas concordam na lição central: top_k pequeno.
    assert o_params["top_k"] <= 2 and h_params["top_k"] <= 2


def test_objetivo_compartilhado_deterministico():
    """A função objetivo única deve ser reprodutível para os mesmos params."""
    a = exp_benchmark.score_config(800, 0.2, 1, 1)
    b = exp_benchmark.score_config(800, 0.2, 1, 1)
    assert a == b
