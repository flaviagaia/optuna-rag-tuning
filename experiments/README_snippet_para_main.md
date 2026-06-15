<!-- Acrescentar esta seção ao README.md principal, antes do rodapé "Part of my LinkedIn series" -->

### Extensão: pruning e benchmark de otimizadores (`experiments/`)

Dois experimentos adicionais que cobrem os posts **B20** e **B21**, na mesma base
de código e dados:

- **B20 — Pruning multi-fidelidade (ASHA):** `experiments/exp_pruning.py` corta
  50% das avaliações (894 vs 1800) preservando a qualidade do melhor trial
  (1.4663 vs 1.4661). 31 de 40 trials são podados.
- **B21 — Hyperopt vs Optuna:** `experiments/exp_benchmark.py` mostra empate
  técnico no ótimo (1.4661 vs 1.4659) — o objetivo importa mais que a biblioteca.

Detalhes e tabelas em [`experiments/README.md`](experiments/README.md).
