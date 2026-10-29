# Time-Zone-Ordered Spillover Networks

Code accompanying the paper:

> M. Parwani, S. Venu, *Time-Zone-Ordered Spillover Networks for Forecasting Volatility in Global Technology Equities: A Clock-Masked Latent-Graph Learner with an Identification Analysis of Contemporaneous Spillovers*.

School of Computer Science and Engineering (SCOPE), Vellore Institute of Technology (VIT), Vellore, Tamil Nadu, India.

## What is here

| File | Contents | Paper equations |
|---|---|---|
| `tzspill/volatility.py` | Overnight return, range estimators, daily target | (1)–(6) |
| `tzspill/clock_mask.py` | Closing ranks and the clock mask | (8)–(9) |
| `tzspill/synthetic.py` | DAG count, synthetic SVAR, NOTEARS, AUROC, F1, violation rate | (10), (52)–(57) |
| `tzspill/forecaster.py` | Masked latent-graph forecaster and training loop | (11)–(24) |
| `tzspill/leakage_test.py` | Three-condition leakage gate | (25) |
| `tzspill/baselines.py` | Random walk, AR(1), HAR | (26)–(27) |
| `tzspill/tvp_var.py` | TVP-VAR filter and connectedness | (30)–(38) |
| `tzspill/connectedness.py` | Generalized FEVD, Proposition 1, clock classification | (36), (39)–(41) |
| `tzspill/evaluation.py` | Folds, metrics, Diebold–Mariano, Spearman, ablation shares | (42)–(51) |
| `run_all.py` | Runs every component and prints the checks | — |
| `figures.py` | Redraws the paper's figures from the recorded results | — |
| `run_output.txt` | Output of `run_all.py` | — |

## Run

```
pip install -r requirements.txt
python run_all.py
python figures.py
```

Runs on a laptop CPU in under a minute. No GPU needed.

## Data

The market panel (daily OHLC for twelve instruments, January 2015 – August 2026) comes from public data providers and is not redistributed here. Parts of `run_all.py` that need real prices run on synthetic data and say so in their output. Deterministic results (the 41-of-132 mask, the DAG count, Proposition 1, the nested-ablation shares, the leakage gate) match the paper exactly.

## Citation

See `CITATION.cff`, or use the DOI badge once the Zenodo archive is created.

## License

MIT. See `LICENSE`.
