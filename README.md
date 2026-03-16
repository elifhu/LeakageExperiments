# Graph-Based Modelling of Information Leakage and Counterparty Trading Behaviour in Electronic Markets

Work in progress research project.

This repository contains code and a preliminary research framework for analysing information leakage and counterparty trading behaviour in electronic financial markets using graph-based methods.

The project studies how post-trade price responses (integrated reval) propagate across related trades and counterparties, and how structural representations of counterparty behaviour can be learned from trading interactions.

The goal is to model information leakage not as an isolated trade-level phenomenon but as a structural process emerging from interactions between trades and counterparties.

---

# Research Motivation

Information leakage occurs when trading activity reveals information that subsequently propagates into market prices. This produces systematic post-trade price responses and creates adverse selection for liquidity providers.

At the same time, different counterparties exhibit distinct trading behaviours and interaction patterns that shape how information propagates through the market. Understanding these behavioural patterns is important for detecting toxic trading flow and modelling trading interactions.

Traditional leakage diagnostics analyse trades independently using regressions or statistical tests. However, trades in electronic markets are structurally related through temporal proximity, shared market conditions and repeated counterparty activity.

This project models leakage as a **signal propagation process on a trade interaction graph**, while simultaneously learning structural representations of **counterparty trading behaviour** from trading interactions.

---

# Methodology

The framework consists of three main components.

## 1. Classical Baseline Diagnostics

Traditional trade-level leakage diagnostics used as reference benchmarks.

- multivariate regression on integrated reval
- nonparametric statistical tests (Mann–Whitney)
- baseline visual diagnostics

These methods treat trades as independent observations and therefore cannot capture structural dependencies across trades.

---

## 2. Graph-Based Leakage Diagnostics

Trades are embedded into a similarity graph capturing both temporal proximity and market-state conditions.

The resulting structure is extended into a **heterogeneous trade–counterparty graph** linking trades to counterparties.

Graph signal processing methods are then used to analyse leakage behaviour:

- trade similarity graph construction
- heterogeneous trade–counterparty graph
- spectral diffusion analysis
- graph signal diagnostics
  - low-pass leakage propagation component
  - high-pass residual signal
  - local graph energy

These diagnostics reveal structural patterns in leakage behaviour across related trades.

---

## 3. Representation Learning (Ongoing Work)

The spectral framework provides interpretable diagnostics but relies on fixed similarity kernels.

To learn interaction structures directly from data, the project extends the framework using **heterogeneous graph neural networks**.

Planned models include:

- heterogeneous Graph Attention Networks (GAT)
- counterparty representation learning
- structural embeddings of counterparty trading behaviour
- learned interaction patterns across trades and counterparties

These models aim to learn counterparty trading styles and toxicity patterns directly from trading interactions.

---

# Repository Structure

leakage-experiments/

data/
cached or processed market data

diagnostics/
analysis outputs and intermediate diagnostics

leakage/
leakage-specific computations and graph signal processing logic

models/
graph neural network models (GAT architecture under development)

outputs/
generated figures and tables from experiments

scripts/
main experiment pipelines and plotting scripts

utils/
helper utilities and shared functions

config.py
experiment configuration

paths.py
repository path management

requirements.txt
project dependencies

---

# Running the Experiments

Example pipelines:

Build trade dataset
python scripts/run_build_trade_table.py

Regression baseline
python scripts/run_regression_pipeline.py

Graph spectral diagnostics
python scripts/run_gsp_pipeline.py

Trade graph analysis
python scripts/run_trade_graph_analysis.py

Generate figures
python scripts/make_spectral_gsp_flow_figure.py


---

# Status

This repository contains ongoing research work.

Future work focuses on learning counterparty trading behaviour representations using heterogeneous graph neural networks.

---

# Research Context

This project is part of ongoing research on market microstructure, information leakage and graph-based modelling of trading interactions.