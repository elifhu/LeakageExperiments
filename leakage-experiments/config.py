from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    symbol: str = "EURUSD=X"
    interval: str = "1m"
    lookback_days: int = 30


@dataclass(frozen=True)
class TradeConfig:
    roll: int = 10
    z_threshold: float = 2.5
    trade_min_gap_s: int = 300
    pre_window_s: int = 60
    post_window_s: int = 60
    liq_lookback_s: int = 30 * 60
    min_points: int = 2
    pip_scale: float = 1e4


@dataclass(frozen=True)
class GraphConfig:
    tau: float = 10800.0
    gamma: float = 2.0
    k_nn: int = 20
    alpha_bt: float = 1.0
    alpha_tt: float = 1.0
    diffusion_times: tuple[float, ...] = (30.0, 60.0, 120.0, 300.0)
    k_eigs: int = 80
    nmax: int = 800


@dataclass(frozen=True)
class SyntheticCounterpartyConfig:
    n_counterparties: int = 80
    tail_df: float = 3.0
    tail_scale: float = 1.0
    beta: float = 1.2
    base_mix: float = 0.05
    alpha: float = 1.0
    noise: float = 0.25
    p_align: float = 0.7
    seed_counterparties: int = 42
    seed_signal: int = 123


DATA_CONFIG = DataConfig()
TRADE_CONFIG = TradeConfig()
GRAPH_CONFIG = GraphConfig()
SYNTHETIC_CONFIG = SyntheticCounterpartyConfig()