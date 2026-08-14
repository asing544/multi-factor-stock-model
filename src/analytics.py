"""Performance analytics and reporting for the backtest."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class PerformanceSummary:
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    avg_monthly_return: float
    information_ratio: float | None
    alpha: float | None
    beta: float | None


def max_drawdown(cumulative: pd.Series) -> float:
    peak = cumulative.cummax()
    dd = (cumulative - peak) / peak
    return float(dd.min())


def compute_performance(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = 0.04,
) -> PerformanceSummary:
    """Compute standard risk/return metrics."""
    if strategy_returns.empty:
        raise ValueError("No strategy returns to analyze")

    n_days = len(strategy_returns)
    trading_days = 252
    rf_daily = (1 + risk_free_rate) ** (1 / trading_days) - 1
    excess = strategy_returns - rf_daily

    total_return = float((1 + strategy_returns).prod() - 1)
    ann_return = float((1 + total_return) ** (trading_days / n_days) - 1)
    ann_vol = float(strategy_returns.std() * np.sqrt(trading_days))
    sharpe = float(excess.mean() / excess.std() * np.sqrt(trading_days)) if excess.std() > 0 else 0.0

    cum = (1 + strategy_returns).cumprod()
    mdd = max_drawdown(cum)
    calmar = ann_return / abs(mdd) if mdd != 0 else 0.0
    win_rate = float((strategy_returns > 0).mean())

    monthly = strategy_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    avg_monthly = float(monthly.mean()) if not monthly.empty else 0.0

    info_ratio = None
    alpha = None
    beta = None

    if benchmark_returns is not None:
        aligned = pd.concat(
            [strategy_returns, benchmark_returns],
            axis=1,
            keys=["strategy", "benchmark"],
        ).dropna()
        if not aligned.empty:
            strat = aligned["strategy"]
            bench = aligned["benchmark"]
            active = strat - bench
            tracking_error = active.std() * np.sqrt(trading_days)
            info_ratio = float(active.mean() / active.std() * np.sqrt(trading_days)) if active.std() > 0 else 0.0

            cov = np.cov(strat, bench)
            beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else 0.0
            alpha = float((ann_return - risk_free_rate) - beta * (bench.mean() * trading_days - risk_free_rate))

    return PerformanceSummary(
        total_return=total_return,
        annualized_return=ann_return,
        annualized_volatility=ann_vol,
        sharpe_ratio=sharpe,
        max_drawdown=mdd,
        calmar_ratio=calmar,
        win_rate=win_rate,
        avg_monthly_return=avg_monthly,
        information_ratio=info_ratio,
        alpha=alpha,
        beta=beta,
    )


def performance_to_frame(summary: PerformanceSummary) -> pd.DataFrame:
    """Format summary as a single-row DataFrame for export."""
    data = asdict(summary)
    formatted = {}
    for k, v in data.items():
        if v is None:
            formatted[k] = "N/A"
        elif isinstance(v, float):
            formatted[k] = round(v, 4)
        else:
            formatted[k] = v
    return pd.DataFrame([formatted])


def factor_ic(
    normalized_factors: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    horizon: int = 21,
) -> pd.DataFrame:
    """
    Information Coefficient: cross-sectional correlation between factor
    and forward returns at each date. Used to validate factor efficacy.
    """
    fwd = forward_returns.shift(-horizon)
    records: list[dict] = []

    for fname, fpanel in normalized_factors.items():
        corrs: list[float] = []
        for date in fpanel.index:
            f = fpanel.loc[date]
            r = fwd.loc[date] if date in fwd.index else None
            if r is None:
                continue
            joined = pd.concat([f, r], keys=["factor", "ret"], axis=1).dropna()
            if len(joined) < 10:
                continue
            corrs.append(joined["factor"].corr(joined["ret"]))

        records.append({
            "factor": fname,
            "mean_ic": np.mean(corrs) if corrs else np.nan,
            "ic_ir": np.mean(corrs) / np.std(corrs) if corrs and np.std(corrs) > 0 else np.nan,
            "pct_positive": np.mean([c > 0 for c in corrs]) if corrs else np.nan,
        })

    return pd.DataFrame(records)
