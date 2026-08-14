"""Event-driven backtest engine for the long-short factor strategy."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .portfolio import build_target_weights, rebalance_dates, select_long_short


@dataclass
class BacktestResult:
    """Container for backtest outputs and diagnostics."""

    daily_returns: pd.Series
    cumulative_returns: pd.Series
    weights_history: pd.DataFrame
    turnover: pd.Series
    rebalance_log: pd.DataFrame
    factor_exposure: pd.DataFrame = field(default_factory=pd.DataFrame)


def _daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change()


def _portfolio_return(
    weights: pd.Series,
    asset_returns: pd.Series,
) -> float:
    aligned = weights.reindex(asset_returns.index).fillna(0.0)
    return float((aligned * asset_returns.fillna(0.0)).sum())


def _turnover(old_w: pd.Series, new_w: pd.Series) -> float:
    """One-way turnover as sum of absolute weight changes / 2."""
    all_idx = old_w.index.union(new_w.index)
    old_aligned = old_w.reindex(all_idx).fillna(0.0)
    new_aligned = new_w.reindex(all_idx).fillna(0.0)
    return float((new_aligned - old_aligned).abs().sum() / 2.0)


def run_backtest(
    prices: pd.DataFrame,
    composite_scores: pd.DataFrame,
    normalized_factors: dict[str, pd.DataFrame],
    config: dict,
) -> BacktestResult:
    """
    Simulate a monthly rebalanced dollar-neutral long-short strategy.

    Mechanics:
    1. At each rebalance date, rank stocks by composite factor score.
    2. Go long top quintile, short bottom quintile (equal-weight each leg).
    3. Hold until next rebalance; apply transaction costs on turnover.
    """
    bcfg = config["backtest"]
    rebal_dates = rebalance_dates(
        composite_scores,
        freq=bcfg["rebalance_freq"],
        start=bcfg["start_date"],
        end=bcfg["end_date"],
    )

    returns = _daily_returns(prices)
    tc = bcfg["transaction_cost_bps"] / 10_000.0

    current_weights = pd.Series(dtype=float)
    daily_rets: list[tuple[pd.Timestamp, float]] = []
    weight_rows: list[pd.Series] = []
    turnover_rows: list[tuple[pd.Timestamp, float]] = []
    rebalance_rows: list[dict] = []
    exposure_rows: list[dict] = []

    rebal_set = set(rebal_dates)
    trading_days = returns.index[
        (returns.index >= pd.Timestamp(bcfg["start_date"]))
        & (returns.index <= pd.Timestamp(bcfg["end_date"]))
    ]

    for i, date in enumerate(trading_days):
        if date in rebal_set or i == 0:
            score_row = composite_scores.loc[:date].iloc[-1]
            longs, shorts = select_long_short(
                score_row,
                long_pct=bcfg["long_pct"],
                short_pct=bcfg["short_pct"],
            )
            new_weights = build_target_weights(longs, shorts)

            if not current_weights.empty and not new_weights.empty:
                turn = _turnover(current_weights, new_weights)
                turnover_rows.append((date, turn))
                cost = turn * tc * 2  # buy + sell
            else:
                turn = 0.0 if current_weights.empty else 1.0
                turnover_rows.append((date, turn))
                cost = turn * tc * 2 if turn > 0 else 0.0

            current_weights = new_weights
            weight_rows.append(current_weights.rename(date))

            rebalance_rows.append({
                "date": date,
                "n_long": len(longs),
                "n_short": len(shorts),
                "long_names": ",".join(longs[:5]) + ("..." if len(longs) > 5 else ""),
                "short_names": ",".join(shorts[:5]) + ("..." if len(shorts) > 5 else ""),
            })

            exp = {"date": date}
            for fname, fpanel in normalized_factors.items():
                row = fpanel.loc[:date].iloc[-1]
                exp[f"long_{fname}"] = row.reindex(longs).mean()
                exp[f"short_{fname}"] = row.reindex(shorts).mean()
                exp[f"net_{fname}"] = exp[f"long_{fname}"] - exp[f"short_{fname}"]
            exposure_rows.append(exp)

            if i == 0:
                continue

        if current_weights.empty:
            daily_rets.append((date, 0.0))
            continue

        day_ret = _portfolio_return(current_weights, returns.loc[date])
        if date in rebal_set and i > 0:
            day_ret -= cost
        daily_rets.append((date, day_ret))

    ret_series = pd.Series(
        dict(daily_rets),
        name="strategy_return",
    ).sort_index()

    cum = (1 + ret_series).cumprod()
    weights_df = pd.DataFrame(weight_rows).T if weight_rows else pd.DataFrame()
    turnover_series = pd.Series(dict(turnover_rows), name="turnover") if turnover_rows else pd.Series(dtype=float)
    rebalance_df = pd.DataFrame(rebalance_rows)
    exposure_df = pd.DataFrame(exposure_rows).set_index("date") if exposure_rows else pd.DataFrame()

    return BacktestResult(
        daily_returns=ret_series,
        cumulative_returns=cum,
        weights_history=weights_df,
        turnover=turnover_series,
        rebalance_log=rebalance_df,
        factor_exposure=exposure_df,
    )
