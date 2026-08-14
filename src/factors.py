"""Compute value, momentum, quality, and low-volatility factor scores."""

from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Cap extreme values at given percentiles."""
    if series.dropna().empty:
        return series
    lo, hi = series.quantile([lower, upper])
    return series.clip(lower=lo, upper=hi)


def cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score each row (date) across stocks; higher = better after orientation."""
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0)


def compute_momentum(
    prices: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
) -> pd.DataFrame:
    """
    12-1 month momentum: return from t-lookback to t-skip.

    Skipping the most recent month avoids short-term reversal effects
    documented in Jegadeesh & Titman (1993) and subsequent literature.
    """
    lagged_start = prices.shift(lookback)
    lagged_end = prices.shift(skip)
    momentum = (lagged_end / lagged_start) - 1.0
    return momentum


def compute_low_vol(
    prices: pd.DataFrame,
    lookback: int = 60,
) -> pd.DataFrame:
    """
    Low-volatility factor: negative of annualized realized vol.

    We orient so higher score = lower volatility (better for this factor).
    """
    daily_returns = prices.pct_change()
    rolling_vol = daily_returns.rolling(lookback).std() * np.sqrt(252)
    return -rolling_vol


def compute_value_from_fundamentals(
    fundamentals: pd.DataFrame,
    prices: pd.DataFrame,
    metric: str = "earnings_yield",
) -> pd.DataFrame:
    """
    Build a dynamic value factor panel.

    Uses trailing EPS (from fundamentals snapshot) divided by daily price so
    earnings yield updates as prices move. This avoids a static value signal
    that never re-ranks stocks.
    """
    if metric == "earnings_yield":
        trailing_pe = fundamentals["trailing_pe"].replace(0, np.nan)
        # Approximate trailing EPS from snapshot P/E and recent price level
        ref_price = prices.ffill().iloc[-1]
        trailing_eps = ref_price / trailing_pe
        value_panel = trailing_eps / prices
    elif metric == "book_to_market":
        ptb = fundamentals["price_to_book"].replace(0, np.nan)
        ref_price = prices.ffill().iloc[-1]
        book_value = ref_price / ptb
        value_panel = book_value / prices
    else:
        raise ValueError(f"Unknown value metric: {metric}")

    return value_panel.replace([np.inf, -np.inf], np.nan).reindex(columns=prices.columns)


def compute_quality_from_fundamentals(
    fundamentals: pd.DataFrame,
    prices: pd.DataFrame,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """
    Quality composite from profitability metrics plus price stability.

    Static ROE/margins are blended with rolling return stability so the
    quality signal varies over time rather than staying fixed.
    """
    metrics = metrics or ["roe", "profit_margin"]
    static_parts: list[pd.Series] = []

    if "roe" in metrics:
        static_parts.append(fundamentals["return_on_equity"])
    if "profit_margin" in metrics:
        static_parts.append(fundamentals["profit_margins"])

    if not static_parts:
        raise ValueError("No quality metrics specified")

    static_quality = pd.concat(static_parts, axis=1).mean(axis=1)

    # Price stability: lower return vol implies higher quality
    return_vol = prices.pct_change().rolling(126).std()
    stability = -return_vol  # higher = more stable = better quality

    static_panel = pd.DataFrame(
        np.tile(static_quality.values, (len(prices.index), 1)),
        index=prices.index,
        columns=prices.columns,
    ).reindex(columns=prices.columns)

    # 60% fundamental quality, 40% price stability
    quality_panel = 0.6 * static_panel + 0.4 * stability
    return quality_panel


def apply_fundamental_lag(factor_panel: pd.DataFrame, lag_days: int) -> pd.DataFrame:
    """Shift fundamentals forward in time to approximate reporting delay."""
    return factor_panel.shift(lag_days)


def build_factor_panels(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    config: dict,
) -> dict[str, pd.DataFrame]:
    """Compute raw factor panels before cross-sectional normalization."""
    fcfg = config["factors"]

    momentum = compute_momentum(
        prices,
        lookback=fcfg["momentum_lookback"],
        skip=fcfg["momentum_skip"],
    )
    low_vol = compute_low_vol(prices, lookback=fcfg["vol_lookback"])

    value_raw = compute_value_from_fundamentals(
        fundamentals.reindex(prices.columns),
        prices,
        metric=fcfg.get("value_metric", "earnings_yield"),
    )
    quality_raw = compute_quality_from_fundamentals(
        fundamentals.reindex(prices.columns),
        prices,
        metrics=fcfg.get("quality_metrics"),
    )

    lag = fcfg.get("fundamental_lag_days", 63)
    value = apply_fundamental_lag(value_raw, lag)
    quality = apply_fundamental_lag(quality_raw, lag)

    return {
        "value": value,
        "momentum": momentum,
        "quality": quality,
        "low_vol": low_vol,
    }


def normalize_factors(factor_panels: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Winsorize and cross-sectionally z-score each factor."""
    normalized: dict[str, pd.DataFrame] = {}
    for name, panel in factor_panels.items():
        winsorized = panel.apply(winsorize, axis=1)
        normalized[name] = cross_sectional_zscore(winsorized)
    return normalized


def composite_score(
    normalized_factors: dict[str, pd.DataFrame],
    weights: dict[str, float],
) -> pd.DataFrame:
    """
    Weighted sum of z-scored factors.

    Default equal-weight (25% each) is a common baseline in multi-factor
    literature (Fama-French + momentum + quality extensions).
    """
    score = None
    for factor, weight in weights.items():
        if factor not in normalized_factors:
            continue
        contribution = normalized_factors[factor] * weight
        score = contribution if score is None else score.add(contribution, fill_value=0)
    if score is None:
        raise ValueError("No factors available for composite score")
    return score
