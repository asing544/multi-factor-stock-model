"""Long-short portfolio construction from composite factor ranks."""

from __future__ import annotations

import pandas as pd


def rebalance_dates(
    scores: pd.DataFrame,
    freq: str = "M",
    start: str | None = None,
    end: str | None = None,
) -> pd.DatetimeIndex:
    """Return month-end (or other freq) dates where we have enough score coverage."""
    idx = scores.index
    if start:
        idx = idx[idx >= pd.Timestamp(start)]
    if end:
        idx = idx[idx <= pd.Timestamp(end)]

    grouped = scores.loc[idx].groupby(pd.Grouper(freq=freq)).last()
    valid = grouped.dropna(how="all").index
    return pd.DatetimeIndex(valid)


def rank_stocks(scores: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank; 1 = best."""
    return scores.rank(ascending=True, pct=True)


def select_long_short(
    scores: pd.Series,
    long_pct: float = 0.20,
    short_pct: float = 0.20,
    min_names: int = 5,
) -> tuple[list[str], list[str]]:
    """
    Select long (top) and short (bottom) baskets by composite score.

    Returns ticker lists for the long and short legs.
    """
    valid = scores.dropna()
    if len(valid) < 2 * min_names:
        return [], []

    n_long = max(min_names, int(len(valid) * long_pct))
    n_short = max(min_names, int(len(valid) * short_pct))

    ranked = valid.sort_values(ascending=False)
    longs = ranked.head(n_long).index.tolist()
    shorts = ranked.tail(n_short).index.tolist()
    return longs, shorts


def equal_weight_weights(tickers: list[str], sign: float = 1.0) -> pd.Series:
    """Equal-weight portfolio weights with optional sign for short leg."""
    if not tickers:
        return pd.Series(dtype=float)
    w = sign / len(tickers)
    return pd.Series(w, index=tickers)


def build_target_weights(
    longs: list[str],
    shorts: list[str],
    gross_exposure: float = 1.0,
) -> pd.Series:
    """
    Dollar-neutral long-short: 50% gross long, 50% gross short by default.

    gross_exposure=1.0 -> 0.5 long + 0.5 short (market neutral).
    """
    if not longs or not shorts:
        return pd.Series(dtype=float)

    long_weight = (gross_exposure / 2.0) / len(longs)
    short_weight = -(gross_exposure / 2.0) / len(shorts)

    weights = pd.concat([
        pd.Series(long_weight, index=longs),
        pd.Series(short_weight, index=shorts),
    ])
    return weights
