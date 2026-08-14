"""Load and cache market data for the factor model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf


def load_config(config_path: str | Path) -> dict:
    import yaml

    with open(config_path) as f:
        return yaml.safe_load(f)


def _unique_tickers(tickers: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for ticker in tickers:
        if ticker not in seen:
            seen.add(ticker)
            ordered.append(ticker)
    return ordered


def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Download adjusted close prices for all tickers.

    Returns a DataFrame indexed by date with one column per ticker.
    """
    tickers = _unique_tickers(tickers)
    cache_dir = Path(cache_dir) if cache_dir else None
    cache_file = cache_dir / "prices.parquet" if cache_dir else None

    if cache_file and cache_file.exists():
        prices = pd.read_parquet(cache_file)
        prices = prices.loc[(prices.index >= start) & (prices.index <= end)]
        missing = [t for t in tickers if t not in prices.columns]
        if not missing:
            return prices[tickers]

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw.to_frame(name=tickers[0])

    prices = prices.dropna(how="all").sort_index()
    prices = prices.loc[:, ~prices.columns.duplicated()]

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(cache_file)

    return prices


def download_fundamentals(
    tickers: list[str],
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Fetch point-in-time fundamental snapshots via yfinance.

    Note: Free fundamental data is limited; we apply a reporting lag in the
    factor engine to mitigate look-ahead bias. For production, use a
    point-in-time fundamental database (Compustat, FactSet, etc.).
    """
    tickers = _unique_tickers(tickers)
    cache_dir = Path(cache_dir) if cache_dir else None
    cache_file = cache_dir / "fundamentals.json" if cache_dir else None

    if cache_file and cache_file.exists():
        with open(cache_file) as f:
            payload = json.load(f)
        return pd.DataFrame(payload).T

    rows: dict[str, dict] = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
        except Exception:
            continue

        rows[ticker] = {
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "price_to_book": info.get("priceToBook"),
            "return_on_equity": info.get("returnOnEquity"),
            "profit_margins": info.get("profitMargins"),
            "debt_to_equity": info.get("debtToEquity"),
            "market_cap": info.get("marketCap"),
        }

    fundamentals = pd.DataFrame(rows).T

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(fundamentals.to_dict(orient="index"), f)

    return fundamentals


def download_benchmark(
    symbol: str,
    start: str,
    end: str,
) -> pd.Series:
    """Download benchmark (e.g., SPY) adjusted close returns."""
    data = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].squeeze()
    else:
        close = data["Close"]
    return close.dropna().sort_index()
