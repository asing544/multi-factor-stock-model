#!/usr/bin/env python3
"""
Multi-Factor Stock Ranking & Long-Short Backtest

Run: python main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.analytics import compute_performance, factor_ic, performance_to_frame
from src.backtest import run_backtest
from src.data_loader import download_benchmark, download_fundamentals, download_prices, load_config
from src.factors import build_factor_panels, composite_score, normalize_factors
from src.visualize import (
    plot_drawdown,
    plot_equity_curve,
    plot_factor_exposure,
    plot_factor_ic,
    plot_monthly_returns_heatmap,
)


def main() -> None:
    config_path = ROOT / "config.yaml"
    data_dir = ROOT / "data"
    output_dir = ROOT / "output"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Multi-Factor Long-Short Backtest")
    print("=" * 60)

    config = load_config(config_path)
    tickers = list(dict.fromkeys(config["universe"]["tickers"]))  # dedupe, preserve order
    bcfg = config["backtest"]

    print(f"\n[1/6] Downloading price data for {len(tickers)} stocks...")
    prices = download_prices(
        tickers,
        start="2016-01-01",  # extra history for factor lookbacks
        end=bcfg["end_date"],
        cache_dir=data_dir,
    )
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.8))
    print(f"      {prices.shape[1]} stocks with sufficient history")

    print("\n[2/6] Fetching fundamental data...")
    fundamentals = download_fundamentals(prices.columns.tolist(), cache_dir=data_dir)
    print(f"      Fundamentals available for {fundamentals.shape[0]} stocks")

    print("\n[3/6] Computing factor scores...")
    raw_factors = build_factor_panels(prices, fundamentals, config)
    normalized = normalize_factors(raw_factors)
    scores = composite_score(normalized, config["factors"]["weights"])
    print("      Factors: value, momentum, quality, low_vol (equal-weight composite)")

    print("\n[4/6] Running long-short backtest...")
    result = run_backtest(prices, scores, normalized, config)
    print(f"      {len(result.rebalance_log)} rebalance events")

    print("\n[5/6] Computing performance metrics...")
    bench_prices = download_benchmark("SPY", bcfg["start_date"], bcfg["end_date"])
    bench_returns = bench_prices.pct_change().dropna()
    bench_cum = (1 + bench_returns).cumprod()

    summary = compute_performance(
        result.daily_returns,
        bench_returns,
        risk_free_rate=bcfg["risk_free_rate"],
    )
    ic = factor_ic(normalized, prices.pct_change())

    print("\n[6/6] Generating charts and reports...")
    plot_equity_curve(result.cumulative_returns, bench_cum, output_dir / "equity_curve.png")
    plot_drawdown(result.daily_returns, output_dir / "drawdown.png")
    plot_factor_exposure(result.factor_exposure, output_dir / "factor_exposure.png")
    plot_monthly_returns_heatmap(result.daily_returns, output_dir / "monthly_returns.png")
    plot_factor_ic(ic, output_dir / "factor_ic.png")

    performance_to_frame(summary).to_csv(output_dir / "performance_summary.csv", index=False)
    ic.to_csv(output_dir / "factor_ic.csv", index=False)
    result.rebalance_log.to_csv(output_dir / "rebalance_log.csv", index=False)
    result.daily_returns.to_csv(output_dir / "daily_returns.csv")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Total Return:        {summary.total_return:>8.1%}")
    print(f"  Annualized Return:   {summary.annualized_return:>8.1%}")
    print(f"  Annualized Vol:      {summary.annualized_volatility:>8.1%}")
    print(f"  Sharpe Ratio:        {summary.sharpe_ratio:>8.2f}")
    print(f"  Max Drawdown:        {summary.max_drawdown:>8.1%}")
    print(f"  Calmar Ratio:        {summary.calmar_ratio:>8.2f}")
    print(f"  Win Rate (daily):    {summary.win_rate:>8.1%}")
    if summary.information_ratio is not None:
        print(f"  Information Ratio:   {summary.information_ratio:>8.2f}")
    if summary.alpha is not None:
        print(f"  Alpha vs SPY:        {summary.alpha:>8.1%}")
    if summary.beta is not None:
        print(f"  Beta vs SPY:         {summary.beta:>8.2f}")

    print("\n  Factor Information Coefficients:")
    for _, row in ic.iterrows():
        print(f"    {row['factor']:>10s}: IC={row['mean_ic']:+.3f}  (% positive: {row['pct_positive']:.0%})")

    print(f"\n  Outputs saved to: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
