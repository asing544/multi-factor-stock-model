"""Visualization helpers for backtest results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _style():
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    })


def plot_equity_curve(
    strategy_cum: pd.Series,
    benchmark_cum: pd.Series | None,
    output_path: str | Path,
) -> None:
    _style()
    fig, ax = plt.subplots()

    strategy_cum.plot(ax=ax, label="Multi-Factor L/S", linewidth=2)
    if benchmark_cum is not None:
        benchmark_cum.plot(ax=ax, label="SPY (Buy & Hold)", linewidth=2, alpha=0.8)

    ax.set_title("Cumulative Returns: Multi-Factor Long-Short vs Benchmark")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_drawdown(
    strategy_returns: pd.Series,
    output_path: str | Path,
) -> None:
    _style()
    cum = (1 + strategy_returns).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()

    fig, ax = plt.subplots()
    dd.plot(ax=ax, color="crimson", linewidth=1.5)
    ax.fill_between(dd.index, dd.values, 0, alpha=0.3, color="crimson")
    ax.set_title("Strategy Drawdown")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_factor_exposure(
    exposure: pd.DataFrame,
    output_path: str | Path,
) -> None:
    if exposure.empty:
        return

    _style()
    net_cols = [c for c in exposure.columns if c.startswith("net_")]
    if not net_cols:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    exposure[net_cols].plot(ax=ax, linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Net Factor Exposure (Long − Short) Over Time")
    ax.set_ylabel("Z-Score Exposure")
    ax.set_xlabel("Rebalance Date")
    ax.legend([c.replace("net_", "").title() for c in net_cols])
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_monthly_returns_heatmap(
    strategy_returns: pd.Series,
    output_path: str | Path,
) -> None:
    _style()
    monthly = strategy_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    df = pd.DataFrame({
        "return": monthly.values,
        "year": monthly.index.year,
        "month": monthly.index.month,
    })
    pivot = df.pivot(index="year", columns="month", values="return")

    fig, ax = plt.subplots(figsize=(12, max(4, len(pivot) * 0.5)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1%",
        cmap="RdYlGn",
        center=0,
        ax=ax,
        cbar_kws={"label": "Monthly Return"},
    )
    ax.set_title("Monthly Returns Heatmap")
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_factor_ic(
    ic_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    if ic_df.empty:
        return

    _style()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2ecc71" if x > 0 else "#e74c3c" for x in ic_df["mean_ic"]]
    ax.barh(ic_df["factor"], ic_df["mean_ic"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Mean Information Coefficient by Factor")
    ax.set_xlabel("Mean IC (correlation with 1-month forward returns)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
