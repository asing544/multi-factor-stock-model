# Multi-Factor Stock Ranking & Long-Short Backtest

A quantitative finance project that ranks US equities on four canonical factors — **Value**, **Momentum**, **Quality**, and **Low Volatility** — and backtests a dollar-neutral long-short portfolio.

## Resume Bullet (copy-paste ready)

> Built a multi-factor equity ranking model (value, momentum, quality, low-volatility) in Python, combining cross-sectional z-scores into an equal-weight composite; backtested a monthly rebalanced long-short portfolio (top/bottom quintile) on 80+ large-cap US stocks (2018–2024), reporting Sharpe ratio, max drawdown, information ratio, and factor IC analysis vs. SPY benchmark.

## What This Project Does

### 1. Factor Construction

| Factor | Definition | Economic Intuition |
|--------|-----------|-------------------|
| **Value** | Earnings yield (1 / trailing P/E) | Cheap stocks tend to outperform over long horizons (Fama-French HML) |
| **Momentum** | 12-1 month return (skip last month) | Winners keep winning; avoids short-term reversal (Jegadeesh & Titman, 1993) |
| **Quality** | Average of ROE and profit margin | Profitable, efficient firms outperform (Novy-Marx, 2013) |
| **Low Volatility** | Negative of 60-day annualized vol | Low-vol anomaly: safer stocks often beat on risk-adjusted basis |

Each factor is **winsorized** (1st/99th percentile) and **cross-sectionally z-scored** at every date, then combined into a composite score with equal 25% weights.

### 2. Portfolio Construction

- **Universe:** ~80 large-cap US equities (S&P 500 subset)
- **Long leg:** Top 20% by composite score (equal-weight)
- **Short leg:** Bottom 20% by composite score (equal-weight)
- **Structure:** Dollar-neutral (50% gross long, 50% gross short)
- **Rebalance:** Monthly
- **Costs:** 10 bps per trade (one-way), applied on turnover

### 3. Backtest Engine

Event-driven simulation that:
1. Ranks stocks at each rebalance date
2. Constructs target weights
3. Computes daily portfolio returns
4. Tracks turnover, factor exposures, and rebalance logs

### 4. Analytics

- Sharpe ratio, max drawdown, Calmar ratio, win rate
- Information ratio and alpha/beta vs. SPY
- Factor Information Coefficient (IC) — validates each factor's predictive power
- Equity curve, drawdown chart, monthly returns heatmap, factor exposure over time

## Project Structure

```
finance project/
├── main.py                 # Entry point — run the full pipeline
├── config.yaml             # Universe, factor weights, backtest params
├── requirements.txt        # Python dependencies
├── src/
│   ├── data_loader.py      # Download & cache prices + fundamentals (yfinance)
│   ├── factors.py          # Compute & normalize all 4 factors
│   portfolio.py            # Long-short selection & weight construction
│   backtest.py             # Simulation engine
│   analytics.py            # Performance metrics & IC analysis
│   visualize.py            # Charts
├── data/                   # Cached market data (auto-created)
└── output/                 # Results: charts, CSVs (auto-created)
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (downloads data, computes factors, backtests, generates charts)
python main.py
```

Results appear in `output/`:
- `equity_curve.png` — Strategy vs. SPY
- `drawdown.png` — Underwater chart
- `factor_exposure.png` — Net long-short exposure per factor
- `monthly_returns.png` — Returns heatmap by year/month
- `factor_ic.png` — Factor predictive power
- `performance_summary.csv` — All metrics
- `factor_ic.csv` — IC by factor
- `rebalance_log.csv` — Portfolio changes each month

## Methodology Notes (for interviews)

**Why z-scores?** Cross-sectional z-scoring makes factors comparable (P/E of 15 vs. momentum of 0.3 are on the same scale) and is standard in institutional factor models.

**Why 12-1 momentum?** Skipping the most recent month avoids the short-term reversal effect that contaminates raw 12-month returns.

**Why a 63-day fundamental lag?** Free fundamental data from yfinance is a current snapshot, not point-in-time. The lag approximates the delay before quarterly reports are publicly available, reducing look-ahead bias.

**Limitations to mention honestly:**
- Survivorship bias (current S&P 500 constituents used throughout)
- Fundamental data is not truly point-in-time (production systems use Compustat/FactSet)
- No slippage model beyond flat transaction costs
- Shorting assumes full borrow availability

## Tech Stack

- **Python 3.10+**
- **pandas / numpy** — vectorized factor computation
- **yfinance** — market data
- **matplotlib / seaborn** — visualization
- **scipy** — statistical metrics

## Customization

Edit `config.yaml` to change:
- Stock universe (`universe.tickers`)
- Backtest period (`backtest.start_date`, `end_date`)
- Factor weights (`factors.weights`)
- Long/short percentiles (`backtest.long_pct`, `short_pct`)
- Transaction costs (`backtest.transaction_cost_bps`)

## License

MIT — free to use on your resume and in interviews.
