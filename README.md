# Risk Before Return: Volatility Targeting Strategies Analysis

This project replicates the results from the paper "Risk Before Return: Targeting Volatility with Higher Frequency Data" by implementing various volatility targeting strategies and comparing their performance against a buy-and-hold benchmark and a trend-following strategy.

## Overview

The analysis implements five different strategies:

1. **Buy-and-Hold (Benchmark)**: 100% allocation to SPY at all times
2. **30D HV Volatility Targeting**: Uses 30-day historical volatility to target 10% annualized volatility
3. **VIX Volatility Targeting**: Uses VIX index to target 10% annualized volatility
4. **2D RV Volatility Targeting**: Uses 2-day realized volatility (from intraday data) to target 10% annualized volatility
5. **200D MA Trend-Following**: Uses 200-day moving average to determine stock vs. cash allocation

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Setup

1. **Get API Keys**:
   - **Massive API Key**: Sign up at [massive.app](https://massive.app) to get your API key
   - **FMP API Key**: Sign up at [financialmodelingprep.com](https://financialmodelingprep.com) to get your API key

2. **Configure Environment Variables**:
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` and add your API keys:
     ```
     MASSIVE_API_KEY=your_massive_api_key_here
     FMP_API_KEY=your_fmp_api_key_here
     ```

## Usage

Run the analysis:

```bash
python volatility_targeting_analysis.py
```

**Note**: At least one API key (MASSIVE_API_KEY or FMP_API_KEY) is required. You can use both or just one depending on which data sources you want to use.

The script will:
- Download all required data (SPY daily, VIX daily, risk-free rate, SPY intraday 15-minute)
- Implement all strategies
- Calculate performance metrics
- Generate visualizations
- Create a summary report

## Output Files

After running the analysis, you'll get:

### Plots (in `plots/` directory)

- **equity_curves.png**: Portfolio value over time for all strategies
- **rolling_volatility.png**: Rolling 1-year volatility comparison between buy-and-hold and 30D HV
- **drawdowns.png**: Drawdown analysis for 200D MA, 2D RV, and buy-and-hold
- **allocations.png**: Stock allocation over time for volatility targeting strategies
- **spy_price.png**: SPY price chart over the analysis period
- **vix_price.png**: VIX index chart over the analysis period
- **risk_free_rate.png**: Risk-free rate (3-month Treasury bill) chart over the analysis period

### Reports (in `reports/` directory)

- **report_[TICKER]_[START_DATE]_[END_DATE]_vol[VOL]pct_tc[TC]bp.txt**: Comprehensive text report with performance metrics and findings. The filename includes the ticker symbol, date range, target volatility percentage, and transaction cost basis points used in the analysis.

## Performance Metrics

Each strategy is evaluated using:

- **CAGR**: Compound Annual Growth Rate
- **Annualized Excess Return**: Average return above risk-free rate (annualized)
- **Annualized Volatility**: Standard deviation of returns (annualized)
- **Sharpe Ratio**: Risk-adjusted return metric
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of days with positive returns

## Key Parameters

- **Target Volatility**: 10% annualized
- **Trading Days per Year**: 252
- **Realized Volatility Scale Factor**: 1.3 (accounts for overnight returns)
- **Historical Volatility Window**: 30 days
- **Realized Volatility Window**: 2 days
- **Moving Average Window**: 200 days
- **Analysis Period**: December 2003 to March 2020

## Strategy Details

### Volatility Targeting Formula

All volatility targeting strategies use the same allocation formula:

```
Allocation = min(1, Target_Volatility / Forecasted_Volatility)
```

Where:
- Target Volatility = 10% (0.10)
- Forecasted Volatility = depends on the strategy (30D HV, VIX, or 2D RV)

### Realized Volatility Calculation

For the 2D RV strategy, daily realized volatility is calculated as:

```
RVariance_t = Σ(r_15min,i²)  for all 15-minute intervals in day t
σ_t^RV = √(252 × RVariance_t) × 1.3
```

Then the 2-day average is used:
```
σ_t^2D_RV = (σ_{t-2}^RV + σ_{t-1}^RV) / 2
```

## Expected Results

Based on the paper, you should expect:

- **2D RV** to achieve the highest Sharpe ratio among volatility targeting strategies
- Volatility targeting strategies to have lower volatility than buy-and-hold
- Volatility targeting to reduce maximum drawdowns compared to buy-and-hold

## Notes

- All strategies use a one-day lag in rebalancing (decisions made at close of day t apply to day t+1)
- Risk-free rate is subtracted from portfolio returns to compute excess returns
- All returns and volatilities are expressed in annualized terms
- Missing data is handled using forward-fill method

## Troubleshooting

If you encounter issues:

1. **API Key Issues**: Make sure your Massive API key is valid and has access to the required data
2. **Data Availability**: Some historical data may not be available for the full period. The script will skip strategies if required data is missing.
3. **Memory Issues**: For large intraday datasets, the script processes data day-by-day to manage memory

## References

Paper: "Risk Before Return: Targeting Volatility with Higher Frequency Data"

