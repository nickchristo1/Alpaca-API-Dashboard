# Nicholas Christophides  nick.christophides@gmail.com

""" In backtest.py the optimizer is back-tested over the last 6 months to determine how the portfolio it gives would
 perform over that time period.
 *NOTE: The optimizer will be given different data in the backtest than in the actual portfolio used for trading. As
 a result of this, this back-tester isn't meant to be used as proof of future performance of the portfolio that is
 given, but instead used as supporting evidence that the optimizer can produce portfolios that have approximately the
 desired level of return and volatility. """

import pandas as pd
from portfolio_optimization import eff_front_no_shorts, find_gmv_return
from estimate_cov_matrix import tickers, estimate_cov_matrix
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt


# 1.) Gather historical data for optimization
# -------------------------------------------
data = yf.download(tickers, period="4y", auto_adjust=True)["Close"].tail(760)  # Extracted prices from yfinance
data.index = pd.to_datetime(data.index)


# 2.) Use the rolling window approach to backtest the strategy over 60 days, with rebalancing occurring every week
# ----------------------------------------------------------------------------------------------------------------
optimal_weights = None  # Hold the array of position weights
previous_week = None  # Used to determine when rebalancing should occur
backtest_period = 60  # Backtest over 60 trading days
portfolio_daily_returns = {}  # Used in storing the portfolio returns
daily_returns = data.tail(backtest_period+1).pct_change(fill_method=None).dropna()  # Series of returns from the stocks
rebalance_dates = []  # Used in visualization

for current_date, day_returns in daily_returns.iterrows():
    current_week = current_date.isocalendar()[1]
    # If new week or beginning of backtest; rebalance
    if optimal_weights is None or current_week != previous_week:

        # Find necessary quantities
        window_data = data.loc[data.index < current_date].tail(700)  # Optimize on past 700 days
        mu_hat = window_data.mean().values
        log_returns = np.log(window_data / window_data.shift(1)).dropna()
        log_returns_std = (log_returns - log_returns.mean()) / log_returns.std()
        cov_mat, _ = estimate_cov_matrix(np.cov(np.array(log_returns_std), rowvar=False), log_returns)

        # In-Sample Optimization
        mu_gmv = find_gmv_return(mu_hat, cov_mat)
        mus = np.linspace(mu_gmv, np.max(mu_hat), 25)
        optimal_weights = np.array(eff_front_no_shorts(mus[12], mu_hat, cov_mat))

        # Update the weekly tracker
        rebalance_dates.append(current_date)
        previous_week = current_week

    portfolio_daily_returns[current_date] = .98 * day_returns.dot(optimal_weights)  # Find portfolio return for the day

returns_series = pd.Series(portfolio_daily_returns)
cumulative_growth = (1 + returns_series).cumprod()  # Portfolio Cumulative Growth


# 3.) Visualization of the results
# --------------------------------
spy = yf.download("SPY", period="3mo", auto_adjust=True)["Close"].tail(60)  # Compare strategy to SPY
spy_return_series = spy.pct_change(fill_method=None).dropna()
spy_cum_return = (1 + spy_return_series).cumprod()

plt.figure(figsize=(12, 6))
plt.plot(cumulative_growth, label="Strategy Growth", color="blue")
plt.plot(spy_cum_return, label="SPY Benchmark", color="red")
rebalance_values = cumulative_growth.loc[rebalance_dates]
plt.scatter(rebalance_dates, rebalance_values, color='red', zorder=5, label="Rebalance Triggered")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.title(f"Back-tested Portfolio over Last {backtest_period} Days (Out-of-Sample)\n Strategy Return = "
          f"{(cumulative_growth.iloc[-1] - 1)*100:.2f}%")
plt.grid(True)
plt.legend()
plt.style.use('seaborn-v0_8')
plt.show()


# 4.) Portfolio Metrics
# ---------------------
# Calculate drawdowns of the strategy
strat_rolling_max = cumulative_growth.cummax()
strat_drawdown = (cumulative_growth - strat_rolling_max) / strat_rolling_max
strat_max_drawdown = strat_drawdown.min()

# Calculate drawdowns of SPY buy-and-hold
spy_rolling_max = spy_cum_return.cummax()
spy_drawdown = (spy_cum_return - spy_rolling_max) / spy_rolling_max
spy_max_drawdown = spy_drawdown.values.min()

returns_list = returns_series.values
spy_returns_list = spy_return_series.values
sharpe = returns_list.mean() / returns_list.std()

table_data = [
    ["Cumulative Return", f"{cumulative_growth.iloc[-1]-1:.4f}", f"{spy_cum_return.values[-1][0]-1:.4f}"],
    ["Mean Return", f"{returns_list.mean():.6f}", f"{spy_returns_list.mean():.6f}"],
    ["Std. Dev.", f"{returns_list.std():.5f}", f"{spy_returns_list.std():.5f}"],
    ["Sharpe Ratio", f"{sharpe:.4f}", f"{spy_returns_list.mean()/spy_returns_list.std():.4f}"],
    ["Maximum Drawdown", f"{-strat_max_drawdown:.4f}", f"{-spy_max_drawdown:.4f}"]
]
