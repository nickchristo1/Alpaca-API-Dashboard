# Nicholas Christophides  Nick.christophides@gmail.com

import os
import math
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from alpaca.trading.client import TradingClient
from dotenv import load_dotenv
from alpaca.trading.requests import GetPortfolioHistoryRequest
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
from io import StringIO
from backtest import run_backtest_calculation
from config import tickers


# 1.) Set-up and Initialization
# -----------------------------

load_dotenv()  # Load keys
app = FastAPI()
backtest_cache = {"data": None, "ready": False}  # Cache storage


@app.on_event("startup")
async def startup_event():
    import threading
    def background_task():
        data = run_backtest_calculation()
        backtest_cache["data"] = data
        backtest_cache["ready"] = True
        
    thread = threading.Thread(target=background_task)
    thread.start()


# --- Alpaca Client ---
api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")
cash_flows_csv = os.getenv("CASH_FLOWS").replace("|", "\n")
trading_client = TradingClient(api_key, secret_key, paper=False)

if os.path.isdir("static"):  # Static frontend
    app.mount("/static", StaticFiles(directory="static"), name="static")


def safe_float(x):
    if x is None:
        return 0.0
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return 0.0
    return float(x)


@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")


# 2.) Dashboard Backend
# ---------------------

@app.get("/api/portfolio")
async def get_portfolio():
    # ------------------------- Query Account information from Alpaca API -------------------------------
    account = trading_client.get_account()
    positions = trading_client.get_all_positions()

    history_request = GetPortfolioHistoryRequest(
        start="2026-04-29",
        end=date.today(),
        timeframe="1D",
        cashflow_types="ALL"
    )

    history = trading_client.get_portfolio_history(history_request)
    # ----------------------------------------------------------------------------------------------------

    # Get Deposit Data
    cash_flows = pd.read_csv(StringIO(cash_flows_csv))
    cash_flows["date"] = pd.to_datetime(cash_flows["date"])

    # Get Portfolio History Daily Data
    portfolio = pd.DataFrame({
        "date": pd.to_datetime(history.timestamp, unit="s", utc=True)
        .tz_convert("America/New_York")
        .normalize()
        .tz_localize(None),
        "equity": history.equity,
    })

    first_valid = portfolio["equity"].ne(0).idxmax()
    portfolio = portfolio.loc[first_valid:].reset_index(drop=True)

    cash_flows["date"] = pd.to_datetime(cash_flows["date"])
    portfolio["date"] = pd.to_datetime(portfolio["date"])

    # Find the next trading day for each cash flow
    trading_days = portfolio["date"].sort_values().reset_index(drop=True)
    idx = trading_days.searchsorted(cash_flows["date"])

    # Drop any deposits after the last portfolio date
    cash_flows = cash_flows[idx < len(trading_days)].copy()
    idx = idx[idx < len(trading_days)]
    cash_flows["date"] = trading_days.iloc[idx].values

    # Combine multiple deposits that land on the same trading day
    cash_flows = (cash_flows.groupby("date", as_index=False).agg({"deposit": "sum", "withdrawal": "sum"}))

    # Merge into portfolio
    portfolio = portfolio.merge(cash_flows, on="date", how="left")

    portfolio["deposit"] = portfolio["deposit"].fillna(0)
    portfolio["withdrawal"] = portfolio["withdrawal"].fillna(0)
    portfolio["net_cashflow"] = portfolio["deposit"] - portfolio["withdrawal"]
    portfolio["begin_equity"] = portfolio["equity"].shift(1)
    portfolio["r_t"] = (
            (portfolio["equity"] - portfolio["begin_equity"] - portfolio["net_cashflow"])
            / (portfolio["begin_equity"] + portfolio["net_cashflow"])
    )

    spy = yf.download("SPY", period="1y", interval="1d")["Close"].dropna()

    if spy is None or len(spy) < 2:
        spy_prices = np.array([1.0, 1.0])
        spy_returns = np.array([0.0])
    else:
        spy_prices = spy.to_numpy().flatten()
        spy_returns = np.diff(spy_prices) / np.where(spy_prices[:-1] == 0, 1e-8, spy_prices[:-1])

    # PnL Metrics
    equity = float(account.equity)  # Total assets summed
    last_equity = float(account.last_equity)  # Yesterday's final summed assets

    pnl_daily = sum([float(p.unrealized_intraday_pl) for p in positions])
    daily_return = pnl_daily / last_equity
    daily_return_pct = daily_return * 100 if last_equity != 0 else 0  # PnL in %

    starting_equity = portfolio["equity"].iloc[0]
    current_equity = float(account.equity)

    total_cashflows = portfolio["net_cashflow"].iloc[1:].sum()
    cum_return = current_equity - starting_equity - total_cashflows  # Dollar Return amount cumulative

    twr = float((1 + portfolio["r_t"].dropna()).prod() - 1)  # Time weighted return (updated to yesterday)
    live_twr = (1 + twr) * (1 + daily_return) - 1  # Time weighted return
    live_twr = safe_float(live_twr)
    # ----------------------------------------------------------------------------------------------------

    # -------------------------------------- Get Position Level Data -------------------------------------
    positions_data = sorted(
        [
            {
                # Stock Info
                "symbol": p.symbol,
                "qty": float(p.qty),
                "market_value": float(p.market_value),

                # Lifetime PnL
                "unrealized_pl": float(p.unrealized_pl),
                "return_pct": float(p.unrealized_plpc) * 100,

                # Today PnL
                "today_pl": float(p.unrealized_intraday_pl)
                if hasattr(p, "unrealized_intraday_pl")
                else float(p.unrealized_pl),
                "today_return_pct": float(p.unrealized_intraday_plpc) * 100
                if hasattr(p, "unrealized_intraday_plpc")
                else float(p.unrealized_plpc) * 100,

                # Portfolio Weight
                "weight": float(p.market_value) / equity
            }
            for p in positions
        ],
        key=lambda x: x["unrealized_pl"],
        reverse=True
    )
    # ----------------------------------------------------------------------------------------------------

    # ---------------------------------- Build the Portfolio Value Chart ---------------------------------
    chart_data = [
        {
            "timestamp": history.timestamp[len(history.equity)-22+i],
            "equity": float(history.equity[len(history.equity)-22+i])
        }
        for i in range(22)
    ]
    # ----------------------------------------------------------------------------------------------------

    # ------------------ Advanced Analytics: 1.) Find historical portfolio performance -------------------
    positions_dict = {  # Get a list of active positions
        p.symbol: p
        for p in positions
    }

    data = yf.download(tickers, period="1y", auto_adjust=True)["Close"]  # Extracted prices from yfinance
    data.index = pd.to_datetime(data.index)

    returns = data.pct_change(fill_method=None).dropna()  # Calculate returns of the active positions
    weights = pd.Series(0.0, index=data.columns)

    yf_map = {"BRK.B": "BRK-B"}  # Fix name mismatch between Alpaca and yfinance

    for t in tickers:
        yf_t = yf_map.get(t, t)

        if yf_t in positions_dict:
            weights[yf_t] = float(positions_dict[t].market_value) / equity

    port_daily_rets = returns @ weights  # Daily returns for portfolio

    if port_daily_rets.empty:
        strategy_return = 0.0
        portfolio_growth = pd.Series([1.0])
    else:
        portfolio_growth = (1 + port_daily_rets).cumprod()  # Find compounded return
        strategy_return = portfolio_growth.iloc[-1] - 1  # Portfolio return over the year

    # --------------------------- Advanced Analytics: 2.) Calculate metrics ----------------------------
    # Drawdown (Strategy)
    running_max = portfolio_growth.cummax()
    drawdown = (portfolio_growth - running_max) / running_max
    max_drawdown = np.min(drawdown)

    # Drawdown (SPY)
    spy_growth = np.cumprod(1+spy_returns)  # SPY compounded return
    spy_running_max = np.maximum.accumulate(spy_growth)
    spy_drawdown = (spy_growth - spy_running_max) / spy_running_max
    spy_max_drawdown = np.min(spy_drawdown)

    # Volatility
    strategy_vol = port_daily_rets.std() * np.sqrt(252)
    spy_vol = np.std(spy_returns) * np.sqrt(252) if len(spy_returns) > 1 else 0.0

    # Return
    spy_return = (spy_prices[-1] / spy_prices[0]) - 1 if len(spy_prices) > 1 else 0.0

    # Sharpe Ratio (Strategy)
    mean_ret = np.mean(port_daily_rets)
    std_ret = np.std(port_daily_rets)
    std_ret = max(std_ret, 1e-6)
    sharpe = (mean_ret / (std_ret + 1e-8)) * np.sqrt(252)

    # Sharpe Ratio (SPY)
    spy_mean_ret = np.mean(spy_returns)
    spy_std_ret = np.maximum(np.std(spy_returns), 1e-6)
    spy_sharpe = (spy_mean_ret / (spy_std_ret + 1e-8)) * np.sqrt(252)

    # VaR
    var_99 = np.percentile(port_daily_rets, 1) if len(port_daily_rets) > 1 else 0.0
    spy_var_99 = np.percentile(spy_returns, 1) if len(spy_returns) > 1 else 0.0
    # ----------------------------------------------------------------------------------------------------

    # --------------------------------- Return Values for the Dashboard ----------------------------------
    analytics = {
        "max_drawdown": safe_float(max_drawdown),
        "var_99": safe_float(var_99),
        "strategy_return": safe_float(strategy_return),
        "spy_return": safe_float(spy_return),
        "strategy_vol": safe_float(strategy_vol),
        "spy_vol": safe_float(spy_vol),
        "sharpe": safe_float(sharpe),
        "spy_sharpe": safe_float(spy_sharpe),
        "spy_var_99": safe_float(spy_var_99),
        "spy_max_drawdown": safe_float(spy_max_drawdown)
    }

    return {
        "equity": equity,
        "pnl_daily": pnl_daily,
        "daily_return_pct": daily_return_pct,
        "cum_return": cum_return,
        "live_twr": live_twr,
        "positions": positions_data,
        "history": chart_data,
        "analytics": analytics
    }


# --------------------------------- Return Values from the Backtest ----------------------------------
@app.get("/api/backtest")
async def get_backtest():
    if not backtest_cache["ready"]:
        return {"status": "processing"}
    return backtest_cache["data"]

