# Nicholas Christophides  Nick.christophides@gmail.com

import time
import os
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr
from dotenv import load_dotenv
from sklearn.linear_model import Ridge
from config import tickers
from datetime import date


# 0.) Functions needed for replication
# ------------------------------------
load_dotenv()  # Load keys
api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")
DATA_FILE = "market_data.csv"


def get_data():
    SECONDS_IN_DAY = 86400 
    
    # Load Data
    if os.path.exists(DATA_FILE) and (time.time() - os.path.getmtime(DATA_FILE) < SECONDS_IN_DAY):
        data = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    else:
        # Download if file doesn't exist or is stale
        data = yf.download(tickers, period="4y", auto_adjust=True, progress=False)["Close"].tail(820)
        data.to_csv(DATA_FILE)

    # Ensure we only have data up to yesterday
    data.index = pd.to_datetime(data.index).normalize()
    
    # If the last row is 'today', drop it
    today = pd.Timestamp(date.today()).normalize()
    if not data.empty and data.index[-1] == today:
        data = data.iloc[:-1]
        
    return data


def get_rmt_threshold(N, d):
    """Calculates the Marchenko-Pastur upper bound."""
    q = N / d  # T/N ratio
    sigma_sq = 1 - (1/q)  # Variance of the 'noise' bulk, simplified for correlation
    lambda_plus = sigma_sq * (1 + np.sqrt(1/q))**2
    return lambda_plus


def eigen_decomp(sample_corr_mat, maximum_retries=20, constant=1e-10):
    """
    Attempts to calculate eigenvalues and eigenvectors, retries with a constant added to the diagonal if it fails
    to converge
    :param sample_corr_mat: sample correlation matrix from the data
    :param maximum_retries: Maximum allowed number of retries of the decomposition
    :param constant: constant to be added to enforce positive-semi-definite matrix
    :return: eigenvals, eigenvecs
    """

    if not np.isfinite(sample_corr_mat).all():
        raise ValueError("Non-finite values in sample correlation matrix. Automatic Termination")

    mat = sample_corr_mat.copy()

    for i in range(maximum_retries):
        try:
            eigenvals, eigenvecs = np.linalg.eigh(mat)
            return eigenvals, eigenvecs
        except np.linalg.LinAlgError as e:
            if i == maximum_retries-1:
                raise e

            np.fill_diagonal(mat, mat.diagonal() + constant)

    raise np.linalg.LinAlgError("Eigen-decomposition Failed")


def estimate_cov_matrix(sample_corr_mat, log_ret):
    """
    Use the Sample Correlation Matrix to estimate the Covariance Matrix using PCA for de-noising. Use RMT in order to
    determine how many factors should be used in the PCA.
    Note: Correlation is used from a STANDARDIZED set of data, so the covariance matrix -> correlation matrix. The
    matrix is descaled back to the covariance matrix in the function and returned
    :param log_ret: log returns of the training data
    :param sample_corr_mat: sample correlation matrix from the data
    :return: estimated covariance matrix and the amount of significant factors used to construct the matrix
    """
    standardized_log_ret = (log_ret - log_ret.mean()) / log_ret.std()
    # A. Find the number of significant factors
    eigenvals, eigenvecs = eigen_decomp(sample_corr_mat)   # Eigenvalues and Eigenvectors of sample covariance matrix

    idx = np.argsort(eigenvals)[::-1]  # Sort eigenvalues from largest to smallest

    eigenvals = eigenvals[idx]  # Sort eigenvalues in descending order
    eigenvecs = eigenvecs[:, idx]  # Sort eigenvectors in descending order

    N = standardized_log_ret.shape[0]  # number of time periods (rows)
    d = standardized_log_ret.shape[1]  # number of assets (columns/diagonal size of S)

    limit = get_rmt_threshold(N, d)

    significant_factors = np.sum(eigenvals > limit)


    # B. Use the Optimal Amount of Factors to Build the Cov Matrix
    beta = eigenvecs[:, :significant_factors]
    lamb = np.diag(eigenvals[:significant_factors])

    common_corr = beta @ lamb @ beta.T

    # Estimate Residual Variance
    d_diag = np.diag(sample_corr_mat) - np.diag(common_corr)
    D = np.diag(np.maximum(d_diag, 0))

    # Estimate Correlation Matrix with the PCA
    pca_corr = common_corr + D

    # Rescale back to Covariance
    vols = log_ret.std().values
    pca_F = np.outer(vols, vols) * pca_corr
    return pca_F, significant_factors


def prepare_panel_data(daily_prices):
    """
    Transforms daily price data into a weekly cross-sectional panel with features and targets.
    """
    # 1. Resample to Weekly (using Friday closes)
    weekly_prices = daily_prices.resample('W-FRI').last()

    # 2. Calculate Forward 1-Week Log Return
    # Shift by -1 so that the features calculated this Friday align with the return next Friday
    forward_returns = np.log(weekly_prices / weekly_prices.shift(1)).shift(-1)

    # 3. Calculate Price-Based Features
    mom_1m = np.log(weekly_prices / weekly_prices.shift(4))  # 1 month
    mom_6m = np.log(weekly_prices / weekly_prices.shift(26))  # 6 months
    mom_12m = np.log(weekly_prices / weekly_prices.shift(52))  # 1 Year

    # Volatility (Using daily data, then resampling to weekly)
    daily_returns = np.log(daily_prices / daily_prices.shift(1))
    vol_3m = daily_returns.rolling(window=63).std() * np.sqrt(252)  # 63 trading days ~ 3 months, annualized
    vol_3m_weekly = vol_3m.resample('W-FRI').last()

    vol_1m = daily_returns.rolling(window=22).std() * np.sqrt(252)  # 22 trading days ~ 1 months, annualized
    vol_1m_weekly = vol_1m.resample('W-FRI').last()

    # 4. Structure the Panel Data
    # Stack the dataframes to create a long format panel: MultiIndex (Date, Ticker)
    panel = pd.DataFrame({
        'Target_Fwd_Ret': forward_returns.stack(),
        'Mom_1M': mom_1m.stack(),
        'Mom_6M': mom_6m.stack(),
        'Mom_12M': mom_12m.stack(),
        'Vol_3M': vol_3m_weekly.stack(),
        'Vol_1M': vol_1m_weekly.stack()
    })

    panel = panel.dropna()

    return panel


def cross_sectional_standardize(panel_df, feature_cols):
    """
    Applies cross-sectional z-scoring to the features for each period.
    """
    standardized_panel = panel_df.copy()

    # Group by Date (level=0) and standardize across the assets for each feature
    standardized_panel[feature_cols] = panel_df.groupby(level=0)[feature_cols].transform(
        lambda x: (x - x.mean()) / x.std()
    )

    return standardized_panel


def walk_forward_ridge(panel_df, feature_cols, target_col='Target_Fwd_Ret', alpha=1.0, min_train_weeks=52):
    """
    Performs expanding-window walk-forward prediction over the panel data.
    """
    df = panel_df.sort_index(level=0)
    dates = df.index.get_level_values(0).unique()

    results = []

    # Iterate through time, starting after our minimum training window
    for i in range(min_train_weeks, len(dates) - 1):
        train_dates = dates[:i]
        test_date = dates[i]

        # Safely slice the MultiIndex panel
        idx = pd.IndexSlice
        train_data = df.loc[idx[train_dates, :], :]
        test_data = df.loc[idx[test_date, :], :]

        X_train = train_data[feature_cols]
        y_train = train_data[target_col]

        X_test = test_data[feature_cols]

        # Fit Ridge model
        model = Ridge(alpha=alpha, solver='svd')
        model.fit(X_train, y_train)

        # Predict out-of-sample for the test week
        preds = model.predict(X_test)

        # Store results
        test_res = test_data.copy()
        test_res['Predicted_Ret'] = preds
        results.append(test_res[[target_col, 'Predicted_Ret']])

    # Concatenate all out-of-sample predictions
    oos_results = pd.concat(results)
    return oos_results


def evaluate_predictions(oos_results, target_col='Target_Fwd_Ret', pred_col='Predicted_Ret'):
    """
    Evaluates predictions using Cross-Sectional Information Coefficient (Rank IC).
    """

    def calc_ic(group):
        corr, _ = spearmanr(group[target_col], group[pred_col])
        return corr

    # Calculate IC for each week
    ic_series = oos_results.groupby(level=0).apply(calc_ic).dropna()

    mean_ic = ic_series.mean()

    # Information Ratio of the IC (Mean IC / Std Dev of IC)
    ic_ir = mean_ic / ic_series.std() if ic_series.std() != 0 else 0

    return mean_ic, ic_ir


def optimize_ridge_penalty(panel_df, feature_cols, target_col='Target_Fwd_Ret',
                           alphas=np.logspace(-2, 4, 15), min_train_weeks=52):
    """
    Tests a grid of L2 penalties to find the optimal alpha for the dataset.
    """
    best_alpha = None
    best_ir = -np.inf
    best_oos_results = None

    for alpha in alphas:
        oos_res = walk_forward_ridge(panel_df, feature_cols, target_col, alpha, min_train_weeks)
        mean_ic, ic_ir = evaluate_predictions(oos_res, target_col)

        # Optimize for IC IR here, could optimize for MSE or Mean IC
        if ic_ir > best_ir:
            best_ir = ic_ir
            best_alpha = alpha

    return best_alpha


def apply_ema_smoothing(oos_results, pred_col='Predicted_Ret', span=3):
    """
    Applies an Exponential Moving Average to the predictions for each ticker over time.
    A span of 3 or 4 weeks is standard for mid-frequency weekly models.
    """
    smoothed_results = oos_results.copy()

    # Group by Ticker (level=1 in a MultiIndex of [Date, Ticker]), apply the EMA to the prediction column
    smoothed_preds = smoothed_results.groupby(level=1)[pred_col].transform(
        lambda x: x.ewm(span=span, adjust=False).mean()
    )

    smoothed_results['Smoothed_Predicted_Ret'] = smoothed_preds

    return smoothed_results.dropna(subset=['Smoothed_Predicted_Ret'])


def eff_front_no_shorts(posterior_returns, cov_matrix, lmbda=3.0):
    """
    Uses Quadratic Optimization to minimize the variance of a portfolio for a target return level.
    :param lmbda: Risk aversion parameter
    :param posterior_returns: Optimal return vector from the Black-Litterman framework
    :param cov_matrix: covariance matrix
    :return: minimum variance no shorting allocations
    """
    n = len(posterior_returns)
    cov_annual = cov_matrix * 252

    # Objective Function: Minimize Portfolio Variance
    def objective(w):
        ret = w.T @ posterior_returns
        risk = w.T @ cov_annual @ w
        return -(ret - (lmbda * risk))

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})  # Sum of weights = 1)
    bounds = tuple((0, .15) for _ in range(n))

    for i in range(10):
        init_guess = np.ones(n) / n

        res = minimize(objective, init_guess, method='SLSQP',
                    bounds=bounds, constraints=constraints)

        if res.success:
            return res.x

    # If the loop finishes without success, raise the error
    raise ValueError(f"Optimization failed after 10 attempts.")


# 1.) Perform the Backtest
# ------------------------
def run_backtest_calculation():
    data = get_data()

    if data is None or data.empty:
            # return {"error": "Failed to load market data."}
            print("error: Failed to load market data.")
        
    data.index = pd.to_datetime(data.index)  # Ensure the index is datetime

    optimal_weights = None  # Hold the array of position weights
    previous_week = None  # Used to determine when rebalancing should occur
    previous_month = None  # Used to determine when to re-optimize the Ridge regression
    backtest_period = 252 # Backtest over 252 trading days
    portfolio_daily_returns = {}  # Used in storing the portfolio returns
    daily_returns = data.tail(backtest_period+1).pct_change(fill_method=None).dropna()  # Series of returns 
    rebalance_dates = []  # Used in visualization
    optimal_alpha = None # Store the optimal alpha for the Ridge regression

    # Calculate the Market Caps for the Black-Litterman framework
    market_caps = {}
    for ticker in tickers:
        try:
            ticker_obj = yf.Ticker(ticker)
            cap = ticker_obj.info.get('marketCap')
            # Handle cases where marketCap might be None
            market_caps[ticker] = cap if cap is not None else 0
        except Exception as e:
            market_caps[ticker] = 0

    # Convert to Series and fill missing/zero values with median
    caps_series = pd.Series(market_caps)
    median_cap = caps_series[caps_series > 0].median()
    caps_series = caps_series.replace(0, median_cap)
    market_weights = caps_series / caps_series.sum()

    # Loop through the backtest period, rebalancing weekly
    for current_date, day_returns in daily_returns.iterrows():
        current_week = current_date.isocalendar()[1]  # Get the week number of the current date
        # If new week or beginning of the backtest, rebalance the portfolio
        if optimal_weights is None or current_week != previous_week:
            
            # Find necessary quantities
            window_data = data.loc[:current_date].tail(700)  # Use the last 700 trading days for training
            log_returns = np.log(window_data / window_data.shift(1)).dropna()
            standardized_log_returns = (log_returns - log_returns.mean()) / log_returns.std()  # Standardize the returns
            sample_corr_mat = np.cov(np.array(standardized_log_returns), rowvar=False)  # Sample correlation matrix
            cov_mat, _ = estimate_cov_matrix(sample_corr_mat, log_returns)  # Estimate the covariance

            # Prepare the panel data for the Ridge regression
            panel_data = prepare_panel_data(window_data)
            features = ['Mom_1M', 'Mom_6M', 'Mom_12M', 'Vol_3M']
            standardized_panel = cross_sectional_standardize(panel_data, features)
            alphas_to_test = np.logspace(0, 4, 20)
            if optimal_alpha is None or current_date.month != previous_month:

                # Get the optimal alpha and the raw predictions from the Ridge regression using walk-forward validation
                optimal_alpha = optimize_ridge_penalty(standardized_panel, features, 
                                                                    alphas=alphas_to_test, min_train_weeks=20)
                
                previous_month = current_date.month  # Update the previous month tracker
            
            raw_predictions = walk_forward_ridge(standardized_panel, features, alpha=optimal_alpha, min_train_weeks=20)

            smoothed_predictions = apply_ema_smoothing(raw_predictions, span=3)

            # Calculate the posterior returns using the smoothed predictions (Black-Litterman framework)
            delta = 3.0  # Standard
            cov_matrix = pd.DataFrame(cov_mat, index=tickers, columns=tickers)
        
            pi = delta * cov_matrix.dot(market_weights)
            pi_annual = pi * 252
            pi_series = pd.Series(pi_annual, index=tickers)  # PI in Black-Litterman Equations

            today_str = current_date.strftime('%Y-%m-%d')

            if today_str in smoothed_predictions.index.get_level_values(0):
                target_date = today_str
            else:
                # If today isn't a trading day, grab the most recent available date
                target_date = smoothed_predictions.index.get_level_values(0).max()

            tau = 0.05
            P = np.eye(len(tickers))  # P is an identity matrix because we have a view on every asset
            Q_raw = smoothed_predictions.xs(target_date, level=0)['Smoothed_Predicted_Ret'] * 52
            Q = Q_raw.reindex(tickers).fillna(0)  # Align Q with tickers and fill missing values with 0
            Q = Q.clip(lower=-0.5, upper=1)  # Don't allow for overly extreme predictions

            omega = np.diag(np.diag(tau * cov_matrix))

            inv_tau_sigma = np.linalg.inv(tau * cov_matrix)
            term1 = np.linalg.inv(inv_tau_sigma + P.T @ np.linalg.inv(omega) @ P)
            term2 = (inv_tau_sigma @ pi) + (P.T @ np.linalg.inv(omega) @ Q)
            posterior_returns = term1 @ term2  # Posterior expected returns from Black-Litterman
            print(posterior_returns)
            print(cov_mat)

            # Calculate the optimal weights using the posterior returns and covariance matrix
            optimal_weights = np.array(eff_front_no_shorts(posterior_returns, cov_mat))

            # Update the week tracker and rebalance date list
            rebalance_dates.append(current_date)
            previous_week = current_week

        portfolio_daily_returns[current_date] = 0.98 * day_returns.dot(optimal_weights)  # Portfolio return at date

    return_series = pd.Series(portfolio_daily_returns).sort_index()


    # 2.) Evaluate the performance of the strategy, calculate the Sharpe ratio and other metrics
    # ------------------------------------------------------------------------------------------
    cumulative_growth = (1 + return_series).cumprod() # Portfolio cumulative growth over the backtest period
    strategy_return = cumulative_growth.iloc[-1] - 1
    strategy_vol = return_series.std() * np.sqrt(252)  # Annualized volatility
    risk_free_rate = 0.04  # Assume a 4% annual risk-free rate
    strategy_sharpe_ratio = (strategy_return - risk_free_rate) / strategy_vol

    spy = yf.download("SPY", period="8mo", auto_adjust=True)["Close"].tail(120)  # Compare strategy to SPY
    spy_return_series = spy.pct_change(fill_method=None).dropna()
    spy_cum_return = (1 + spy_return_series).cumprod()

    historical_var_95 = return_series.quantile(0.05)  # 5% quantile for VaR
    historical_cvar_95 = return_series[return_series <= historical_var_95].mean()  # CVaR at 5%
    historical_var_99 = return_series.quantile(0.01)  # 1% quantile for VaR
    historical_cvar_99 = return_series[return_series <= historical_var_99].mean()  # CVaR at 1%

    max_drawdown = (cumulative_growth / cumulative_growth.cummax() - 1).min()

    # Risk Attribution: Contribution of each asset to the portfolio's risk
    asset_vols = daily_returns.std() * np.sqrt(252)  # Annualized volatility of each asset

    portfolio_weights = pd.Series(optimal_weights, index=tickers)
    cov_matrix = pd.DataFrame(cov_mat, index=tickers, columns=tickers)
    cov_matrix_annualized = cov_matrix * 252  # Annualize the covariance matrix
    portfolio_variance_annualized = portfolio_weights.T @ cov_matrix_annualized @ portfolio_weights
    portfolio_vol_annualized = np.sqrt(portfolio_variance_annualized)
    marginal_contribution = (cov_matrix_annualized @ portfolio_weights) / portfolio_vol_annualized
    vol_contribution = portfolio_weights * marginal_contribution
    vol_contribution_percent = vol_contribution / portfolio_vol_annualized

    return {
        "metrics": {
            "return": float(strategy_return),
            "volatility": float(strategy_vol),
            "sharpe": float(strategy_sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "var_95": float(historical_var_95),
            "var_99": float(historical_var_99)
        },
        "attribution": {
            ticker: float(val) for ticker, val in vol_contribution_percent.items() if val > 0.001  
        }
    }
