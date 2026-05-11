# Nicholas Christophides  nick.christophides@gmail.com

""" In portfolio_optimization.py the estimated covariance matrix from estimate_cov_matrix.py is employed in performing
portfolio optimization of the chosen assets.
The theoretically efficient portfolio is then used in auto_rebalance.py. """

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from estimate_cov_matrix import log_returns, pca_F, significant_factors
from scipy.optimize import minimize


# 1.) Find the Efficient Frontier using the PCA Covariance Matrix Estimate
# ------------------------------------------------------------------------

def eff_front_no_shorts(mu_target, mu_hat, cov_matrix):
    """
    Uses Quadratic Optimization to minimize the variance of a portfolio for a target return level.
    :param mu_target: target return
    :param mu_hat: mean return vector
    :param cov_matrix: covariance matrix
    :return: minimum variance no shorting allocations
    """
    n = len(mu_hat)

    # Objective Function: Minimize Portfolio Variance
    def objective(w):
        return w.T @ cov_matrix @ w

    # Constraints
    constraints = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum of weights = 1
        {'type': 'eq', 'fun': lambda w: w.T @ mu_hat - mu_target}  # Target return
    )

    # Bounds: No shorting
    bounds = tuple((0, 1) for _ in range(n))

    init_guess = np.ones(n) / n

    res = minimize(objective, init_guess, method='SLSQP',
                   bounds=bounds, constraints=constraints)

    return res.x if res.success else None


def find_gmv_return(mu_hat, cov_matrix):
    """
    The Global Minimum Variance (GMV) portfolio has the absolute lowest risk of the efficient allocations. Therefore,
    the return level at this point can be used as the minimum target return threshold.
    :param mu_hat: mean return vector
    :param cov_matrix: covariance matrix
    :return: minimum target return
    """
    n = len(mu_hat)

    def objective(w):
        return w @ cov_matrix @ w

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = tuple((0, 1) for _ in range(n))
    init_guess = np.ones(n) / n

    res = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return res.x @ mu_hat  # This is the return of the GMV portfolio


mu_hat = log_returns.mean().values

mu_gmv = find_gmv_return(mu_hat, pca_F)
mu_max = np.max(mu_hat)

std_devs = []
expected_returns = []
mus = np.linspace(mu_gmv, mu_max, 25)

for mu_star in mus:
    w_opt = eff_front_no_shorts(mu_star, mu_hat, pca_F)

    if w_opt is not None:
        std_devs.append(np.sqrt(w_opt.T @ pca_F @ w_opt))
        expected_returns.append(w_opt.T @ mu_hat)

std_devs_pca = np.array(std_devs)
expected_returns_pca = np.array(expected_returns)

# Plot curves
plt.figure(figsize=(12, 6))
plt.plot(std_devs, expected_returns, marker="o", label=f"PCA {significant_factors} Factor Model")
plt.xlabel("Std. Dev. (Risk)")
plt.ylabel("Expected Return")
plt.title("Sample Efficient Frontier Based on Shrinkage Estimates of ∑ using PCA")
plt.grid(True)
plt.legend()
plt.style.use('seaborn-v0_8')
# plt.show()


# 2.) Choose an Expected Return and Find the Portfolio Weights
# ------------------------------------------------------------

optimal_weights = eff_front_no_shorts(mus[12], mu_hat, pca_F)

optimal_portfolio = pd.DataFrame({"Asset": log_returns.columns,
                                  "Weight": optimal_weights}
                                 ).sort_values(by="Weight", ascending=False).set_index("Asset")

print(f"Expected Daily Return of the Portfolio: {optimal_weights.T @ mu_hat}\n"
      f"Expected Daily Volatility of the Portfolio: {optimal_weights.T @ pca_F @ optimal_weights}\n"
      f"Optimal Portfolio Weights: \n{optimal_portfolio.round(4)}\n"
      f"Total Portfolio Weight: {np.sum(optimal_weights):.4f}")
