# Nicholas Christophides  nick.christophides@gmail.com

""" In estimate_cov_matrix.py the covariance matrix between the chosen assets in the portfolio is estimated and
de-noised by using Principal Component Analysis (PCA). Random Matrix Theory (RMT) is employed in order to choose
the amount of factors that are deemed to be significant. This estimate is then used in the portfolio optimization. """


import numpy as np
import yfinance as yf


def get_rmt_threshold(N, d):
    """Calculates the Marchenko-Pastur upper bound."""
    q = N / d  # T/N ratio
    sigma_sq = 1 - (1/q)  # Variance of the 'noise' bulk, simplified for correlation
    lambda_plus = sigma_sq * (1 + np.sqrt(1/q))**2
    return lambda_plus


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# 1.) Get Financial Data and Find the Sample Covariance Matrix
# ------------------------------------------------------------
tickers = ['AMT',     # American Tower      Sector: Real Estate
           'BRK-B',   # Berkshire Hathaway  Sector: Financials
           'CAT',     # Caterpillar         Sector: Industrials
           'COST',    # Costco              Sector: Consumer Staples
           'GE',      # GE Aerospace        Sector: Industrials
           'HD',      # Home Depot          Sector: Consumer Disc.
           'JNJ',     # Johnson & Johnson   Sector: Health Care
           'MSFT',    # Microsoft           Sector: Information Tech
           'NEE',     # NextEra Energy      Sector: Utilities
           'NVDA',    # NVIDIA              Sector: Information Tech
           'PG',      # Proctor & Gamble    Sector: Consumer Staples
           'PLD',     # Prologis            Sector: Real Estate
           'SPY',     # S&P 500             Market Index
           'TSLA',    # Tesla               Sector: Consumer Disc.
           'UNH',     # UnitedHealth        Sector: Health Care
           'V',       # Visa                Sector: Financials
           'XOM']     # ExxonMobil          Sector: Energy

data_amount = 700  # Amount of data wanted in the optimization
prices = yf.download(tickers, period="4y", auto_adjust=True)["Close"]  # Extracted prices from yfinance
prices = prices.dropna().tail(data_amount)
log_returns = np.log(prices / prices.shift(1)).dropna()  # Prices turned into returns
log_returns_std = (log_returns - log_returns.mean()) / log_returns.std()

# For standardized data, the Covariance Matrix = Correlation Matrix
sample_corr = np.cov(np.array(log_returns_std), rowvar=False)  # Sample Covariance matrix of the assets listed above


# 2.) Estimate Covariance Matrix using PCA and Significant Factors using RMT
# --------------------------------------------------------------------------

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
                print(f"Failed to converge after {maximum_retries} retries.")
                raise e

            print(f"Converge failed on try {i+1}, adding a constant...")

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

    # print(f"Marchenko-Pastur Noise Threshold: {limit:.4f}")
    # print(f"Number of 'Signal' Factors: {significant_factors}")

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


pca_F, significant_factors = estimate_cov_matrix(sample_corr, log_returns)
