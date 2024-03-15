import numpy as np
import cvxpy as cp
from portfolio_optimizer.portfolio_optimizer import PortfolioOptimizer

def generate_gaussian_data(T, n, mean_vector, cov_matrix):
    """
    This function is used to generate multivariate and univariate Gaussian time series.

    Args:
    T: Number of time steps.
    n: Number of features in the multivariate time series.
    mean_vector: Mean vector for the multivariate time series.
    cov_matrix: Covariance matrix for the multivariate time series.

    Function returns:
    X: Multivariate Gaussian time series (T x n).
    y: Univariate Gaussian time series (T).
    """
    X = np.random.multivariate_normal(mean_vector, cov_matrix, size=T)
    y = np.random.normal(loc=2.0, scale=1.0, size=T)
    return X, y

def test_portfolio_optimizer():
    """
    This function tests the portfolio optimizer using synthetic data.
    """
    T = 100
    n = 5
    mean_vector = np.random.rand(n)
    cov_matrix = np.random.rand(n, n)
    cov_matrix = np.dot(cov_matrix, cov_matrix.T)
    X, y = generate_gaussian_data(T, n, mean_vector, cov_matrix)

    # Test portfolio optimizer here
    n = X.shape[1]
    w = cp.Variable(n)
    objective = cp.Minimize(cp.norm2(X @ w - y) ** 2)
    constraints = [cp.sum(w) == 1, w >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS, abstol=1e-6, reltol=1e-6)

    # To check optimization status
    if problem.status == 'optimal':
        logger.info("Optimization successful!")

        # Check optimized weights
        if np.isclose(np.sum(w.value), 1.0) and np.all(w.value >= 0):
            logger.info("Optimized weights sum to 1 and are non-negative.")
        else:
            logger.warning("Optimized weights do not meet requirements.")
    else:
        logger.error("Optimization failed:", problem.status)

    
