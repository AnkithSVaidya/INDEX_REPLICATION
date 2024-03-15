import cvxpy as cp
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error

class PortfolioOptimizer(BaseEstimator):
    def __init__(self,
                lambda_val,
                mu):
        self.lambda_val = lambda_val
        self.mu = mu

    def fit(self,
            stock_returns,
            X,
            y = None
            ):
        w = cp.Variable(X.shape[1])
        stock_expected_returns_array = np.array(stock_returns.mean()).reshape(1, -1)
        y_reshaped = y.reshape(-1, 1)
        xw_reshaped = (X @ w).reshape((-1, 1),
                       order='C')

        stock_expected_returns_repeated = np.repeat(stock_expected_returns_array,
                                                    repeats=xw_reshaped.shape[0], axis=0)

        objective = cp.Minimize(0.5 * cp.sum_squares(xw_reshaped - y_reshaped) + self.lambda_val * cp.norm(w, 1) +
                                self.mu * cp.sum_squares(xw_reshaped - stock_expected_returns_repeated))
        constraints = [cp.sum(w) == 1, w >= 0]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        self.optimal_weights_ = w.value
        return self

def tracking_error_scorer(estimator,
                          X,
                          y):
    portfolio_returns = X @ estimator.optimal_weights_
    return -mean_squared_error(y, portfolio_returns)

def calculate_annualized_stats(daily_returns):
    annualized_return = daily_returns.mean() * 252
    annualized_volatility = daily_returns.std() * np.sqrt(252)
    return annualized_return, annualized_volatility
