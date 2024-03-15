import pandas as pd
import numpy as np
import logging
from data.data_loader import load_data
from portfolio_optimizer.portfolio_optimizer import PortfolioOptimizer, calculate_annualized_stats
# Set up the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def main():
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']  # Increased number of stocks for diversification
    start_date = '2015-01-01'  # Increased data window for backtesting
    end_date = '2023-03-10'
    index_ticker = '^GSPC'
    X, y, stock_returns, index_returns  = load_data(tickers,index_ticker, start_date, end_date)

    model = PortfolioOptimizer(lambda_val=0.1, mu=1.0)
    model.fit(stock_returns, X, y)

    optimal_weights = model.optimal_weights_

    portfolio_returns = (stock_returns * optimal_weights).sum(axis=1)
    annualized_return, annualized_volatility = calculate_annualized_stats(portfolio_returns)

    excess_returns = portfolio_returns - index_returns
    information_ratio = annualized_return / annualized_volatility

    logger.info(f"Tracking Error: {np.std(excess_returns)}")
    logger.info(f"Annualized Return: {annualized_return:.4f}")
    logger.info(f"Annualized Volatility: {annualized_volatility:.4f}")
    logger.info(f"Information Ratio: {information_ratio:.4f}")

if __name__ == "__main__":
    main()

