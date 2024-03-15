import pandas as pd
import yfinance as yf

def load_data(tickers, index_ticker, start_date, end_date):
    try:
        # Downloading the stock data from Yahoo Finance
        stock_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

        # Downloading the index data from Yahoo Finance
        index_data = yf.download(index_ticker, start=start_date, end=end_date)['Adj Close']

        # Calculating daily returns for stocks
        stock_returns = stock_data.pct_change().dropna()
        X = stock_returns.values

        # Calculating daily returns for the index
        index_returns = index_data.pct_change().dropna()
        y = index_returns.values

        return X, y, stock_returns, index_returns

    except Exception as e:
        logger.info(f"An error occurred during data loading: {e}")
        return None, None
