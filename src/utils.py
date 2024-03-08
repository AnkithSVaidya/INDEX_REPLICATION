from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd
import requests

def load_data():
    # Load environment variables
    api_key = os.environ.get("API_KEY")
    company_name = os.environ.get("COMPANY_NAME")
    timing = os.environ.get("TIMING")

    # Constructing the URL
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={company_name}&interval={timing}&apikey={api_key}'

    try:
        # Making HTTP request
        r = requests.get(url)
        r.raise_for_status()  # This line will raise an exception for 4xx or 5xx status codes

        # Here I am parsing the JSON response
        data = r.json()
        return data

    except requests.exceptions.RequestException as e:
        print("Error making HTTP request:", e)
        return None  # This line returns None to indicate failure

    # Return X_train, y_train, X_test, y_test


def preprocess_data(data):
    # Preprocess the loaded data
    # Return preprocessed data
    # In the below lines I am extracting the time-series data 
    time_series_data = data['Time Series (5min)']
    time_series_data = pd.DataFrame(time_series_data).T
    time_series_data.index.name = 'timestamp'

    # Renaming the columns
    time_series_data.columns = [col.split('. ')[1] for col in time_series_data.columns]

    # Convert columns to numeric
    time_series_data = time_series_data.apply(pd.to_numeric)

    # This line can be included/modified based on the requirements of analysis, if forward filling, backward filling or any other imputation techniques 
    # preprocessed_data = time_series_data.fillna(0)

    return time_series_data

