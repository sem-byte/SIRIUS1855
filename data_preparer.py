import pandas as pd
import pandas_ta as ta
from bybit_client import BybitClient

def prepare_data(symbol, start_date, end_date):
    """
    Fetches historical OHLCV data from Bybit and calculates technical indicators.

    Args:
        symbol (str): The ticker symbol to fetch data for.
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format.

    Returns:
        pandas.DataFrame: A DataFrame with OHLCV data and technical indicators.
    """
    # Initialize Bybit client and fetch data
    client = BybitClient()
    df = client.fetch_ohlcv(symbol, start_date, end_date)

    if df.empty:
        raise ValueError("No data fetched for the given symbol and date range.")

    # Handle MultiIndex columns from yfinance (in case the placeholder is used)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Fetch additional data
    open_interest = client.fetch_open_interest(symbol, start_date, end_date)
    long_short_ratio = client.fetch_long_short_ratio(symbol, start_date, end_date)
    funding_rate = client.fetch_funding_rate(symbol, start_date, end_date)

    # Merge additional data
    df = df.join(open_interest).join(long_short_ratio).join(funding_rate)

    # Calculate technical indicators using pandas_ta
    df.ta.ema(length=200, append=True)
    df.ta.vwap(length=24, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)

    # Normalize ATR
    df['ATR_p'] = (df['ATRr_14'] / df['Close']) * 100

    # Drop rows with NaN values
    df.dropna(inplace=True)

    return df

def filter_trend_data(df):
    """
    Filters the DataFrame for trend conditions (ADX > 25).
    """
    return df[df['ADX_14'] > 25].copy()

def filter_range_data(df):
    """
    Filters the DataFrame for range-bound conditions (ADX < 20).
    """
    return df[df['ADX_14'] < 20].copy()
