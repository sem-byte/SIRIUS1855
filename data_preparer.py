import yfinance as yf
import pandas_ta as ta
import pandas as pd

def prepare_data(symbol, start_date, end_date):
    """
    Fetches historical OHLCV data from yfinance and calculates technical indicators.

    Args:
        symbol (str): The ticker symbol to fetch data for.
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format.

    Returns:
        pandas.DataFrame: A DataFrame with OHLCV data and technical indicators.
    """
    # Fetch data from yfinance
    df = yf.download(symbol, start=start_date, end=end_date)

    if df.empty:
        raise ValueError("No data fetched for the given symbol and date range.")

    # Handle MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Calculate technical indicators using pandas_ta
    df.ta.ema(length=200, append=True)
    df.ta.vwap(length=24, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.atr(length=14, append=True)

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
