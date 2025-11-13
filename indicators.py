import pandas as pd
import numpy as np

def calculate_moving_averages(df, windows=[5, 20]):
    """
    Calculate moving averages for specified windows.
    
    Args:
        df (pd.DataFrame): DataFrame with 'Close' column
        windows (list): List of window sizes for moving averages
    
    Returns:
        pd.DataFrame: DataFrame with moving average columns added
    """
    df_copy = df.copy()
    
    for window in windows:
        df_copy[f'MA_{window}'] = df_copy['Close'].rolling(window=window).mean()
    
    return df_copy

def calculate_daily_returns(df):
    """
    Calculate daily returns.
    
    Args:
        df (pd.DataFrame): DataFrame with 'Close' column
    
    Returns:
        pd.DataFrame: DataFrame with 'Daily_Return' column added
    """
    df_copy = df.copy()
    df_copy['Daily_Return'] = df_copy['Close'].pct_change()
    return df_copy

def calculate_volatility(df, window=10):
    """
    Calculate rolling volatility (standard deviation of returns).
    
    Args:
        df (pd.DataFrame): DataFrame with 'Daily_Return' column
        window (int): Rolling window size
    
    Returns:
        pd.DataFrame: DataFrame with 'Volatility' column added
    """
    df_copy = df.copy()
    df_copy['Volatility'] = df_copy['Daily_Return'].rolling(window=window).std()
    return df_copy

def calculate_rsi(df, window=14):
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        df (pd.DataFrame): DataFrame with 'Close' column
        window (int): RSI window size
    
    Returns:
        pd.DataFrame: DataFrame with 'RSI' column added
    """
    df_copy = df.copy()
    
    # Calculate price changes
    delta = df_copy['Close'].diff()
    
    # Separate gains and losses
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    # Calculate RS and RSI
    rs = gain / loss
    df_copy['RSI'] = 100 - (100 / (1 + rs))
    
    return df_copy

def calculate_bollinger_bands(df, window=20, num_std=2):
    """
    Calculate Bollinger Bands.
    
    Args:
        df (pd.DataFrame): DataFrame with 'Close' column
        window (int): Moving average window
        num_std (int): Number of standard deviations
    
    Returns:
        pd.DataFrame: DataFrame with Bollinger Bands columns added
    """
    df_copy = df.copy()
    
    # Calculate moving average
    df_copy['BB_Middle'] = df_copy['Close'].rolling(window=window).mean()
    
    # Calculate standard deviation
    bb_std = df_copy['Close'].rolling(window=window).std()
    
    # Calculate upper and lower bands
    df_copy['BB_Upper'] = df_copy['BB_Middle'] + (bb_std * num_std)
    df_copy['BB_Lower'] = df_copy['BB_Middle'] - (bb_std * num_std)
    
    return df_copy

def calculate_macd(df, fast=12, slow=26, signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        df (pd.DataFrame): DataFrame with 'Close' column
        fast (int): Fast EMA period
        slow (int): Slow EMA period
        signal (int): Signal line period
    
    Returns:
        pd.DataFrame: DataFrame with MACD columns added
    """
    df_copy = df.copy()
    
    # Calculate EMAs
    ema_fast = df_copy['Close'].ewm(span=fast).mean()
    ema_slow = df_copy['Close'].ewm(span=slow).mean()
    
    # Calculate MACD line
    df_copy['MACD'] = ema_fast - ema_slow
    
    # Calculate signal line
    df_copy['MACD_Signal'] = df_copy['MACD'].ewm(span=signal).mean()
    
    # Calculate histogram
    df_copy['MACD_Histogram'] = df_copy['MACD'] - df_copy['MACD_Signal']
    
    return df_copy

def add_all_technical_indicators(df):
    """
    Add all technical indicators to the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with 'Close' column
    
    Returns:
        pd.DataFrame: DataFrame with all technical indicators added
    """
    df_copy = df.copy()
    
    # Add basic indicators
    df_copy = calculate_daily_returns(df_copy)
    df_copy = calculate_moving_averages(df_copy)
    df_copy = calculate_volatility(df_copy)
    
    # Add advanced indicators
    df_copy = calculate_rsi(df_copy)
    df_copy = calculate_bollinger_bands(df_copy)
    df_copy = calculate_macd(df_copy)
    
    return df_copy 