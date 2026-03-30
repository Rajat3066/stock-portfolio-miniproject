import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import entropy
from sentiment_builder import build_sentiment_series


def hurst(ts):
    lags = range(2, 20)
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    return np.polyfit(np.log(lags), np.log(tau), 1)[0]


def rolling_entropy(series, window=20):
    return series.rolling(window).apply(
        lambda x: entropy(np.histogram(x, bins=10)[0] + 1)
    )


def load_stock_data(ticker="AAPL", period="5y"):
    data = yf.download(ticker, period=period, interval="1d")

    if data is None or data.empty:
        raise ValueError(f"Failed to download data for {ticker}")

    data.columns = data.columns.get_level_values(0)
    data = data[['Close']]
    data = data.tail(1200)

    # Returns
    data['Return'] = np.log(data['Close'] / data['Close'].shift(1))
    data['Ret_1'] = data['Return']
    data['Ret_3'] = data['Close'].pct_change(3)
    data['Ret_5'] = data['Close'].pct_change(5)

    # Z-Score
    rolling_mean = data['Close'].rolling(20).mean()
    rolling_std  = data['Close'].rolling(20).std()
    data['ZScore'] = (data['Close'] - rolling_mean) / rolling_std

    # Trend
    ma_short = data['Close'].rolling(10).mean()
    ma_long  = data['Close'].rolling(30).mean()
    data['Trend'] = ma_short - ma_long

    # Volatility regime
    data['Vol_Regime'] = data['Return'].rolling(20).std()

    # Momentum strength
    data['Momentum_Strength'] = data['Ret_5'] / (data['Vol_Regime'] + 1e-6)

    # Hurst exponent
    data['Hurst'] = data['Close'].rolling(50).apply(hurst)

    # Entropy
    data['Entropy'] = rolling_entropy(data['Return'])

    # Clipping
    data['Momentum_Strength'] = data['Momentum_Strength'].clip(-10, 10)
    data['Hurst'] = data['Hurst'].clip(0, 1)
    data['Entropy'] = data['Entropy'].clip(0, 5)

    # Clean
    data = data.bfill().ffill()
    data.replace([np.inf, -np.inf], 0, inplace=True)

    print(f"Final data shape: {data.shape}")

    # Sentiment
    print("Building sentiment... (this may take a moment per date)")
    sentiment_dict = build_sentiment_series(ticker, data.index)
    data['Sentiment'] = data.index.strftime("%Y-%m-%d").map(sentiment_dict)
    data['Sentiment'] = data['Sentiment'].fillna(0)
    print("Sentiment added")

    return data
