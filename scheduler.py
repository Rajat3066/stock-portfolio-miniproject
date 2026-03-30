#!/usr/bin/env python
# ============================================================
# STOCK MARKET PORTFOLIO — MIDNIGHT TRAINING SCHEDULER
# Trains:
#   3-Month  → BiLSTM + XGBoost + LightGBM + Ridge Meta-Ensemble
#   1-Year   → GRU + Sentiment Analysis (Finnhub)
#   5-Year   → Transformer + CAGR Blend
# Saves all predictions to predictions_cache.json
# ============================================================

import json, math, os, sys, time, warnings, schedule, logging
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')
log = logging.getLogger(__name__)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

import ta
from scipy.stats import entropy as scipy_entropy

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, LSTM, Dropout, Bidirectional,
                                     Input, LayerNormalization,
                                     MultiHeadAttention, GlobalAveragePooling1D,
                                     GRU)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

CACHE_FILE = "predictions_cache.json"
TICKERS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA']

# ─────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────
def safe_float(v):
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    return v

def row_to_dict(row):
    return {k: safe_float(v) for k, v in row.items()}

# ─────────────────────────────────────────────
# 3-MONTH MODEL  (BiLSTM + XGBoost + LightGBM)
# ─────────────────────────────────────────────
def train_3month():
    log.info("=== 3-MONTH TRAINING STARTED ===")

    stocks = {}
    for t in TICKERS:
        df = yf.download(t, period='3y', interval='1d')
        df.columns = df.columns.get_level_values(0)
        stocks[t] = df

    vix_df = yf.download('^VIX', period='3y', interval='1d')
    vix_df.columns = vix_df.columns.get_level_values(0)
    vix_df = vix_df[['Close']].rename(columns={'Close': 'VIX'})

    FEATURE_COLS = [
        'Return','Return_lag1','Return_lag2','Return_lag3','Return_lag5',
        'MA_ratio','RSI','MACD','MACD_diff',
        'BB_width','BB_position','Volume_ratio',
        'High_Low_ratio','Close_Open_ratio','VIX'
    ]
    TIMESTEPS = 90

    def add_features(df):
        df = df.copy()
        close  = df['Close'].squeeze()
        high   = df['High'].squeeze()
        low    = df['Low'].squeeze()
        volume = df['Volume'].squeeze()
        open_  = df['Open'].squeeze()

        df['Return']      = close.pct_change()
        df['Return_lag1'] = df['Return'].shift(1)
        df['Return_lag2'] = df['Return'].shift(2)
        df['Return_lag3'] = df['Return'].shift(3)
        df['Return_lag5'] = df['Return'].shift(5)
        df['MA_20']       = close.rolling(20).mean()
        df['MA_50']       = close.rolling(50).mean()
        df['MA_ratio']    = df['MA_20'] / df['MA_50']
        df['RSI']         = ta.momentum.RSIIndicator(close).rsi()
        macd              = ta.trend.MACD(close)
        df['MACD']        = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff']   = macd.macd_diff()
        bb                = ta.volatility.BollingerBands(close)
        df['BB_width']    = (bb.bollinger_hband() - bb.bollinger_lband()) / close
        df['BB_position'] = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-9)
        df['Volume_MA']   = volume.rolling(20).mean()
        df['Volume_ratio']= volume / (df['Volume_MA'] + 1e-9)
        df['High_Low_ratio']   = high / low
        df['Close_Open_ratio'] = close / open_
        df['Target_90d']  = df['Close'].pct_change(90).shift(-90)
        cap               = df['Target_90d'].quantile(0.80)
        df['Target_90d']  = df['Target_90d'].clip(upper=cap)
        df.dropna(inplace=True)
        return df

    def create_dataset(features, target, ts=90):
        X, y = [], []
        for i in range(len(features) - ts - 1):
            X.append(features[i:i+ts, :])
            y.append(target[i+ts, 0])
        return np.array(X), np.array(y)

    def build_bilstm(ts=90, nf=15):
        m = Sequential([
            Bidirectional(LSTM(128, return_sequences=True), input_shape=(ts, nf)),
            Dropout(0.2),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.2),
            Bidirectional(LSTM(32)),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        m.compile(loss='mse', optimizer='adam')
        return m

    cbs = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, verbose=0)
    ]

    for t in TICKERS:
        stocks[t] = add_features(stocks[t])
        stocks[t] = stocks[t].join(vix_df['VIX'], how='left')
        stocks[t]['VIX'] = stocks[t]['VIX'].ffill().bfill()
        stocks[t].dropna(inplace=True)

    results_3m = {}

    for ticker in TICKERS:
        log.info(f"  3M | Training {ticker}...")
        df = stocks[ticker].copy()

        fsc  = MinMaxScaler(); tsc = MinMaxScaler(); t9sc = MinMaxScaler()
        fs   = fsc.fit_transform(df[FEATURE_COLS].values)
        _    = tsc.fit_transform(df[['Return']].values)
        t9s  = t9sc.fit_transform(df[['Target_90d']].values)

        split = int(len(fs) * 0.65)
        X_tr, y_tr = create_dataset(fs[:split],    t9s[:split],   TIMESTEPS)
        X_te, y_te = create_dataset(fs[split:],    t9s[split:],   TIMESTEPS)

        bilstm = build_bilstm(TIMESTEPS, len(FEATURE_COLS))
        bilstm.fit(X_tr, y_tr, epochs=200, batch_size=32,
                   validation_split=0.1, callbacks=cbs, verbose=0)

        X_tr_2d = X_tr.reshape(len(X_tr), -1)
        X_te_2d = X_te.reshape(len(X_te), -1)
        xgb = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8,
                           min_child_weight=3, reg_alpha=0.1, verbosity=0)
        xgb.fit(X_tr_2d, y_tr)

        lgbm = LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, verbose=-1)
        lgbm.fit(X_tr_2d, y_tr)

        b_tr = t9sc.inverse_transform(bilstm.predict(X_tr, verbose=0).reshape(-1,1))
        b_te = t9sc.inverse_transform(bilstm.predict(X_te, verbose=0).reshape(-1,1))
        x_tr = t9sc.inverse_transform(xgb.predict(X_tr_2d).reshape(-1,1))
        x_te = t9sc.inverse_transform(xgb.predict(X_te_2d).reshape(-1,1))
        l_tr = t9sc.inverse_transform(lgbm.predict(X_tr_2d).reshape(-1,1))
        l_te = t9sc.inverse_transform(lgbm.predict(X_te_2d).reshape(-1,1))
        y_tr_inv = t9sc.inverse_transform(y_tr.reshape(-1,1))

        mn = min(len(b_tr), len(x_tr), len(l_tr))
        meta_X_tr = np.column_stack([b_tr[:mn], x_tr[:mn], l_tr[:mn]])
        meta = Ridge(alpha=1.0)
        meta.fit(meta_X_tr, y_tr_inv[:mn])

        x_inp = fs[-TIMESTEPS:].reshape(1, TIMESTEPS, len(FEATURE_COLS))
        pred_scaled = bilstm.predict(x_inp, verbose=0)[0][0]
        cumret = float(t9sc.inverse_transform([[pred_scaled]])[0][0])
        cumret = np.clip(cumret, -0.50, 0.50)

        daily_base = (1 + cumret) ** (1/90) - 1
        hist_vol   = float(stocks[ticker]['Return'].std())
        np.random.seed(42)
        noise      = np.random.normal(0, hist_vol * 0.3, 90)
        daily_eq   = (daily_base + noise).tolist()

        hist_vol_ann = hist_vol * np.sqrt(252)
        risk_free    = 0.065
        ann_ret      = (1 + cumret) ** (365/90) - 1
        sharpe       = (ann_ret - risk_free) / (hist_vol_ann + 1e-9)
        risk_label   = 'Low' if hist_vol_ann < 0.15 else ('Medium' if hist_vol_ann < 0.30 else 'High')
        current_price = float(df['Close'].iloc[-1])

        results_3m[ticker] = {
            'cumulative_return_pct': round(cumret * 100, 2),
            'annualized_pct':        round(ann_ret * 100, 2),
            'sharpe':                round(sharpe, 2),
            'volatility_pct':        round(hist_vol_ann * 100, 2),
            'risk':                  risk_label,
            'current_price':         round(current_price, 2),
            'daily_returns':         daily_eq,
            'score': (0.50 * cumret * 100 + 0.30 * sharpe * 10 + 0.20 * (1 / (hist_vol_ann + 1)))
        }
        log.info(f"  3M | {ticker} done — predicted 90d return: {cumret*100:.2f}%")

    log.info("=== 3-MONTH TRAINING COMPLETE ===")
    return results_3m

# ─────────────────────────────────────────────
# 1-YEAR MODEL  (GRU + Sentiment)
# ─────────────────────────────────────────────
def hurst(ts):
    lags = range(2, 20)
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    return np.polyfit(np.log(lags), np.log(tau), 1)[0]

def rolling_entropy(series, window=20):
    return series.rolling(window).apply(
        lambda x: scipy_entropy(np.histogram(x, bins=10)[0] + 1)
    )

def fetch_sentiment_for_ticker(ticker, dates, api_key):
    """
    Fetch Finnhub news in MONTHLY batches (6 months only) instead of
    one API call per day.  1200 calls → ~6 calls per ticker.
    """
    import requests
    from nltk.sentiment import SentimentIntensityAnalyzer
    from datetime import timedelta
    sia = SentimentIntensityAnalyzer()

    # Only care about the last 6 months — older news doesn't help 1-year prediction
    cutoff = pd.Timestamp.today() - pd.DateOffset(months=6)
    recent_dates = [d for d in dates if pd.Timestamp(d) >= cutoff]

    # Build monthly windows: one API call per month instead of per day
    sentiment_dict = {}
    if not recent_dates:
        return sentiment_dict

    window_start = pd.Timestamp(min(recent_dates)).to_pydatetime().date()
    window_end   = pd.Timestamp(max(recent_dates)).to_pydatetime().date()

    all_articles = []
    cursor = window_start
    while cursor <= window_end:
        month_end = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        batch_end = min(month_end, window_end)
        try:
            resp = requests.get(
                "https://finnhub.io/api/v1/company-news",
                params={
                    "symbol": ticker,
                    "from":   cursor.strftime("%Y-%m-%d"),
                    "to":     batch_end.strftime("%Y-%m-%d"),
                    "token":  api_key
                },
                timeout=12
            )
            batch = resp.json()
            if isinstance(batch, list):
                all_articles.extend(batch)
            log.info(f"  1Y | {ticker} sentiment batch {cursor} → {batch_end}  ({len(batch) if isinstance(batch,list) else 0} articles)")
        except Exception as e:
            log.warning(f"  1Y | sentiment batch failed {cursor}: {e}")

        # Move to next month
        cursor = batch_end + timedelta(days=1)

    # Map each article to its date, score it, then average per day
    day_scores = {}
    for article in all_articles:
        if not isinstance(article, dict):
            continue
        ts = article.get("datetime", 0)
        if not ts:
            continue
        day = pd.Timestamp(ts, unit="s").strftime("%Y-%m-%d")
        headline = article.get("headline", "")
        if headline:
            score = sia.polarity_scores(headline)["compound"]
            day_scores.setdefault(day, []).append(score)

    for day, scores in day_scores.items():
        sentiment_dict[day] = float(np.mean(scores))

    log.info(f"  1Y | {ticker} — sentiment done ({len(sentiment_dict)} days covered)")
    return sentiment_dict

def load_1year_stock_data(ticker, api_key):
    data = yf.download(ticker, period="5y", interval="1d")
    if data is None or data.empty:
        raise ValueError(f"No data for {ticker}")
    data.columns = data.columns.get_level_values(0)
    data = data[['Close']].tail(1200)

    data['Return']     = np.log(data['Close'] / data['Close'].shift(1))
    data['Ret_1']      = data['Return']
    data['Ret_3']      = data['Close'].pct_change(3)
    data['Ret_5']      = data['Close'].pct_change(5)
    rolling_mean       = data['Close'].rolling(20).mean()
    rolling_std        = data['Close'].rolling(20).std()
    data['ZScore']     = (data['Close'] - rolling_mean) / rolling_std
    data['Trend']      = data['Close'].rolling(10).mean() - data['Close'].rolling(30).mean()
    data['Vol_Regime'] = data['Return'].rolling(20).std()
    data['Momentum_Strength'] = (data['Ret_5'] / (data['Vol_Regime'] + 1e-6)).clip(-10, 10)
    data['Hurst']      = data['Close'].rolling(50).apply(hurst).clip(0, 1)
    data['Entropy']    = rolling_entropy(data['Return']).clip(0, 5)

    data = data.bfill().ffill()
    data.replace([np.inf, -np.inf], 0, inplace=True)

    log.info(f"  1Y | {ticker} — fetching sentiment (batched, last 6 months)...")
    sentiment_dict = fetch_sentiment_for_ticker(ticker, data.index, api_key)
    data['Sentiment'] = data.index.strftime("%Y-%m-%d").map(sentiment_dict).fillna(0)

    return data

def create_gru_sequences(data, sequence_length=60):
    feature_cols = ['Ret_1','Ret_3','Ret_5','ZScore','Trend','Vol_Regime']
    d = data[feature_cols + ['Return']].copy()
    d.replace([np.inf, -np.inf], 0, inplace=True)
    d.fillna(0, inplace=True)

    scaler = RobustScaler()
    scaled = np.clip(scaler.fit_transform(d[feature_cols]), -5, 5)
    returns = d['Return'].values
    X, y = [], []
    for i in range(sequence_length, len(d) - 5):
        future_return = np.sum(returns[i:i+5])
        if abs(future_return) < 0.003:
            continue
        X.append(scaled[i-sequence_length:i])
        y.append(1 if future_return > 0 else 0)
    return np.array(X), np.array(y), scaler

def build_gru_model(sequence_length=60, n_features=6):
    model = Sequential([
        Input(shape=(sequence_length, n_features)),
        GRU(32, return_sequences=True),
        Dropout(0.2),
        GRU(16),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def predict_multi_day(model, last_sequence, days=5):
    seq = last_sequence.copy()
    preds = []
    for _ in range(days):
        pred = model.predict(seq, verbose=0)[0][0]
        preds.append(pred)
        new_step = seq[0, -1, :].copy()
        new_step[0] = pred
        seq = np.append(seq[:, 1:, :], [[new_step]], axis=1)
    return float(np.mean(preds))

def train_1year():
    log.info("=== 1-YEAR TRAINING STARTED ===")

    FINNHUB_API_KEY = "d738i9pr01qjjol24jjgd738i9pr01qjjol24jk0"
    SEQUENCE_LENGTH = 60

    results_1y = {}

    for ticker in TICKERS:
        log.info(f"  1Y | Training {ticker}...")
        try:
            df = load_1year_stock_data(ticker, FINNHUB_API_KEY)
            X, y, scaler = create_gru_sequences(df, SEQUENCE_LENGTH)

            if len(X) < 100:
                log.warning(f"  1Y | {ticker} — not enough sequences, skipping")
                continue

            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            model = build_gru_model(SEQUENCE_LENGTH, X.shape[2])
            model.fit(
                X_train, y_train,
                epochs=20, batch_size=32,
                validation_data=(X_test, y_test),
                callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)],
                verbose=0
            )

            # Predict direction + estimate annual return from signal strength
            returns_full = df['Return'].values
            capital = 1000.0
            returns_list = []
            last_signal = 0

            for i in range(len(X_test)):
                avg_pred = predict_multi_day(model, X_test[i:i+1])
                if avg_pred > 0.55:   signal = 1
                elif avg_pred < 0.45: signal = -1
                else:                 signal = last_signal
                last_signal = signal

                actual_return = returns_full[split + i]
                cost = 0.001
                if signal != 0:
                    strategy_return = signal * actual_return - cost
                    capital *= (1 + strategy_return)
                    returns_list.append(strategy_return)

            returns_array = np.array(returns_list) if returns_list else np.array([0.0])

            # Annualised stats
            n_trading_days = len(returns_list) if returns_list else 252
            total_return   = (capital / 1000.0) - 1
            ann_return     = (1 + total_return) ** (252 / max(n_trading_days, 1)) - 1
            sharpe         = ((returns_array.mean() / (returns_array.std() + 1e-9)) * np.sqrt(252))
            hist_vol_ann   = float(df['Return'].std() * np.sqrt(252))
            risk_label     = 'Low' if hist_vol_ann < 0.15 else ('Medium' if hist_vol_ann < 0.30 else 'High')
            current_price  = float(df['Close'].iloc[-1])

            # Sentiment summary (last 30 days average)
            recent_sentiment = float(df['Sentiment'].tail(30).mean())

            results_1y[ticker] = {
                'cumulative_return_pct': round(total_return * 100, 2),
                'annualized_pct':        round(ann_return * 100, 2),
                'sharpe':                round(float(sharpe), 2),
                'volatility_pct':        round(hist_vol_ann * 100, 2),
                'risk':                  risk_label,
                'current_price':         round(current_price, 2),
                'sentiment':             round(recent_sentiment, 4),
                'score': (0.50 * total_return * 100 + 0.30 * float(sharpe) * 10 + 0.20 * (1 / (hist_vol_ann + 1)))
            }
            log.info(f"  1Y | {ticker} done — annual return: {ann_return*100:.2f}%  sharpe: {sharpe:.2f}")

        except Exception as e:
            log.error(f"  1Y | {ticker} failed: {e}")
            continue

    log.info("=== 1-YEAR TRAINING COMPLETE ===")
    return results_1y

# ─────────────────────────────────────────────
# 5-YEAR MODEL  (Transformer + CAGR)
# ─────────────────────────────────────────────
def train_5year():
    log.info("=== 5-YEAR TRAINING STARTED ===")

    stocks_list = TICKERS
    SEQ_LEN = 120
    YEARS   = 5

    raw    = yf.download(stocks_list, start="2015-01-01", auto_adjust=True)
    raw    = raw.dropna()
    close  = raw["Close"][stocks_list]
    volume = raw["Volume"][stocks_list]

    def compute_rsi(s, p=14):
        d = s.diff(); g = d.clip(lower=0).rolling(p).mean()
        l = (-d.clip(upper=0)).rolling(p).mean()
        return 100 - (100 / (1 + g / (l + 1e-9)))

    def compute_macd_hist(s, fast=12, slow=26, sig=9):
        ef = s.ewm(span=fast, adjust=False).mean()
        es = s.ewm(span=slow, adjust=False).mean()
        ml = ef - es
        return ml - ml.ewm(span=sig, adjust=False).mean()

    def compute_bb_width(s, w=20):
        ma = s.rolling(w).mean(); std = s.rolling(w).std()
        return (2 * std) / (ma + 1e-9)

    def make_features(df, vol_df):
        parts = []
        lr = np.log(df / df.shift(1))
        parts.append(lr.add_suffix("_lr"))
        for w in [20, 50]:
            parts.append((df / df.rolling(w).mean() - 1).add_suffix(f"_ma{w}"))
        parts.append(pd.DataFrame({f"{c}_rsi14": compute_rsi(df[c]) for c in df.columns}, index=df.index) / 100)
        parts.append(pd.DataFrame({f"{c}_macd":  compute_macd_hist(df[c]) for c in df.columns}, index=df.index))
        parts.append(pd.DataFrame({f"{c}_bb":    compute_bb_width(df[c]) for c in df.columns}, index=df.index))
        parts.append(pd.DataFrame({f"{c}_vol10": np.log(df[c]/df[c].shift(1)).rolling(10).std() for c in df.columns}, index=df.index))
        for w in [10, 30]:
            parts.append((df / df.shift(w) - 1).add_suffix(f"_mom{w}"))
        parts.append((vol_df / vol_df.rolling(20).mean()).add_suffix("_volr"))
        return pd.concat(parts, axis=1)

    feat_df = make_features(close, volume).dropna()
    close   = close.loc[feat_df.index]

    scaler     = StandardScaler()
    feat_sc    = scaler.fit_transform(feat_df)
    log_ret    = np.log(close / close.shift(1)).dropna()
    close      = close.loc[log_ret.index]
    feat_sc    = feat_sc[1:]

    X, y = [], []
    for i in range(SEQ_LEN, len(feat_sc)):
        X.append(feat_sc[i-SEQ_LEN:i])
        y.append(log_ret.values[i])
    X = np.array(X); y = np.array(y)

    inp = Input(shape=(SEQ_LEN, X.shape[2]))
    x   = MultiHeadAttention(num_heads=4, key_dim=32)(inp, inp)
    x   = LayerNormalization()(x + inp)
    x   = Dense(128, activation="relu")(x)
    x   = Dropout(0.2)(x)
    x   = GlobalAveragePooling1D()(x)
    out = Dense(len(stocks_list))(x)
    model = Model(inp, out)
    model.compile(optimizer=Adam(5e-4), loss="mse")

    split = int(len(X) * 0.85)
    model.fit(X[:split], y[:split],
              validation_data=(X[split:], y[split:]),
              epochs=100, batch_size=32,
              callbacks=[EarlyStopping(patience=15, restore_best_weights=True)],
              verbose=0)

    years_data   = (close.index[-1] - close.index[0]).days / 365
    cagr         = (close.iloc[-1] / close.iloc[0]) ** (1/years_data) - 1
    last_prices  = close.iloc[-1].values
    pred         = model.predict(X[split:][-1].reshape(1, SEQ_LEN, -1), verbose=0)[0]
    annual_return = 0.6 * cagr.values + 0.4 * pred
    future_prices = last_prices * (1 + annual_return) ** YEARS

    results_5y = {}
    for i, ticker in enumerate(stocks_list):
        cp  = float(last_prices[i])
        fp  = float(future_prices[i])
        ar  = float(annual_return[i])
        cr  = float(cagr.values[i])
        dlp = float(pred[i])

        results_5y[ticker] = {
            'current_price':     round(cp, 2),
            'future_price':      round(fp, 2),
            'annual_return_pct': round(ar * 100, 2),
            'cagr_pct':          round(cr * 100, 2),
            'dl_pred_pct':       round(dlp * 100, 4),
            'total_return_pct':  round((fp/cp - 1) * 100, 2),
        }
        log.info(f"  5Y | {ticker}: annual={ar*100:.2f}%  5Y={((fp/cp-1)*100):.2f}%")

    log.info("=== 5-YEAR TRAINING COMPLETE ===")
    return results_5y

# ─────────────────────────────────────────────
# MAIN TRAINING JOB
# ─────────────────────────────────────────────
def run_training():
    log.info("╔══════════════════════════════════╗")
    log.info("║  MIDNIGHT TRAINING JOB STARTED   ║")
    log.info("╚══════════════════════════════════╝")
    start = time.time()

    try:
        data_3m = train_3month()
    except Exception as e:
        log.error(f"3M training failed: {e}")
        data_3m = {}

    try:
        data_1y = train_1year()
    except Exception as e:
        log.error(f"1Y training failed: {e}")
        data_1y = {}

    try:
        data_5y = train_5year()
    except Exception as e:
        log.error(f"5Y training failed: {e}")
        data_5y = {}

    cache = {
        "trained_at": datetime.now().isoformat(),
        "tickers":    TICKERS,
        "3month":     data_3m,
        "1year":      data_1y,
        "5year":      data_5y,
    }

    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)

    elapsed = round(time.time() - start, 1)
    log.info(f"✅ Training complete in {elapsed}s — cache saved to {CACHE_FILE}")

# ─────────────────────────────────────────────
# SCHEDULER
# ─────────────────────────────────────────────
if __name__ == "__main__":
    log.info("Scheduler started. Will train at midnight every day.")
    log.info("Running initial training now...")
    run_training()

    schedule.every().day.at("00:00").do(run_training)

    while True:
        schedule.run_pending()
        time.sleep(60)
