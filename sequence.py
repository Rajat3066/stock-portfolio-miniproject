from sklearn.preprocessing import RobustScaler
import numpy as np


def create_sequences(data, sequence_length=60):
    feature_cols = [
        'Ret_1',
        'Ret_3',
        'Ret_5',
        'ZScore',
        'Trend',
        'Vol_Regime'
    ]

    data = data[feature_cols + ['Return']].copy()
    data = data.replace([np.inf, -np.inf], 0)
    data = data.fillna(0)

    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(data[feature_cols])
    scaled_features = np.clip(scaled_features, -5, 5)

    returns = data['Return'].values
    X, y = [], []

    for i in range(sequence_length, len(data) - 5):
        future_return = np.sum(returns[i:i + 5])

        if abs(future_return) < 0.003:
            continue

        X.append(scaled_features[i - sequence_length:i])
        y.append(1 if future_return > 0 else 0)

    return np.array(X), np.array(y)
