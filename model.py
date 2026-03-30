from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense, Input


def build_lstm_model(sequence_length=60, n_features=6):
    model = Sequential([
        Input(shape=(sequence_length, n_features)),
        GRU(32, return_sequences=True),
        Dropout(0.2),
        GRU(16),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
