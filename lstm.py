import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense


def load_and_normalize(country, feature_col):
    path = f"data/{country.replace(' ', '_').lower()}_tempered.csv"
    df = pd.read_csv(path)
    df['year_month'] = pd.to_datetime(df['year_month'])
    series = df[feature_col].values.reshape(-1,1)
    scaler = MinMaxScaler()
    norm = scaler.fit_transform(series)
    return df, norm, scaler


def create_sequences(data, window_size=12):
    x, y = [], []
    for i in range(window_size, len(data)):
        x.append(data[i - window_size:i])
        y.append(data[i])
    return np.array(x), np.array(y)


def split_data(x, y, train_ratio=0.8):
    split = int(len(x) * train_ratio)
    return x[:split], x[split:], y[:split], y[split:]


def build_lstm_model(window_size, units=64):
    model = Sequential([
        LSTM(units, activation='relu', input_shape=(window_size, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def train_model(model, x_train, y_train, x_val, y_val,
                epochs=5, batch_size=64, verbose=1):
    return model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose
    )


def predict_inverse(model, x_test, scaler):
    y_pred = model.predict(x_test)
    y_real_pred = scaler.inverse_transform(y_pred)
    return y_real_pred


def plot_results(dates, y_real_test, y_real_pred, country, mse, mae):
    plt.figure(figsize=(12, 5))
    plt.plot(dates, y_real_test, label='Actual', color='C0')
    plt.plot(dates, y_real_pred, label='Predicted', linestyle='--', color='C1')
    plt.title(f"{country} Surface Temp\n MSE: {mse:.4f} | MAE: {mae:.4f}")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()


def run_pipeline(country_list,
                 feature_col='Average surface temperature daily',
                 window_size=12,
                 train_ratio=0.8,
                 epochs=5,
                 batch_size=64):
    for country in country_list:
        df, norm_data, scaler = load_and_normalize(country, feature_col)

        x, y = create_sequences(norm_data, window_size)

        x_train, x_test, y_train, y_test = split_data(x, y, train_ratio)

        model = build_lstm_model(window_size)
        train_model(model, x_train, y_train, x_test, y_test,
                    epochs=epochs, batch_size=batch_size, verbose=0)

        y_real_pred = predict_inverse(model, x_test, scaler)
        y_real_test = scaler.inverse_transform(y_test.reshape(-1, 1))

        mse = mean_squared_error(y_real_test, y_real_pred)
        mae = mean_absolute_error(y_real_test, y_real_pred)

        dates = df['year_month'].iloc[-len(y_real_test):]
        plot_results(dates, y_real_test, y_real_pred, country, mse, mae)


if __name__ == "__main__":
    countries = ['Chad', 'Faroe Islands', 'Jamaica']
    run_pipeline(countries,
                 window_size=12,
                 train_ratio=0.8,
                 epochs=175,
                 batch_size=16)
