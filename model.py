import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

import keras_tuner as kt

import plotly.graph_objects as go
from plotly.subplots import make_subplots

class StockPredictor:
    def __init__(self):
        self.ticker = None
        self.df = None
        self.scaler = None
        self.model = None
        self.prediction_days = 7  # predict next 7 days
        self.future_predictions = None
        self.future_dates = None

    # Fetch historical data
    def get_user_input(self, ticker):
        self.ticker = ticker
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=7*365)
        return start_date, end_date

    def fetch_live_data(self, start_date, end_date):
        self.df = yf.download(self.ticker, start=start_date, end=end_date)
        if self.df.empty:
            raise ValueError("No data found for the given ticker and date range.")
        self.df['Pct_Change'] = self.df['Close'].pct_change() * 100
        self.df.ffill(inplace=True)
        return self.df

    # Add technical indicators
    def add_technical_indicators(self):
        self.df['MA_50'] = self.df['Close'].rolling(window=50).mean()
        self.df['MA_200'] = self.df['Close'].rolling(window=200).mean()
        self.df.ffill(inplace=True)
        return self.df

    # Prepare data for LSTM
    def prepare_data(self, look_back=60):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(self.df[['Close']].values)
        X, y = [], []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i - look_back:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        split = int(0.9 * len(X))
        return X[:split], X[split:], y[:split], y[split:]

    # Build LSTM model with Keras Tuner
    def build_model_tuner(self, hp):
        model = Sequential()
        units_1 = hp.Int('units_1', min_value=32, max_value=128, step=32)
        model.add(Bidirectional(LSTM(units_1, return_sequences=True), input_shape=(60,1)))
        model.add(Dropout(hp.Float('dropout_1', 0.2, 0.5, step=0.1)))

        units_2 = hp.Int('units_2', min_value=32, max_value=128, step=32)
        model.add(Bidirectional(LSTM(units_2)))
        model.add(Dropout(hp.Float('dropout_2', 0.2, 0.5, step=0.1)))

        model.add(Dense(hp.Int('dense_units', 16, 64, step=16), activation='relu'))
        model.add(Dense(1))

        lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
        return model

    # Train LSTM model
    def train_model(self, X_train, y_train):
        tuner = kt.RandomSearch(
            self.build_model_tuner,
            objective='val_loss',
            max_trials=5,
            executions_per_trial=1,
            directory='stock_tuner',
            project_name='stock_prediction'
        )
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
        tuner.search(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, callbacks=[early_stop], verbose=1)

        self.model = tuner.get_best_models(num_models=1)[0]
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("\nBest Hyperparameters Found:")
        print(f"LSTM Layer 1 Units: {best_hp.get('units_1')}")
        print(f"Dropout 1: {best_hp.get('dropout_1')}")
        print(f"LSTM Layer 2 Units: {best_hp.get('units_2')}")
        print(f"Dropout 2: {best_hp.get('dropout_2')}")
        print(f"Dense Units: {best_hp.get('dense_units')}")
        print(f"Learning Rate: {best_hp.get('learning_rate')}")

    # Evaluate model
    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_actual = self.scaler.inverse_transform(y_pred)
        return y_test_actual, y_pred_actual

    # Print performance metrics
    def print_metrics(self, y_test_actual, y_pred_actual):
        mse = mean_squared_error(y_test_actual, y_pred_actual)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_actual, y_pred_actual)
        r2 = r2_score(y_test_actual, y_pred_actual)
        mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100
        print(f"\nModel Performance on Test Data:")
        print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}, MAPE: {mape:.2f}%")

    # Predict next 7 days
    def predict_future(self, X_last):
        current_sequence = X_last.flatten().copy()
        last_date = self.df.index[-1]
        self.future_dates = pd.date_range(start=last_date + dt.timedelta(days=1), periods=self.prediction_days)
        future_df = pd.DataFrame(index=self.future_dates, columns=['Open', 'High', 'Low', 'Close'])
        last_close = self.df['Close'].iloc[-1]

        for i in range(self.prediction_days):
            next_close = self.model.predict(current_sequence.reshape(1, -1, 1))[0, 0]
            next_close = self.scaler.inverse_transform(np.array([[next_close]]))[0, 0]
            open_price = last_close * (1 + np.random.uniform(-0.005, 0.005)) if i == 0 else future_df['Close'].iloc[i-1] * (1 + np.random.uniform(-0.005, 0.005))
            high = next_close * (1 + np.random.uniform(0, 0.01))
            low = next_close * (1 - np.random.uniform(0, 0.01))
            future_df.iloc[i] = [open_price, high, low, next_close]
            next_scaled = self.scaler.transform(np.array([[next_close]]))[0, 0]
            current_sequence = np.append(current_sequence[1:], next_scaled)

        self.future_predictions = future_df
        return future_df

    # Visualize historical + predicted data
    def visualize_results(self, y_test_actual, y_pred_actual):
        last_date = self.df.index[-1]
        test_dates = pd.date_range(start=self.df.index[-len(y_test_actual)], end=last_date)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.1,
                            subplot_titles=('Historical Data', 'Predictions'))

        # Historical Candlestick + MA
        fig.add_trace(go.Candlestick(x=self.df.index[-100:], open=self.df['Open'][-100:], high=self.df['High'][-100:], 
                                     low=self.df['Low'][-100:], close=self.df['Close'][-100:], name='Price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.df.index[-100:], y=self.df['MA_50'][-100:], line=dict(color='blue', width=1), name='50-day MA'), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.df.index[-100:], y=self.df['MA_200'][-100:], line=dict(color='red', width=1), name='200-day MA'), row=1, col=1)
        fig.add_trace(go.Bar(x=self.df.index[-100:], y=self.df['Volume'][-100:], marker_color='rgba(100,100,100,0.5)', name='Volume', yaxis='y2'), row=1, col=1)

        # Predictions
        fig.add_trace(go.Scatter(x=test_dates, y=y_test_actual.flatten(), line=dict(color='blue', width=2), name='Actual Price'), row=2, col=1)
        fig.add_trace(go.Scatter(x=test_dates, y=y_pred_actual.flatten(), line=dict(color='red', width=2, dash='dot'), name='Predicted Price'), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.future_dates, y=self.future_predictions['Close'], line=dict(color='green', width=2, dash='dot'), name='7-Day Forecast'), row=2, col=1)

        fig.update_layout(title=f'{self.ticker} Stock Analysis & 7-Day Prediction', height=850, xaxis_rangeslider_visible=False,
                          showlegend=True, yaxis2=dict(overlaying='y', side='right', showgrid=False))
        fig.show()

    # Run everything
    def run(self, ticker="AAPL"):
        try:
            start_date, end_date = self.get_user_input(ticker)
            self.fetch_live_data(start_date, end_date)
            self.add_technical_indicators()
            X_train, X_test, y_train, y_test = self.prepare_data()
            self.train_model(X_train, y_train)
            y_test_actual, y_pred_actual = self.evaluate_model(X_test, y_test)
            self.print_metrics(y_test_actual, y_pred_actual)
            future_predictions = self.predict_future(X_test[-1])
            self.visualize_results(y_test_actual, y_pred_actual)
            print("\nPredicted Close Prices for Next 7 Days:")
            print(future_predictions['Close'])
        except Exception as e:
            print(f"Error: {e}\nPlease check your inputs and try again.")

# Example usage:
# predictor = StockPredictor()
# predictor.run("AAPL")
if __name__ == "__main__":
    predictor = StockPredictor()
    predictor.run("AAPL")  # You can change ticker here