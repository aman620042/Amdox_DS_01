# Amdox_DS_01
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet

st.set_page_config(page_title="Crypto Time Series Dashboard", layout="wide")

# ======================
# Sidebar
# ======================
st.sidebar.title("Crypto Dashboard")
coin = st.sidebar.selectbox("Select Cryptocurrency", ["BTC-USD", "ETH-USD"])
period = st.sidebar.selectbox("Select Time Period", ["6mo", "1y", "2y", "5y"])

# ======================
# Load Data
# ======================
df = yf.download(coin, period=period)
df.reset_index(inplace=True)

# 🔥 FIX FOR yfinance MULTIINDEX
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# ======================
# Calculations
# ======================
df["Daily Return"] = df["Close"].pct_change()
df["MA7"] = df["Close"].rolling(7).mean()
df["MA30"] = df["Close"].rolling(30).mean()
df["Volatility"] = df["Daily Return"].rolling(7).std()

# ======================
# Technical Indicators FIX
# ======================
close_series = df["Close"].astype(float).squeeze()

df["RSI"] = ta.momentum.RSIIndicator(close_series).rsi()

macd = ta.trend.MACD(close_series)
df["MACD"] = macd.macd()
df["Signal"] = macd.macd_signal()

bb = ta.volatility.BollingerBands(close_series)
df["BB_High"] = bb.bollinger_hband()
df["BB_Low"] = bb.bollinger_lband()

# ======================
# Title
# ======================
st.title("🚀 Cryptocurrency Time Series Analysis – 24 Dashboards")

# ======================
# Metrics FIX
# ======================
current_price = float(df["Close"].iloc[-1])
high_price = float(df["High"].max())
low_price = float(df["Low"].min())

c1, c2, c3 = st.columns(3)
c1.metric("Current Price", f"${current_price:.2f}")
c2.metric("Highest Price", f"${high_price:.2f}")
c3.metric("Lowest Price", f"${low_price:.2f}")

# ======================
# Dashboards
# ======================

st.subheader("1️⃣ Price Trend")
st.plotly_chart(px.line(df, x="Date", y="Close"), use_container_width=True)

st.subheader("2️⃣ Trading Volume")
st.plotly_chart(px.bar(df, x="Date", y="Volume"), use_container_width=True)

st.subheader("3️⃣ Moving Averages")
st.plotly_chart(px.line(df, x="Date", y=["Close", "MA7", "MA30"]),
                use_container_width=True)

st.subheader("4️⃣ Daily Returns")
st.plotly_chart(px.line(df, x="Date", y="Daily Return"),
                use_container_width=True)

st.subheader("5️⃣ Volatility")
st.plotly_chart(px.line(df, x="Date", y="Volatility"),
                use_container_width=True)

st.subheader("6️⃣ RSI Indicator")
st.plotly_chart(px.line(df, x="Date", y="RSI"),
                use_container_width=True)

st.subheader("7️⃣ MACD Indicator")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], name="MACD"))
fig.add_trace(go.Scatter(x=df["Date"], y=df["Signal"], name="Signal"))
st.plotly_chart(fig, use_container_width=True)

st.subheader("8️⃣ Bollinger Bands")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Close"))
fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_High"], name="Upper"))
fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_Low"], name="Lower"))
st.plotly_chart(fig, use_container_width=True)

st.subheader("9️⃣ Price Distribution")
st.plotly_chart(px.histogram(df, x="Close", nbins=40),
                use_container_width=True)

df["Month"] = df["Date"].dt.month
st.subheader("🔟 Monthly Average")
st.plotly_chart(px.bar(df.groupby("Month")["Close"].mean()),
                use_container_width=True)

df["Year"] = df["Date"].dt.year
st.subheader("1️⃣1️⃣ Yearly Average")
st.plotly_chart(px.bar(df.groupby("Year")["Close"].mean()),
                use_container_width=True)

df["Cumulative"] = (1 + df["Daily Return"]).cumprod()
st.subheader("1️⃣2️⃣ Cumulative Returns")
st.plotly_chart(px.line(df, x="Date", y="Cumulative"),
                use_container_width=True)

rolling_max = df["Close"].cummax()
drawdown = (df["Close"] - rolling_max) / rolling_max
st.subheader("1️⃣3️⃣ Drawdown Analysis")
st.plotly_chart(px.line(x=df["Date"], y=drawdown),
                use_container_width=True)

st.subheader("1️⃣4️⃣ ARIMA Forecast")
ts = df.set_index("Date")["Close"].dropna()
model = ARIMA(ts, order=(5, 1, 0))
fit = model.fit()
forecast = fit.forecast(30)
st.line_chart(pd.concat([ts, forecast]))

st.subheader("1️⃣5️⃣ Actual vs Forecast")
st.plotly_chart(px.line(pd.DataFrame({
    "Actual": ts[-30:],
    "Forecast": forecast[:30]
})), use_container_width=True)

st.subheader("1️⃣6️⃣ LSTM Forecast")
# Preprocess data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

seq_length = 60
if len(scaled_data) > seq_length:
    X, y = create_sequences(scaled_data, seq_length)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train model
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)  # Reduced epochs for speed

    # Forecast next 30 days
    last_sequence = scaled_data[-seq_length:]
    predictions = []
    for _ in range(30):
        pred = model.predict(last_sequence.reshape(1, seq_length, 1), verbose=0)
        predictions.append(pred[0][0])
        last_sequence = np.append(last_sequence[1:], pred[0][0])

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    forecast_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=30, freq='D')
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'LSTM Forecast': predictions})

    # Plot
    combined_df = pd.concat([df[['Date', 'Close']].rename(columns={'Close': 'Historical'}), forecast_df])
    st.line_chart(combined_df.set_index('Date'))
else:
    st.write("Not enough data for LSTM forecasting.")

st.subheader("1️⃣7️⃣ SARIMA Forecast")
# SARIMA model with seasonal parameters
try:
    sarima_model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))  # Assuming monthly seasonality
    sarima_fit = sarima_model.fit(disp=False)
    sarima_forecast = sarima_fit.forecast(30)
    sarima_forecast_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=30, freq='D')
    sarima_forecast_df = pd.DataFrame({'Date': sarima_forecast_dates, 'SARIMA Forecast': sarima_forecast})

    # Plot
    combined_sarima_df = pd.concat([df[['Date', 'Close']].rename(columns={'Close': 'Historical'}), sarima_forecast_df])
    st.line_chart(combined_sarima_df.set_index('Date'))
except Exception as e:
    st.write(f"Error in SARIMA forecasting: {e}")

st.subheader("1️⃣8️⃣ Sharpe Ratio")
# Calculate Sharpe Ratio (assuming risk-free rate = 0)
mean_return = df["Daily Return"].mean()
std_return = df["Daily Return"].std()
sharpe_ratio = mean_return / std_return if std_return != 0 else 0
st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

st.subheader("1️⃣9️⃣ Correlation Matrix")
# Fetch data for multiple cryptos
cryptos = ['BTC-USD', 'ETH-USD', 'ADA-USD', 'BNB-USD']
corr_data = {}
for crypto in cryptos:
    try:
        crypto_df = yf.download(crypto, period=period)
        if isinstance(crypto_df.columns, pd.MultiIndex):
            crypto_df.columns = crypto_df.columns.get_level_values(0)
        corr_data[crypto] = crypto_df['Close']
    except Exception as e:
        st.write(f"Error fetching data for {crypto}: {e}")

if corr_data:
    try:
        corr_df = pd.DataFrame(corr_data).dropna()
        if not corr_df.empty and len(corr_df.columns) > 1:
            corr_matrix = corr_df.corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        else:
            st.write("Insufficient data for correlation matrix.")
    except Exception as e:
        st.write(f"Error computing correlation matrix: {e}")
else:
    st.write("Unable to fetch correlation data.")

st.subheader("2️⃣0️⃣ Candlestick Chart")
# Candlestick chart using OHLC data
fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])
fig.update_layout(title="Candlestick Chart", xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

st.subheader("2️⃣1️⃣ Maximum Drawdown")
# Calculate Maximum Drawdown
rolling_max = df["Close"].cummax()
drawdown = (df["Close"] - rolling_max) / rolling_max
max_drawdown = drawdown.min()
st.metric("Maximum Drawdown", f"{max_drawdown:.2%}")
st.plotly_chart(px.line(x=df["Date"], y=drawdown, title="Drawdown Over Time"), use_container_width=True)

st.subheader("2️⃣2️⃣ Prophet Forecast")
# Prepare data for Prophet
prophet_df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

# Fit Prophet model
prophet_model = Prophet()
prophet_model.fit(prophet_df)

# Forecast next 30 days
future = prophet_model.make_future_dataframe(periods=30)
prophet_forecast = prophet_model.predict(future)

# Plot forecast
fig = go.Figure()
fig.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], name='Historical'))
fig.add_trace(go.Scatter(x=prophet_forecast['ds'], y=prophet_forecast['yhat'], name='Forecast'))
fig.add_trace(go.Scatter(x=prophet_forecast['ds'], y=prophet_forecast['yhat_lower'], fill='tonexty', mode='lines', name='Lower Bound', line=dict(color='lightblue')))
fig.add_trace(go.Scatter(x=prophet_forecast['ds'], y=prophet_forecast['yhat_upper'], fill='tonexty', mode='lines', name='Upper Bound', line=dict(color='lightblue')))
fig.update_layout(title="Prophet Forecast", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig, use_container_width=True)

st.subheader("2️⃣3️⃣ Raw Data")
st.dataframe(df.tail(25))

st.subheader("2️⃣4️⃣ Forecast Comparison")
# Overlay ARIMA, SARIMA, LSTM, and Prophet forecasts
fig = go.Figure()

# Historical data
fig.add_trace(go.Scatter(x=ts.index, y=ts.values, name='Historical', mode='lines'))

# ARIMA forecast (assuming forecast is already computed earlier)
arima_forecast_dates = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')
fig.add_trace(go.Scatter(x=arima_forecast_dates, y=forecast.values, name='ARIMA Forecast', mode='lines'))

# SARIMA forecast (assuming sarima_forecast is computed)
if 'sarima_forecast' in locals():
    fig.add_trace(go.Scatter(x=sarima_forecast_dates, y=sarima_forecast.values, name='SARIMA Forecast', mode='lines'))

# LSTM forecast (assuming predictions is computed)
if 'predictions' in locals():
    fig.add_trace(go.Scatter(x=forecast_dates, y=predictions, name='LSTM Forecast', mode='lines'))

# Prophet forecast
fig.add_trace(go.Scatter(x=prophet_forecast['ds'], y=prophet_forecast['yhat'], name='Prophet Forecast', mode='lines'))

fig.update_layout(title="Forecast Comparison", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig, use_container_width=True)
