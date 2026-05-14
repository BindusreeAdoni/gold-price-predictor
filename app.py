import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import timedelta

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gold Price Predictor",
    page_icon="🥇",
    layout="wide"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0f;
    color: #e8e0d0;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0f0e1a 50%, #0a0f0a 100%);
    min-height: 100vh;
}

/* Header */
.main-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
    border-bottom: 1px solid rgba(212, 175, 55, 0.2);
    margin-bottom: 2rem;
}
.main-header h1 {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #d4af37, #f5e17a, #d4af37);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 2px;
    margin: 0;
}
.main-header p {
    color: #8a8070;
    font-size: 0.95rem;
    font-weight: 300;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 0.5rem;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 2rem;
    flex-wrap: wrap;
}
.metric-card {
    background: rgba(212, 175, 55, 0.05);
    border: 1px solid rgba(212, 175, 55, 0.2);
    border-radius: 12px;
    padding: 1.2rem 1.8rem;
    flex: 1;
    min-width: 140px;
    text-align: center;
}
.metric-card .label {
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #8a8070;
    margin-bottom: 0.4rem;
}
.metric-card .value {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    color: #d4af37;
    font-weight: 700;
}
.metric-card .sub {
    font-size: 0.75rem;
    color: #5a5550;
    margin-top: 0.2rem;
}

/* Section headers */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    color: #d4af37;
    border-left: 3px solid #d4af37;
    padding-left: 1rem;
    margin: 2rem 0 1rem 0;
}

/* Forecast card */
.forecast-card {
    background: rgba(212, 175, 55, 0.04);
    border: 1px solid rgba(212, 175, 55, 0.15);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Footer */
.footer {
    text-align: center;
    padding: 2rem 0;
    color: #3a3530;
    font-size: 0.8rem;
    letter-spacing: 1px;
    border-top: 1px solid rgba(212, 175, 55, 0.08);
    margin-top: 3rem;
}

/* Streamlit overrides */
div[data-testid="stSlider"] > div {
    color: #d4af37 !important;
}
.stSlider [data-testid="stMarkdownContainer"] p {
    color: #d4af37;
}
div.stButton > button {
    background: linear-gradient(135deg, #d4af37, #b8960c);
    color: #0a0a0f;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 2rem;
    font-family: 'DM Sans', sans-serif;
    letter-spacing: 1px;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #f5e17a, #d4af37);
}
</style>
""", unsafe_allow_html=True)

# ── LOAD ASSETS ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model    = load_model('cnn_bilstm_gold.h5')
    scaler_X = joblib.load('scaler_X.pkl')
    scaler_y = joblib.load('scaler_y.pkl')
    df       = pd.read_csv('GoldUSD_processed.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return model, scaler_X, scaler_y, df

model, scaler_X, scaler_y, df = load_assets()

SEQ_LEN  = 60
FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA7', 'MA30']

# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🥇 GOLD PRICE PREDICTOR</h1>
    <p>CNN + Bidirectional LSTM · Deep Learning Forecast</p>
</div>
""", unsafe_allow_html=True)

# ── METRIC CARDS ─────────────────────────────────────────────────────────────
latest      = df['Close'].iloc[-1]
prev        = df['Close'].iloc[-2]
change      = latest - prev
change_pct  = (change / prev) * 100
year_ago    = df['Close'].iloc[-252] if len(df) > 252 else df['Close'].iloc[0]
ytd_return  = ((latest - year_ago) / year_ago) * 100
all_time_high = df['High'].max()

st.markdown(f"""
<div class="metric-row">
    <div class="metric-card">
        <div class="label">Latest Close</div>
        <div class="value">${latest:,.2f}</div>
        <div class="sub">USD per oz</div>
    </div>
    <div class="metric-card">
        <div class="label">Daily Change</div>
        <div class="value" style="color:{'#4caf87' if change>=0 else '#e05555'}">
            {'▲' if change>=0 else '▼'} ${abs(change):.2f}
        </div>
        <div class="sub">{change_pct:+.2f}%</div>
    </div>
    <div class="metric-card">
        <div class="label">1-Year Return</div>
        <div class="value" style="color:{'#4caf87' if ytd_return>=0 else '#e05555'}">
            {ytd_return:+.1f}%
        </div>
        <div class="sub">vs 252 trading days ago</div>
    </div>
    <div class="metric-card">
        <div class="label">All-Time High</div>
        <div class="value">${all_time_high:,.2f}</div>
        <div class="sub">Intraday high</div>
    </div>
    <div class="metric-card">
        <div class="label">Data Range</div>
        <div class="value" style="font-size:1.1rem">{df['Date'].iloc[0].year} – {df['Date'].iloc[-1].year}</div>
        <div class="sub">{len(df):,} trading days</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── SECTION 1: PRICE HISTORY ──────────────────────────────────────────────────
st.markdown('<div class="section-title">Price History</div>', unsafe_allow_html=True)

fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(
    x=df['Date'], y=df['Close'],
    name='Gold Close Price',
    line=dict(color='#d4af37', width=1.5),
    fill='tozeroy',
    fillcolor='rgba(212,175,55,0.06)'
))
fig_hist.add_trace(go.Scatter(
    x=df['Date'], y=df['MA30'],
    name='30-Day MA',
    line=dict(color='#8a6ff0', width=1.2, dash='dot'),
))
fig_hist.update_layout(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#e8e0d0', family='DM Sans'),
    xaxis=dict(showgrid=False, color='#3a3530'),
    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)', title='Price (USD)'),
    legend=dict(bgcolor='rgba(0,0,0,0)'),
    hovermode='x unified',
    height=380,
    margin=dict(l=0, r=0, t=20, b=0)
)
st.plotly_chart(fig_hist, use_container_width=True)

# ── SECTION 2: PREDICTION VS ACTUAL ──────────────────────────────────────────
st.markdown('<div class="section-title">CNN + Bi-LSTM — Predicted vs Actual</div>', unsafe_allow_html=True)

@st.cache_data
def get_test_predictions():
    X_scaled = scaler_X.transform(df[FEATURES].values)
    y_scaled = scaler_y.transform(df[['Close']].values)
    split    = int(len(X_scaled) * 0.8)

    X_test_sc = X_scaled[split:]
    y_test_sc = y_scaled[split:]

    # Build sequences
    Xs, ys = [], []
    for i in range(SEQ_LEN, len(X_test_sc)):
        Xs.append(X_test_sc[i-SEQ_LEN:i])
        ys.append(y_test_sc[i])
    Xs, ys = np.array(Xs), np.array(ys)

    preds   = model.predict(Xs, verbose=0)
    y_pred  = scaler_y.inverse_transform(preds).ravel()
    y_true  = scaler_y.inverse_transform(ys).ravel()
    dates   = df['Date'].values[split + SEQ_LEN:]
    return dates, y_true, y_pred

dates_test, y_true, y_pred = get_test_predictions()

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)
r2   = r2_score(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("RMSE",  f"${rmse:.2f}")
col2.metric("MAE",   f"${mae:.2f}")
col3.metric("R²",    f"{r2:.4f}")
col4.metric("MAPE",  f"{mape:.2f}%")

fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(
    x=dates_test, y=y_true,
    name='Actual Price',
    line=dict(color='#d4af37', width=2)
))
fig_pred.add_trace(go.Scatter(
    x=dates_test, y=y_pred,
    name='CNN + Bi-LSTM Prediction',
    line=dict(color='#8a6ff0', width=1.8, dash='dot')
))
fig_pred.update_layout(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#e8e0d0', family='DM Sans'),
    xaxis=dict(showgrid=False, color='#3a3530'),
    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)', title='Price (USD)'),
    legend=dict(bgcolor='rgba(0,0,0,0)'),
    hovermode='x unified',
    height=400,
    margin=dict(l=0, r=0, t=20, b=0)
)
st.plotly_chart(fig_pred, use_container_width=True)

# ── SECTION 3: FUTURE FORECAST ────────────────────────────────────────────────
st.markdown('<div class="section-title">Future Price Forecast</div>', unsafe_allow_html=True)

st.markdown('<div class="forecast-card">', unsafe_allow_html=True)
n_days = st.select_slider(
    "Select forecast horizon",
    options=[7, 14, 30],
    value=14,
    format_func=lambda x: f"{x} Days"
)

if st.button("🔮 Generate Forecast"):
    with st.spinner("Running CNN + Bi-LSTM forecast..."):

        X_scaled = scaler_X.transform(df[FEATURES].values)
        last_seq = X_scaled[-SEQ_LEN:]  # shape (60, 7)

        future_preds = []
        current_seq  = last_seq.copy()

        for _ in range(n_days):
            inp   = current_seq.reshape(1, SEQ_LEN, len(FEATURES))
            pred  = model.predict(inp, verbose=0)[0][0]
            future_preds.append(pred)

            # Roll: drop oldest, append new row
            # Update Close (index 3) and leave others as last known
            new_row         = current_seq[-1].copy()
            new_row[3]      = pred  # Close (scaled)
            current_seq     = np.vstack([current_seq[1:], new_row])

        # Inverse transform
        dummy    = np.zeros((n_days, len(FEATURES)))
        dummy[:, 3] = future_preds
        future_prices = scaler_y.inverse_transform(
            np.array(future_preds).reshape(-1, 1)
        ).ravel()

        # Generate future dates (skip weekends)
        last_date    = df['Date'].iloc[-1]
        future_dates = []
        d = last_date
        while len(future_dates) < n_days:
            d += timedelta(days=1)
            if d.weekday() < 5:
                future_dates.append(d)

        # Plot: last 90 days actual + forecast
        hist_window  = df.tail(90)
        fig_forecast = go.Figure()

        fig_forecast.add_trace(go.Scatter(
            x=hist_window['Date'], y=hist_window['Close'],
            name='Recent Actual',
            line=dict(color='#d4af37', width=2)
        ))
        fig_forecast.add_trace(go.Scatter(
            x=future_dates, y=future_prices,
            name=f'{n_days}-Day Forecast',
            line=dict(color='#4caf87', width=2.5, dash='dot'),
            mode='lines+markers',
            marker=dict(size=6, color='#4caf87')
        ))
        # Confidence band (±2% as a simple visual band)
        upper = future_prices * 1.02
        lower = future_prices * 0.98
        fig_forecast.add_trace(go.Scatter(
            x=future_dates + future_dates[::-1],
            y=list(upper) + list(lower[::-1]),
            fill='toself',
            fillcolor='rgba(76,175,135,0.08)',
            line=dict(color='rgba(0,0,0,0)'),
            name='±2% Band',
            showlegend=True
        ))
        fig_forecast.add_vline(
            x=str(last_date),
            line=dict(color='rgba(212,175,55,0.4)', dash='dash', width=1.5)
        )
        fig_forecast.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e8e0d0', family='DM Sans'),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)', title='Price (USD)'),
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            hovermode='x unified',
            height=420,
            margin=dict(l=0, r=0, t=20, b=0)
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

        # Forecast table
        forecast_df = pd.DataFrame({
            'Date':            [d.strftime('%d %b %Y') for d in future_dates],
            'Predicted Price': [f"${p:,.2f}" for p in future_prices],
            'Change vs Today': [f"{((p-latest)/latest)*100:+.2f}%" for p in future_prices]
        })
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    CNN + Bidirectional LSTM · Gold Price Intelligence · Not Financial Advice
</div>
""", unsafe_allow_html=True)
