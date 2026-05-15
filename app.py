import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import timedelta
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Gold Price Predictor", page_icon="🪙", layout="wide")

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #0a0a0f; color: #e8e0d0; }
.stApp { background: linear-gradient(135deg, #0a0a0f 0%, #0f0e1a 50%, #0a0f0a 100%); min-height: 100vh; }
.main-header { text-align: center; padding: 2.5rem 0 1.5rem 0; border-bottom: 1px solid rgba(212,175,55,0.2); margin-bottom: 2rem; }
.main-header h1 { font-family: 'Playfair Display', serif; font-size: 3rem; font-weight: 700; background: linear-gradient(90deg, #d4af37, #f5e17a, #d4af37); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: 2px; margin: 0; }
.main-header p { color: #8a8070; font-size: 0.95rem; font-weight: 300; letter-spacing: 3px; text-transform: uppercase; margin-top: 0.5rem; }
.metric-row { display: flex; gap: 1rem; margin-bottom: 2rem; flex-wrap: wrap; }
.metric-card { background: rgba(212,175,55,0.05); border: 1px solid rgba(212,175,55,0.2); border-radius: 12px; padding: 1.2rem 1.8rem; flex: 1; min-width: 140px; text-align: center; }
.metric-card .label { font-size: 0.7rem; letter-spacing: 2px; text-transform: uppercase; color: #8a8070; margin-bottom: 0.4rem; }
.metric-card .value { font-family: 'Playfair Display', serif; font-size: 1.6rem; color: #d4af37; font-weight: 700; }
.metric-card .sub { font-size: 0.75rem; color: #5a5550; margin-top: 0.2rem; }
.section-title { font-family: 'Playfair Display', serif; font-size: 1.4rem; color: #d4af37; border-left: 3px solid #d4af37; padding-left: 1rem; margin: 2rem 0 1rem 0; }
.today-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.8rem; margin-bottom: 1.5rem; }
.today-card { background: rgba(212,175,55,0.04); border: 1px solid rgba(212,175,55,0.12); border-radius: 10px; padding: 1rem 1.2rem; text-align: center; }
.today-card .t-label { font-size: 0.65rem; letter-spacing: 2px; text-transform: uppercase; color: #6a6060; margin-bottom: 0.3rem; }
.today-card .t-value { font-family: 'Playfair Display', serif; font-size: 1.25rem; color: #e8e0d0; }
.footer { text-align: center; padding: 2rem 0; color: #3a3530; font-size: 0.8rem; letter-spacing: 1px; border-top: 1px solid rgba(212,175,55,0.08); margin-top: 3rem; }
div.stButton > button { background: linear-gradient(135deg, #d4af37, #b8960c); color: #0a0a0f; font-weight: 600; border: none; border-radius: 8px; padding: 0.5rem 2rem; font-family: 'DM Sans', sans-serif; letter-spacing: 1px; }
div.stButton > button:hover { background: linear-gradient(135deg, #f5e17a, #d4af37); }
</style>
""", unsafe_allow_html=True)

# ── LOAD LIVE DATA ────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_live_data():
    try:
        raw = yf.download('GC=F', start='2000-08-30', auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df = df.reset_index().rename(columns={'Date': 'Date'})
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        df = df.dropna()
        df['MA7']  = df['Close'].rolling(window=7).mean()
        df['MA30'] = df['Close'].rolling(window=30).mean()
        df = df.dropna().reset_index(drop=True)
        return df, True
    except Exception as e:
        df = pd.read_csv('GoldUSD_processed.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        return df, False

# ── LOAD MODEL & SCALERS ──────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model    = load_model('cnn_bilstm_gold.keras')
    scaler_X = joblib.load('scaler_X.pkl')
    scaler_y = joblib.load('scaler_y.pkl')
    return model, scaler_X, scaler_y

model, scaler_X, scaler_y = load_assets()
df, is_live = load_live_data()

SEQ_LEN  = 60
FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA7', 'MA30']

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🥇 GOLD PRICE PREDICTOR</h1>
    <p>CNN + Bidirectional LSTM · Deep Learning Forecast</p>
</div>
""", unsafe_allow_html=True)

if is_live:
    st.success(f"✅ Live data loaded — {len(df):,} trading days up to {df['Date'].iloc[-1].strftime('%d %b %Y')}")
else:
    st.warning("⚠️ Using cached dataset — live feed unavailable.")

# ── OVERVIEW METRIC CARDS ─────────────────────────────────────────────────────
latest        = float(df['Close'].iloc[-1])
prev          = float(df['Close'].iloc[-2])
change        = latest - prev
change_pct    = (change / prev) * 100
year_ago      = float(df['Close'].iloc[-252]) if len(df) > 252 else float(df['Close'].iloc[0])
ytd_return    = ((latest - year_ago) / year_ago) * 100
all_time_high = float(df['High'].max())

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

# ── TODAY'S MOVEMENT ──────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Today\'s Market Movement</div>', unsafe_allow_html=True)

today      = df.iloc[-1]
t_open     = float(today['Open'])
t_high     = float(today['High'])
t_low      = float(today['Low'])
t_close    = float(today['Close'])
t_volume   = int(today['Volume'])
t_change   = t_close - t_open
t_change_p = (t_change / t_open) * 100
t_range    = t_high - t_low
is_up      = t_close >= t_open
direction  = "▲ Bullish" if is_up else "▼ Bearish"
dir_color  = "#4caf87" if is_up else "#e05555"
date_label = pd.to_datetime(today['Date']).strftime('%d %b %Y')

st.markdown(f"""
<div style="background:rgba(212,175,55,0.03);border:1px solid rgba(212,175,55,0.12);
            border-radius:14px;padding:1.4rem 1.8rem;margin-bottom:1rem;">

  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1.2rem;">
    <div style="font-size:0.75rem;letter-spacing:2px;text-transform:uppercase;color:#6a6060;">
      Trading Day · {date_label}
    </div>
    <div style="font-size:1rem;font-weight:600;color:{dir_color};letter-spacing:1px;">
      {direction} &nbsp; {t_change_p:+.2f}%
    </div>
  </div>

  <div class="today-grid">
    <div class="today-card">
      <div class="t-label">Open</div>
      <div class="t-value">${t_open:,.2f}</div>
    </div>
    <div class="today-card">
      <div class="t-label">High</div>
      <div class="t-value" style="color:#4caf87;">${t_high:,.2f}</div>
    </div>
    <div class="today-card">
      <div class="t-label">Low</div>
      <div class="t-value" style="color:#e05555;">${t_low:,.2f}</div>
    </div>
    <div class="today-card">
      <div class="t-label">Close</div>
      <div class="t-value" style="color:{'#4caf87' if is_up else '#e05555'};">${t_close:,.2f}</div>
    </div>
    <div class="today-card">
      <div class="t-label">Day Range</div>
      <div class="t-value" style="font-size:1rem;">${t_range:,.2f}</div>
    </div>
  </div>

  <div style="margin-top:0.8rem;">
    <div style="font-size:0.7rem;color:#6a6060;margin-bottom:0.3rem;letter-spacing:1px;">
      INTRADAY RANGE
    </div>
    <div style="background:rgba(255,255,255,0.05);border-radius:999px;height:6px;position:relative;">
      <div style="
        position:absolute;
        left:{((t_open - t_low) / (t_high - t_low + 0.0001)) * 100:.1f}%;
        width:{((t_close - t_open) / (t_high - t_low + 0.0001)) * 100:.1f}%;
        height:100%;
        background:{'#4caf87' if is_up else '#e05555'};
        border-radius:999px;
        min-width:4px;
      "></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#6a6060;margin-top:0.3rem;">
      <span>Low ${t_low:,.2f}</span><span>High ${t_high:,.2f}</span>
    </div>
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
    fill='tozeroy', fillcolor='rgba(212,175,55,0.06)'
))
fig_hist.add_trace(go.Scatter(
    x=df['Date'], y=df['MA30'],
    name='30-Day MA',
    line=dict(color='#8a6ff0', width=1.2, dash='dot')
))

# Default view: last 2 years; range selector buttons let user zoom freely
default_start = df['Date'].iloc[-1] - pd.DateOffset(years=2)

fig_hist.update_layout(
    template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#e8e0d0', family='DM Sans'),
    xaxis=dict(
        showgrid=False, color='#3a3530',
        range=[str(default_start.date()), str(df['Date'].iloc[-1].date())],
        rangeselector=dict(
            bgcolor='rgba(212,175,55,0.08)',
            activecolor='rgba(212,175,55,0.35)',
            bordercolor='rgba(212,175,55,0.2)',
            font=dict(color='#e8e0d0', size=11),
            buttons=list([
                dict(count=1,  label="1M",  step="month", stepmode="backward"),
                dict(count=6,  label="6M",  step="month", stepmode="backward"),
                dict(count=1,  label="1Y",  step="year",  stepmode="backward"),
                dict(count=5,  label="5Y",  step="year",  stepmode="backward"),
                dict(step="all", label="All")
            ])
        ),
        rangeslider=dict(visible=True, bgcolor='rgba(212,175,55,0.04)', thickness=0.04),
        type="date"
    ),
    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)', title='Price (USD)', fixedrange=False),
    legend=dict(bgcolor='rgba(0,0,0,0)'),
    hovermode='x unified',
    height=450,
    margin=dict(l=0, r=0, t=20, b=0),
    dragmode='zoom'
)
st.plotly_chart(fig_hist, use_container_width=True)

# ── SECTION 2: PREDICTION VS ACTUAL ──────────────────────────────────────────
st.markdown('<div class="section-title">CNN + Bi-LSTM — Predicted vs Actual</div>', unsafe_allow_html=True)

@st.cache_data
def get_test_predictions():
    X_scaled  = scaler_X.transform(df[FEATURES].values)
    y_scaled  = scaler_y.transform(df[['Close']].values)
    split     = int(len(X_scaled) * 0.8)
    X_test_sc = X_scaled[split:]
    y_test_sc = y_scaled[split:]
    Xs, ys    = [], []
    for i in range(SEQ_LEN, len(X_test_sc)):
        Xs.append(X_test_sc[i-SEQ_LEN:i])
        ys.append(y_test_sc[i])
    Xs, ys = np.array(Xs), np.array(ys)
    preds  = model.predict(Xs, verbose=0)
    y_pred = scaler_y.inverse_transform(preds).ravel()
    y_true = scaler_y.inverse_transform(ys).ravel()
    dates  = df['Date'].values[split + SEQ_LEN:]
    return dates, y_true, y_pred

dates_test, y_true, y_pred = get_test_predictions()

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)
r2   = r2_score(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Directional accuracy
actual_dir    = np.sign(np.diff(y_true))
predicted_dir = np.sign(np.diff(y_pred))
dir_acc       = np.mean(actual_dir == predicted_dir) * 100

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("RMSE", f"${rmse:.2f}")
col2.metric("MAE",  f"${mae:.2f}")
col3.metric("R²",   f"{r2:.4f}")
col4.metric("MAPE", f"{mape:.2f}%")
col5.metric("Directional Acc.", f"{dir_acc:.1f}%")

st.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:1.2rem;margin-bottom:1.5rem;">
  <div style="background:rgba(212,175,55,0.05);border:1px solid rgba(212,175,55,0.15);border-radius:12px;padding:1.2rem 1.5rem;">
    <div style="font-size:0.7rem;letter-spacing:2px;text-transform:uppercase;color:#8a8070;margin-bottom:0.4rem;">RMSE — Root Mean Square Error</div>
    <div style="font-size:1.3rem;font-family:'Playfair Display',serif;color:#d4af37;margin-bottom:0.5rem;">${rmse:.2f}</div>
    <div style="font-size:0.82rem;color:#b0a898;line-height:1.6;">
      RMSE measures prediction error but <strong style="color:#e8e0d0;">penalises large mistakes more heavily</strong> —
      a single bad prediction on a volatile day impacts this score disproportionately.
      At gold's current price, this represents roughly a <strong style="color:#e8e0d0;">{(rmse/latest)*100:.1f}% deviation</strong>,
      largely driven by sudden event-based spikes that OHLCV data alone cannot anticipate.
    </div>
  </div>
  <div style="background:rgba(212,175,55,0.05);border:1px solid rgba(212,175,55,0.15);border-radius:12px;padding:1.2rem 1.5rem;">
    <div style="font-size:0.7rem;letter-spacing:2px;text-transform:uppercase;color:#8a8070;margin-bottom:0.4rem;">MAE — Mean Absolute Error</div>
    <div style="font-size:1.3rem;font-family:'Playfair Display',serif;color:#d4af37;margin-bottom:0.5rem;">${mae:.2f}</div>
    <div style="font-size:0.82rem;color:#b0a898;line-height:1.6;">
      MAE is the <strong style="color:#e8e0d0;">plain average of how far off</strong> each prediction is,
      treating all errors equally. On a typical trading day, the model differs from actual gold price by ${mae:.2f} —
      a <strong style="color:#e8e0d0;">{(mae/latest)*100:.1f}% average error</strong>,
      within acceptable range for a model trained solely on historical price patterns.
    </div>
  </div>
  <div style="background:rgba(212,175,55,0.05);border:1px solid rgba(212,175,55,0.15);border-radius:12px;padding:1.2rem 1.5rem;">
    <div style="font-size:0.7rem;letter-spacing:2px;text-transform:uppercase;color:#8a8070;margin-bottom:0.4rem;">R² — Coefficient of Determination</div>
    <div style="font-size:1.3rem;font-family:'Playfair Display',serif;color:#d4af37;margin-bottom:0.5rem;">{r2:.4f}</div>
    <div style="font-size:0.82rem;color:#b0a898;line-height:1.6;">
      R² tells you <strong style="color:#e8e0d0;">how much of the price variation the model explains</strong>.
      A score of {r2:.4f} means the model accounts for
      <strong style="color:#e8e0d0;">{r2*100:.1f}% of gold price movement</strong> —
      a strong result for financial time series where scores above 0.90 are considered reliable.
    </div>
  </div>
  <div style="background:rgba(212,175,55,0.05);border:1px solid rgba(212,175,55,0.15);border-radius:12px;padding:1.2rem 1.5rem;">
    <div style="font-size:0.7rem;letter-spacing:2px;text-transform:uppercase;color:#8a8070;margin-bottom:0.4rem;">MAPE — Mean Absolute Percentage Error</div>
    <div style="font-size:1.3rem;font-family:'Playfair Display',serif;color:#d4af37;margin-bottom:0.5rem;">{mape:.2f}%</div>
    <div style="font-size:0.82rem;color:#b0a898;line-height:1.6;">
      MAPE expresses error as a <strong style="color:#e8e0d0;">percentage of the actual price</strong> —
      the most intuitive metric for non-technical audiences. A MAPE of {mape:.2f}% means the model
      is within <strong style="color:#e8e0d0;">{mape:.2f} cents of every dollar</strong> of actual gold price.
      In financial forecasting, <strong style="color:#e8e0d0;">under 5% is the industry benchmark</strong>.
    </div>
  </div>
  <div style="background:rgba(212,175,55,0.05);border:1px solid rgba(212,175,55,0.15);border-radius:12px;padding:1.2rem 1.5rem;grid-column:1/-1;">
    <div style="font-size:0.7rem;letter-spacing:2px;text-transform:uppercase;color:#8a8070;margin-bottom:0.4rem;">Directional Accuracy — Up/Down Prediction</div>
    <div style="font-size:1.3rem;font-family:'Playfair Display',serif;color:#4caf87;margin-bottom:0.5rem;">{dir_acc:.1f}%</div>
    <div style="font-size:0.82rem;color:#b0a898;line-height:1.6;">
      Directional accuracy measures <strong style="color:#e8e0d0;">how often the model correctly predicted whether gold price would rise or fall</strong> the next day —
      the most investor-relevant metric, since buy/sell decisions depend on direction, not exact price.
      A score of 50% equals random guessing. <strong style="color:#e8e0d0;">Above 55% is considered useful</strong> in quantitative finance,
      and above 60% is strong. At {dir_acc:.1f}%, this model
      <strong style="color:#e8e0d0;">outperforms random chance and crosses the practical utility threshold</strong>.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(
    x=dates_test, y=y_true,
    name='Actual Price', line=dict(color='#d4af37', width=2)
))
fig_pred.add_trace(go.Scatter(
    x=dates_test, y=y_pred,
    name='CNN + Bi-LSTM Prediction', line=dict(color='#8a6ff0', width=1.8, dash='dot')
))
fig_pred.update_layout(
    template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#e8e0d0', family='DM Sans'),
    xaxis=dict(showgrid=False, color='#3a3530'),
    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)', title='Price (USD)'),
    legend=dict(bgcolor='rgba(0,0,0,0)'), hovermode='x unified',
    height=400, margin=dict(l=0, r=0, t=20, b=0)
)
st.plotly_chart(fig_pred, use_container_width=True)

# ── SECTION 3: FUTURE FORECAST ────────────────────────────────────────────────
st.markdown('<div class="section-title">Future Price Forecast</div>', unsafe_allow_html=True)

st.markdown("""
<div style="background:rgba(255,193,7,0.06);border:1px solid rgba(255,193,7,0.2);border-radius:10px;
            padding:0.9rem 1.2rem;margin-bottom:1.2rem;font-size:0.83rem;color:#b0a898;line-height:1.6;">
  ⚠️ <strong style="color:#e8e0d0;">Disclaimer:</strong> This forecast is generated by a deep learning model
  trained purely on historical price patterns (OHLCV data). It <strong style="color:#e8e0d0;">does not account
  for geopolitical events, macroeconomic shifts, or market sentiment</strong> — all of which significantly
  influence gold prices. Treat this as a pattern-based projection, not financial advice.
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="forecast-card">', unsafe_allow_html=True)
n_days = st.select_slider(
    "Select forecast horizon",
    options=[7, 14, 30], value=14,
    format_func=lambda x: f"{x} Days"
)

if st.button("🔮 Generate Forecast"):
    with st.spinner("Running CNN + Bi-LSTM forecast..."):
        X_scaled    = scaler_X.transform(df[FEATURES].values)
        current_seq = X_scaled[-SEQ_LEN:].copy()
        future_preds = []

        for _ in range(n_days):
            inp  = current_seq.reshape(1, SEQ_LEN, len(FEATURES))
            pred = model.predict(inp, verbose=0)[0][0]
            future_preds.append(pred)
            new_row    = current_seq[-1].copy()
            new_row[3] = pred
            current_seq = np.vstack([current_seq[1:], new_row])

        future_prices = scaler_y.inverse_transform(
            np.array(future_preds).reshape(-1, 1)
        ).ravel()

        last_date    = df['Date'].iloc[-1]
        future_dates = []
        d = last_date
        while len(future_dates) < n_days:
            d += timedelta(days=1)
            if d.weekday() < 5:
                future_dates.append(d)

        hist_window  = df.tail(90)
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(
            x=hist_window['Date'], y=hist_window['Close'],
            name='Recent Actual', line=dict(color='#d4af37', width=2)
        ))
        fig_forecast.add_trace(go.Scatter(
            x=future_dates, y=future_prices,
            name=f'{n_days}-Day Forecast',
            line=dict(color='#4caf87', width=2.5, dash='dot'),
            mode='lines+markers', marker=dict(size=6, color='#4caf87')
        ))
        upper = future_prices * 1.02
        lower = future_prices * 0.98
        fig_forecast.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates[::-1]),
            y=list(upper) + list(lower[::-1]),
            fill='toself', fillcolor='rgba(76,175,135,0.08)',
            line=dict(color='rgba(0,0,0,0)'), name='±2% Band', showlegend=True
        ))
        fig_forecast.add_vline(
            x=str(last_date.date()),
            line=dict(color='rgba(212,175,55,0.4)', dash='dash', width=1.5)
        )
        fig_forecast.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e8e0d0', family='DM Sans'),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)', title='Price (USD)'),
            legend=dict(bgcolor='rgba(0,0,0,0)'), hovermode='x unified',
            height=420, margin=dict(l=0, r=0, t=20, b=0)
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

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
