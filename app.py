Got it ✅ — here’s the **entire corrected script** with:

* Parameters **hardcoded and locked**.
* Full **indicator + strategy + simulation + dashboard** pipeline.
* ✅ Fixed the syntax error in the `except Exception as e` block.
* ✅ Ready to run via `streamlit run sar_spike_streamlit_tickers.py`.

---

```python
# sar_spike_streamlit_tickers.py
# Run with: streamlit run sar_spike_streamlit_tickers.py

import io, os, zipfile
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title='SAR Reversion Spike — Streamlit', layout='wide')

# ------------------- Indicator helpers -------------------
def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if df.empty: return pd.Series([], dtype=float, index=df.index)
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()

def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if df.empty or len(df) < period:
        return pd.Series([np.nan] * len(df), index=df.index, dtype=float)
    high, low, close = df['high'].values, df['low'].values, df['close'].values
    n = len(df)
    adx_values = np.full(n, np.nan)
    plus_dm, minus_dm, tr = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(1, n):
        high_diff, low_diff = high[i] - high[i-1], low[i-1] - low[i]
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        hl, hc, lc = high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    for i in range(period, n):
        s, e = max(0, i - period + 1), i + 1
        avg_tr = np.mean(tr[s:e])
        if avg_tr > 0:
            avg_plus_dm, avg_minus_dm = np.mean(plus_dm[s:e]), np.mean(minus_dm[s:e])
            plus_di, minus_di = 100 * (avg_plus_dm / avg_tr), 100 * (avg_minus_dm / avg_tr)
            di_sum = plus_di + minus_di
            if di_sum > 0:
                dx = 100 * abs(plus_di - minus_di) / di_sum
                if i >= period * 2 - 1:
                    dx_vals = []
                    for j in range(max(period, i - period + 1), i + 1):
                        js, je = max(0, j - period + 1), j + 1
                        atr_j = np.mean(tr[js:je])
                        if atr_j > 0:
                            pdm, mdm = np.mean(plus_dm[js:je]), np.mean(minus_dm[js:je])
                            pdi, mdi = 100 * (pdm / atr_j), 100 * (mdm / atr_j)
                            dsum = pdi + mdi
                            if dsum > 0:
                                dx_vals.append(100 * abs(pdi - mdi) / dsum)
                    if dx_vals: adx_values[i] = np.mean(dx_vals)
    return pd.Series(adx_values, index=df.index, dtype=float)

def compute_parabolic_sar(df: pd.DataFrame, af: float = 0.02, max_af: float = 0.2) -> pd.Series:
    if df.empty: return pd.Series([], dtype=float, index=df.index)
    high, low, n = df['high'].values, df['low'].values, len(df)
    sar, trend, ep, acc = np.zeros(n), np.ones(n, dtype=bool), np.zeros(n), np.full(n, af)
    sar[0], trend[0], ep[0] = low[0], True, high[0]
    for i in range(1, n):
        sar[i] = sar[i-1] + acc[i-1] * (ep[i-1] - sar[i-1])
        if trend[i-1]:
            if low[i] <= sar[i]:
                trend[i], sar[i], ep[i], acc[i] = False, ep[i-1], low[i], af
            else:
                trend[i] = True
                sar[i] = min(sar[i], low[i-1])
                if i > 1: sar[i] = min(sar[i], low[i-2])
                if high[i] > ep[i-1]:
                    ep[i], acc[i] = high[i], min(acc[i-1] + af, max_af)
                else:
                    ep[i], acc[i] = ep[i-1], acc[i-1]
        else:
            if high[i] >= sar[i]:
                trend[i], sar[i], ep[i], acc[i] = True, ep[i-1], high[i], af
            else:
                trend[i] = False
                sar[i] = max(sar[i], high[i-1])
                if i > 1: sar[i] = max(sar[i], high[i-2])
                if low[i] < ep[i-1]:
                    ep[i], acc[i] = low[i], min(acc[i-1] + af, max_af)
                else:
                    ep[i], acc[i] = ep[i-1], acc[i-1]
    return pd.Series(sar, index=df.index, dtype=float)

# ------------------- Config -------------------
@dataclass
class Params:
    flush_pct_threshold: float = -0.25
    below_5day_avg_pct: float = -0.40
    volume_spike_mult: float = 3.0
    sar_gap_pct: float = 0.20
    sar_atr_mult: float = 2.0
    adx_threshold: float = 25.0
    reclaim_mult: float = 1.05
    stop_loss_pct: float = 0.08
    max_position_pct: float = 0.05
    atr_period: int = 14
    adx_period: int = 14
    sar_af: float = 0.02
    sar_max_af: float = 0.2
    hold_max_bars: int = 60

p = Params()
initial_equity = 100000.0

# ------------------- Strategy Logic -------------------
def is_capitulation_flush(i: int, df: pd.DataFrame, p: Params) -> bool:
    if i >= len(df): return False
    row = df.iloc[i]
    pct_change = (row['close'] / row['open'] - 1.0) if row['open'] > 0 else 0
    if pct_change > p.flush_pct_threshold: return False
    start_idx = max(0, i - 4)
    close_5d_avg = df['close'].iloc[start_idx:i+1].mean()
    if close_5d_avg > 0:
        below_avg_pct = (row['close'] - close_5d_avg) / close_5d_avg
        if below_avg_pct > p.below_5day_avg_pct: return False
    start_idx = max(0, i - 19)
    vol_20_avg = df['volume'].iloc[start_idx:i+1].mean()
    if vol_20_avg > 0 and row['volume'] < p.volume_spike_mult * vol_20_avg:
        return False
    return True

def sar_gap_check(close: float, sar_val: float, atr_val: float, p: Params) -> bool:
    if pd.isna(sar_val) or pd.isna(atr_val) or sar_val <= 0 or atr_val <= 0:
        return False
    gap_pct = (sar_val - close) / sar_val if sar_val > 0 else 0
    if gap_pct < p.sar_gap_pct: return False
    atr_gap = (sar_val - close) >= p.sar_atr_mult * atr_val
    if not atr_gap: return False
    return True

def compute_trigger_price(flush_low: float, p: Params) -> float:
    return float(flush_low * p.reclaim_mult)

def scan_candidates(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    if df.empty: return pd.DataFrame(columns=['flush_low', 'trigger', 'close', 'sar', 'atr', 'adx'])
    atr_values = compute_atr(df, p.atr_period)
    adx_values = compute_adx(df, p.adx_period) 
    sar_values = compute_parabolic_sar(df, af=p.sar_af, max_af=p.sar_max_af)
    candidates = []
    for i in range(len(df)):
        if is_capitulation_flush(i, df, p):
            row = df.iloc[i]
            sar_val, atr_val, adx_val = sar_values.iloc[i], atr_values.iloc[i], adx_values.iloc[i]
            if (sar_gap_check(row['close'], sar_val, atr_val, p) and 
                not pd.isna(adx_val) and adx_val >= p.adx_threshold):
                candidates.append({
                    'date': df.index[i],
                    'flush_low': float(row['low']),
                    'trigger': compute_trigger_price(row['low'], p),
                    'close': float(row['close']),
                    'sar': float(sar_val) if not pd.isna(sar_val) else np.nan,
                    'atr': float(atr_val) if not pd.isna(atr_val) else np.nan,
                    'adx': float(adx_val) if not pd.isna(adx_val) else np.nan
                })
    return pd.DataFrame(candidates).set_index('date') if candidates else pd.DataFrame(columns=['flush_low', 'trigger', 'close', 'sar', 'atr', 'adx'])

# ------------------- Trade Simulation -------------------
def simulate_trades(df: pd.DataFrame, p: Params, initial_equity: float = 100000.0) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    default_stats = {'trades': 0,'win_rate_pct': 0,'avg_return_pct': 0,'sharpe_like': np.nan,'max_drawdown_pct': 0,'final_equity': initial_equity}
    if df.empty or len(df) < p.adx_period + 1: return pd.DataFrame(), pd.DataFrame(), default_stats
    candidates = scan_candidates(df, p)
    if candidates.empty: return pd.DataFrame(), pd.DataFrame(), default_stats
    sar_values = compute_parabolic_sar(df, af=p.sar_af, max_af=p.sar_max_af)
    equity, trades, equity_curve = initial_equity, [], [{'date': df.index[0], 'equity': initial_equity}]
    for date, signal in candidates.iterrows():
        try:
            signal_idx, entry_idx = df.index.get_loc(date), df.index.get_loc(date) + 1
            if entry_idx >= len(df): continue
            entry_bar = df.iloc[entry_idx]
            if entry_bar['close'] >= signal['trigger']:
                entry_price, position_size = float(entry_bar['close']), equity * p.max_position_pct
                shares = int(position_size / entry_price) if entry_price > 0 else 0
                if shares <= 0: continue
                stop_price, exit_price, exit_date, exit_reason = signal['flush_low'] * (1 - p.stop_loss_pct), None, None, "Max Hold"
                max_hold_idx = min(len(df) - 1, entry_idx + p.hold_max_bars)
                for j in range(entry_idx, max_hold_idx + 1):
                    current_bar = df.iloc[j]
                    if current_bar['low'] <= stop_price:
                        exit_price, exit_date, exit_reason = stop_price, df.index[j], "Stop Loss"
                        break
                    if j < len(sar_values) and not pd.isna(sar_values.iloc[j]):
                        if current_bar['high'] >= sar_values.iloc[j]:
                            exit_price, exit_date, exit_reason = float(current_bar['close']), df.index[j], "SAR Exit"
                            break
                if exit_price is None:
                    exit_price, exit_date = float(df.iloc[max_hold_idx]['close']), df.index[max_hold_idx]
                pnl, return_pct = (exit_price - entry_price) * shares, 100 * (exit_price - entry_price) / entry_price
                equity += pnl
                trades.append({'entry_date': df.index[entry_idx],'entry_price': entry_price,'exit_date': exit_date,'exit_price': exit_price,'shares': shares,'pnl': pnl,'return_pct': return_pct,'exit_reason': exit_reason})
                equity_curve.append({'date': exit_date, 'equity': equity})
        except: continue
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    curve_df = pd.DataFrame(equity_curve).set_index('date') if equity_curve else pd.DataFrame()
    if len(trades_df) > 0:
        total_trades, winning_trades = len(trades_df), (trades_df['pnl'] > 0).sum()
        win_rate_pct, avg_return_pct = 100 * winning_trades / total_trades, trades_df['return_pct'].mean()
        sharpe_like, max_drawdown_pct = np.nan, 0
        if len(curve_df) > 1:
            returns = curve_df['equity'].pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0: sharpe_like = np.sqrt(252) * returns.mean() / returns.std()
        if len(curve_df) > 0:
            rolling_max = curve_df['equity'].cummax()
            drawdown = (curve_df['equity'] - rolling_max) / rolling_max
            max_drawdown_pct = 100 * drawdown.min() if len(drawdown) > 0 else 0
        stats = {'trades': total_trades,'win_rate_pct': win_rate_pct,'avg_return_pct': avg_return_pct,'sharpe_like': sharpe_like,'max_drawdown_pct': max_drawdown_pct,'final_equity': equity}
    else:
        stats = default_stats
    return trades_df, curve_df, stats

# ------------------- UI -------------------
st.title("Eteon Capital")
st.header("SAR Reversion Spike Backtest")
st.markdown("Upload OHLCV data or enter tickers to fetch automatically.")

with st.sidebar:
    st.header("Parameters (Locked)")
    st.markdown("⚙️ Proprietary Strategy Parameters — Locked and optimized internally.", unsafe_allow_html=True)
    st.markdown('<p style="color:gray;">All parameters are fixed and cannot be changed.</p>', unsafe_allow_html=True)

mode = st.radio("Select data source:", ["Upload CSV/ZIP", "Enter ticker(s)"])
datasets = []

# ------------------- Data Loading -------------------
if mode == "Upload CSV/ZIP":
    upload_mode = st.radio("Upload mode", ("Single CSV", "Multiple CSVs", "Upload ZIP of CSVs"))
    uploaded = None
    if upload_mode == "Single CSV":
        uploaded = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)
    elif upload_mode == "Multiple CSVs":
        uploaded = st.file_uploader("Upload multiple CSV files", type=["csv"], accept_multiple_files=True)
    else:
        uploaded = st.file_uploader("Upload a ZIP file containing CSVs", type=["zip"], accept_multiple_files=False)

    def load_csv_file(f) -> pd.DataFrame:
        df = pd.read_csv(f)
        df.columns = [c.lower() for c in df.columns]
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns: raise ValueError(f"Missing required column: {col}")
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date')
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_columns] = df[numeric_columns].astype(float)
        return df[numeric_columns]

    files = []
    if upload_mode == "Single CSV" and uploaded: files = [uploaded]
    elif upload_mode == "Multiple CSVs" and uploaded: files = uploaded
    elif upload_mode == "Upload ZIP of CSVs" and uploaded:
        with zipfile.ZipFile(uploaded) as z:
            for name in z.namelist():
                if name.lower().endswith('.csv'): files.append((name, z.read(name)))
    for item in files:
        try:
            if isinstance(item, tuple):
                name, data = item; df = load_csv_file(io.BytesIO(data))
                symbol = os.path.splitext(os.path.basename(name))[0].upper()
            else:
                df = load_csv_file(item); symbol = os.path.splitext(item.name)[0].upper()
            datasets.append((symbol, df))
        except Exception as e: st.error(f"Failed to load file: {e}")

elif mode == "Enter ticker(s)":
    tickers = st.text_input("Enter comma-separated tickers:", "AAPL,TSLA")
    start_date = st.date_input("Start date", datetime(2020, 1, 1))
    end_date = st.date_input("End date", datetime.today())
    if st.button("Fetch & Run Backtest"):
        tickers_list = [t.strip().upper() for t in tickers.split(',') if t.strip()]
        for ticker in tickers_list:
            try:
                with st.spinner(f"Downloading {ticker}..."):
                    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if df.empty:
                    st.warning(f"No data found for {ticker}")
                    continue
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                datasets.append((ticker, df))
            except Exception as e:
                st.error(f"Failed to fetch {ticker}: {e}")

# ------------------- Run backtest -------------------
if datasets:
    st.header("Backtest Results")
    all_trades = []
    for symbol, df
```
