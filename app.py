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
    if df.empty:
        return pd.Series([], dtype=float, index=df.index)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=1).mean()
    return atr.reindex(df.index)

def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if df.empty or len(df) < period:
        return pd.Series([np.nan]*len(df), index=df.index)
    high, low, close = df['high'], df['low'], df['close']
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=1).mean().replace(0, np.nan)
    plus_di = 100 * (plus_dm.rolling(period).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(period).sum() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(period, min_periods=1).mean()
    return adx.reindex(df.index)  # ✅ ensures alignment

def compute_parabolic_sar(df: pd.DataFrame, af: float = 0.02, max_af: float = 0.2) -> pd.Series:
    if df.empty:
        return pd.Series([], dtype=float, index=df.index)
    high, low = df['high'].values, df['low'].values
    n = len(df); sar = np.zeros(n)
    up = True; ep = high[0]; sar[0] = low[0]
    a = af; af_local = a
    for i in range(1, n):
        prev = sar[i-1]
        curr_sar = prev + af_local * (ep - prev)
        if up:
            curr_sar = min(curr_sar, low[i-1], low[i-2] if i-2>=0 else low[i-1])
        else:
            curr_sar = max(curr_sar, high[i-1], high[i-2] if i-2>=0 else high[i-1])
        if up:
            if low[i] < curr_sar:
                up = False; curr_sar = ep; ep = low[i]; af_local = a
            else:
                if high[i] > ep:
                    ep = high[i]; af_local = min(af_local + a, max_af)
        else:
            if high[i] > curr_sar:
                up = True; curr_sar = ep; ep = high[i]; af_local = a
            else:
                if low[i] < ep:
                    ep = low[i]; af_local = min(af_local + a, max_af)
        sar[i] = curr_sar
    return pd.Series(sar, index=df.index)

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

# ------------------- Strategy Logic -------------------
def is_capitulation_flush(i: int, df: pd.DataFrame, p: Params) -> bool:
    row = df.iloc[i]
    pct_change = (row['close'] / row['open'] - 1.0)
    close_5d_avg = df['close'].iloc[max(0, i-4):i+1].mean()
    cond1 = pct_change <= p.flush_pct_threshold
    cond2 = (row['close'] - close_5d_avg) / close_5d_avg <= p.below_5day_avg_pct
    vol_20 = df['volume'].iloc[max(0, i-19):i+1].mean()
    cond3 = row['volume'] >= p.volume_spike_mult * (vol_20 if vol_20>0 else 1)
    return bool(cond1 and cond2 and cond3)

def sar_gap_check(close: float, sar_val: float, atr_val: float, p: Params) -> bool:
    if pd.isna(sar_val) or pd.isna(atr_val) or sar_val <= 0 or atr_val <= 0:
        return False
    gap_pct = (sar_val - close) / sar_val
    atr_gap = (sar_val - close) >= p.sar_atr_mult * atr_val
    return (gap_pct >= p.sar_gap_pct) and atr_gap

def compute_trigger_price(flush_low: float, p: Params) -> float:
    return float(flush_low * p.reclaim_mult)

def scan_candidates(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    d = df.copy()
    d['ATR'] = compute_atr(d, p.atr_period)
    d['ADX'] = compute_adx(d, p.adx_period)
    d['SAR'] = compute_parabolic_sar(d, af=p.sar_af, max_af=p.sar_max_af)
    out = []
    for i in range(len(d)):
        if is_capitulation_flush(i, d, p):
            row = d.iloc[i]
            if sar_gap_check(row['close'], row['SAR'], row['ATR'], p) and (row['ADX'] >= p.adx_threshold):
                out.append({
                    'date': d.index[i],
                    'flush_low': float(row['low']),
                    'trigger': float(compute_trigger_price(row['low'], p)),
                    'close': float(row['close']),
                    'sar': float(row['SAR']),
                    'atr': float(row['ATR']),
                    'adx': float(row['ADX'])
                })
    return pd.DataFrame(out).set_index('date') if out else pd.DataFrame(columns=['flush_low','trigger','close','sar','atr','adx'])

def simulate_trades(df: pd.DataFrame, p: Params, initial_equity: float = 100000.0) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if df.empty or len(df) < p.adx_period + 1:
        return pd.DataFrame(), pd.DataFrame(), {'trades':0,'win_rate_pct':0,'avg_return_pct':0,'sharpe_like':np.nan,'max_drawdown_pct':0,'final_equity':initial_equity}
    candidates = scan_candidates(df, p)
    equity = initial_equity; equity_curve = []; trades = []
    for idx, sig in candidates.iterrows():
        i = df.index.get_loc(idx); next_i = i + 1
        if next_i >= len(df): continue
        next_bar = df.iloc[next_i]
        if next_bar['close'] >= sig['trigger']:
            entry_price = float(next_bar['close'])
            shares = int((equity * p.max_position_pct) // entry_price)
            if shares <= 0: continue
            flush_low = float(sig['flush_low'])
            stop_price = float(flush_low * (1 - p.stop_loss_pct))
            exit_price = None; exit_date = None
            for j in range(next_i, min(len(df), next_i + p.hold_max_bars)):
                if df['low'].iloc[j] <= stop_price:
                    exit_price = stop_price; exit_date = df.index[j]; break
                if df['high'].iloc[j] >= df['SAR'].iloc[j]:
                    exit_price = float(df['close'].iloc[j]); exit_date = df.index[j]; break
            if exit_price is None:
                exit_price = float(df['close'].iloc[min(len(df)-1, next_i + 20)])
                exit_date = df.index[min(len(df)-1, next_i + 20)]
            pnl = (exit_price - entry_price) * shares
            ret = pnl / (entry_price * shares) if entry_price*shares>0 else 0.0
            equity += pnl
            trades.append({
                'entry_date': df.index[next_i],
                'entry_price': entry_price,
                'exit_date': exit_date,
                'exit_price': exit_price,
                'shares': shares,
                'pnl': pnl,
                'return_pct': 100*ret
            })
        equity_curve.append({'date': df.index[next_i], 'equity': equity})
    trades_df = pd.DataFrame(trades)
    curve_df = pd.DataFrame(equity_curve).set_index('date') if len(equity_curve) else pd.DataFrame(columns=['equity'])
    wins = (trades_df['pnl'] > 0).sum() if len(trades_df) else 0
    total = len(trades_df); win_rate = 100 * wins / total if total else 0.0
    avg_ret = trades_df['return_pct'].mean() if total else 0.0
    daily_returns = curve_df['equity'].pct_change().dropna() if len(curve_df) else pd.Series(dtype=float)
    sharpe = (np.sqrt(252) * daily_returns.mean() / daily_returns.std()) if len(daily_returns) and daily_returns.std()>0 else np.nan
    if len(curve_df):
        roll_max = curve_df['equity'].cummax(); drawdown = curve_df['equity'] / roll_max - 1.0; max_dd = 100 * drawdown.min() if len(drawdown) else 0.0
    else:
        max_dd = 0.0
    stats = {'trades': total,'win_rate_pct': win_rate,'avg_return_pct': avg_ret,'sharpe_like': float(sharpe) if sharpe==sharpe else np.nan,'max_drawdown_pct': max_dd,'final_equity': equity}
    return trades_df, curve_df, stats

# ------------------- UI -------------------
st.title("SAR Reversion Spike — Streamlit App")
st.markdown("Upload OHLCV data or enter tickers to fetch automatically.")

with st.sidebar:
    st.header("Parameters")
    p = Params(
        flush_pct_threshold = st.slider("Flush % drop (Close/Open - 1)", -0.60, -0.05, -0.25, 0.01),
        below_5day_avg_pct = st.slider("Below 5-day avg close", -0.80, -0.05, -0.40, 0.01),
        volume_spike_mult   = st.slider("Volume spike multiple (vs 20-day)", 1.0, 10.0, 3.0, 0.1),
        sar_gap_pct         = st.slider("SAR distance ≥ %", 0.05, 0.60, 0.20, 0.01),
        sar_atr_mult        = st.slider("SAR distance ≥ ATR ×", 0.5, 5.0, 2.0, 0.1),
        adx_threshold       = st.slider("ADX threshold", 5, 60, 25, 1),
        reclaim_mult        = st.slider("Reclaim trigger (× low)", 1.00, 1.20, 1.05, 0.01),
        stop_loss_pct       = st.slider("Stop below flush low", 0.02, 0.20, 0.08, 0.01),
        max_position_pct    = st.slider("Max position % equity", 0.01, 0.20, 0.05, 0.01),
        atr_period          = st.slider("ATR period", 5, 30, 14, 1),
        adx_period          = st.slider("ADX period", 5, 30, 14, 1),
        sar_af              = st.slider("SAR AF (step)", 0.01, 0.1, 0.02, 0.01),
        sar_max_af          = st.slider("SAR Max AF", 0.1, 0.5, 0.2, 0.05),
        hold_max_bars       = st.slider("Max hold bars (days)", 5, 90, 60, 5),
    )
    initial_equity = st.number_input("Initial equity ($)", 10000.0, 10000000.0, 100000.0, step=1000.0)

mode = st.radio("Select data source:", ["Upload CSV/ZIP", "Enter ticker(s)"])

datasets = []
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
        for needed in ['date','open','high','low','close','volume']:
            if needed not in df.columns:
                raise ValueError(f"Missing column {needed}")
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date')
        return df[['open','high','low','close','volume']].astype(float)

    files = []
    if upload_mode == "Single CSV":
        files = [uploaded] if uploaded else []
    elif upload_mode == "Multiple CSVs":
        files = uploaded if uploaded else []
    else:
        if uploaded:
            with zipfile.ZipFile(uploaded) as z:
                for name in z.namelist():
                    if name.lower().endswith('.csv'):
                        files.append((name, z.read(name)))

    for item in files:
        try:
            if isinstance(item, tuple):
                name, data = item
                df = load_csv_file(io.BytesIO(data))
                symbol = os.path.splitext(os.path.basename(name))[0].upper()
                df.attrs['symbol'] = symbol; datasets.append((symbol, df))
            else:
                name = item.name
                df = load_csv_file(item)
                symbol = os.path.splitext(os.path.basename(name))[0].upper()
                df.attrs['symbol'] = symbol; datasets.append((symbol, df))
        except Exception as e:
            st.error(f"Failed to load {getattr(item,'name',str(item))}: {e}")

elif mode == "Enter ticker(s)":
    tickers = st.text_input("Enter comma-separated tickers:", "AAPL,TSLA")
    start_date = st.date_input("Start date", datetime(2020,1,1))
    end_date = st.date_input("End date", datetime.today())
    if st.button("Fetch & Run Backtest"):
        for t in [x.strip().upper() for x in tickers.split(',') if x.strip()]:
            try:
                df = yf.download(t, start=start_date, end=end_date)
                if df.empty:
                    st.warning(f"No data found for {t} in this date range.")
                    continue
                df = df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
                df = df[['open','high','low','close','volume']]
                df.attrs['symbol'] = t
                datasets.append((t, df))
            except Exception as e:
                st.error(f"Failed to fetch {t}: {e}")

# ------------------- Run backtest -------------------
if datasets:
    all_trades = []
    for symbol, df in datasets:
        trades_df, curve_df, stats = simulate_trades(df, p, initial_equity=initial_equity)
        if stats['trades'] == 0:
            st.warning(f"No trades generated for {symbol} in this range.")
            continue
        trades_df['symbol'] = symbol
        all_trades.append(trades_df)

        st.subheader(f"Results for {symbol}")
        st.write(f"Trades: {stats['trades']}, Win%: {stats['win_rate_pct']:.1f}, Sharpe: {stats['sharpe_like']:.2f}, Final Equity: ${stats['final_equity']:,.0f}")

        fig = plt.figure(figsize=(10,4))
        plt.plot(df.index, df['close'], label='Close')
        plt.plot(df.index, compute_parabolic_sar(df, p.sar_af, p.sar_max_af), '--', label='SAR')
        if not trades_df.empty:
            plt.scatter(trades_df['entry_date'], trades_df['entry_price'], marker='^', color='g', label='Entry')
            plt.scatter(trades_df['exit_date'], trades_df['exit_price'], marker='v', color='r', label
