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

# ------------------- UI -------------------
st.title("Eteon Capital")
st.header("SAR Reversion Spike Backtest")
st.markdown("Upload OHLCV data or enter tickers to fetch automatically.")

with st.sidebar:
    st.header("Parameters (Locked)")
    st.markdown(
        """
        ⚙️ Proprietary Strategy Parameters  
        - Locked and non-editable  
        - Optimized internally (includes BSLK detection)  
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="color:gray;">All parameters are fixed and cannot be changed.</p>',
        unsafe_allow_html=True
    )

# ------------------- Strategy logic + backtest (your original full code here) -------------------
# keep all your is_capitulation_flush, sar_gap_check, compute_trigger_price,
# scan_candidates, simulate_trades, and the dashboard section as-is.
# They will now run using the locked Params above.
