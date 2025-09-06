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
    return tr.ewm(alpha=1/period, adjust=False).mean()

def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if df.empty or len(df) < period * 2:
        return pd.Series([np.nan] * len(df), index=df.index)
    
    high, low, close = df['high'], df['low'], df['close']

    move_up = high.diff()
    move_down = low.diff().mul(-1)
    plus_dm = pd.Series(np.where((move_up > move_down) & (move_up > 0), move_up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((move_down > move_up) & (move_down > 0), move_down, 0.0), index=df.index)
    
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).abs()
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx

def compute_parabolic_sar(df: pd.DataFrame, af: float = 0.02, max_af: float = 0.2) -> pd.Series:
    if df.empty: return pd.Series([], dtype=float, index=df.index)
    high, low = df['high'], df['low']
    sar = pd.Series(index=df.index, dtype=float)
    
    uptrend = True
    accel = af
    extreme_point = high.iloc[0]
    sar.iloc[0] = low.iloc[0]

    for i in range(1, len(df)):
        prev_sar = sar.iloc[i-1]
        sar.iloc[i] = prev_sar + accel * (extreme_point - prev_sar)

        if uptrend:
            if low.iloc[i] < sar.iloc[i]:
                uptrend = False
                sar.iloc[i] = extreme_point
                extreme_point = low.iloc[i]
                accel = af
            else:
                sar.iloc[i] = min(sar.iloc[i], low.iloc[i-1], low.iloc[i-2] if i > 1 else low.iloc[i-1])
                if high.iloc[i] > extreme_point:
                    extreme_point = high.iloc[i]
                    accel = min(max_af, accel + af)
        else: # Downtrend
            if high.iloc[i] > sar.iloc[i]:
                uptrend = True
                sar.iloc[i] = extreme_point
                extreme_point = high.iloc[i]
                accel = af
            else:
                sar.iloc[i] = max(sar.iloc[i], high.iloc[i-1], high.iloc[i-2] if i > 1 else high.iloc[i-1])
                if low.iloc[i] < extreme_point:
                    extreme_point = low.iloc[i]
                    accel = min(max_af, accel + af)
    return sar

# ------------------- Config -------------------
@dataclass
class Params:
    # --- START OF TUNED PARAMETERS ---
    flush_pct_threshold: float = -0.05   # Relaxed from -0.25
    below_5day_avg_pct: float = -0.15    # Relaxed from -0.40
    volume_spike_mult: float = 1.5       # Relaxed from 3.0
    # --- END OF TUNED PARAMETERS ---
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
    if i < 20: return False
    row = df.iloc[i]
    pct_change = (row['close'] / row['open'] - 1.0) if row['open'] > 0 else 0
    if pct_change > p.flush_pct_threshold: return False
    
    close_5d_avg = df['close'].iloc[i-4:i+1].mean()
    if close_5d_avg > 0:
        below_avg_pct = (row['close'] - close_5d_avg) / close_5d_avg
        if below_avg_pct > p.below_5day_avg_pct: return False
        
    vol_20_avg = df['volume'].iloc[i-19:i+1].mean()
    if vol_20_avg > 0 and row['volume'] < p.volume_spike_mult * vol_20_avg:
        return False
    return True

def sar_gap_check(close: float, sar_val: float, atr_val: float, p: Params) -> bool:
    if pd.isna(sar_val) or pd.isna(atr_val) or sar_val <= 0 or atr_val <= 0: return False
    gap_pct = (sar_val - close) / sar_val
    if gap_pct < p.sar_gap_pct: return False
    atr_gap = (sar_val - close) >= p.sar_atr_mult * atr_val
    if not atr_gap: return False
    return True

def compute_trigger_price(flush_low: float, p: Params) -> float:
    return float(flush_low * p.reclaim_mult)

def scan_candidates(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()
    df['atr'] = compute_atr(df, p.atr_period)
    df['adx'] = compute_adx(df, p.adx_period) 
    df['sar'] = compute_parabolic_sar(df, af=p.sar_af, max_af=p.sar_max_af)
    
    candidates = []
    for i in range(len(df)):
        if is_capitulation_flush(i, df, p):
            row = df.iloc[i]
            if (sar_gap_check(row['close'], row['sar'], row['atr'], p) and 
                not pd.isna(row['adx']) and row['adx'] >= p.adx_threshold):
                candidates.append({
                    'date': df.index[i],
                    'flush_low': float(row['low']),
                    'trigger': compute_trigger_price(row['low'], p),
                    'close': float(row['close']),
                    'sar': float(row['sar']), 'atr': float(row['atr']), 'adx': float(row['adx'])
                })
    return pd.DataFrame(candidates).set_index('date') if candidates else pd.DataFrame()

# ------------------- Trade Simulation -------------------
def simulate_trades(df: pd.DataFrame, p: Params, initial_equity: float = 100000.0) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    default_stats = {'trades': 0,'win_rate_pct': 0,'avg_return_pct': 0,'sharpe_like': 0,'max_drawdown_pct': 0,'final_equity': initial_equity}
    if df.empty or len(df) < p.adx_period * 2: return pd.DataFrame(), pd.DataFrame(), default_stats

    candidates = scan_candidates(df, p)
    if candidates.empty: return pd.DataFrame(), pd.DataFrame(), default_stats
    
    equity, trades, equity_curve = initial_equity, [], [{'date': df.index[0], 'equity': initial_equity}]
    
    for date, signal in candidates.iterrows():
        try:
            signal_idx = df.index.get_loc(date)
            if signal_idx + 1 >= len(df): continue
            
            entry_bar = df.iloc[signal_idx + 1]
            entry_date = df.index[signal_idx + 1]

            if entry_bar['open'] >= signal['trigger']:
                entry_price = float(entry_bar['open'])
            elif entry_bar['low'] <= signal['trigger'] <= entry_bar['high']:
                entry_price = signal['trigger']
            else:
                continue

            position_size = equity * p.max_position_pct
            shares = int(position_size / entry_price) if entry_price > 0 else 0
            if shares <= 0: continue
            
            stop_price = signal['flush_low'] * (1 - p.stop_loss_pct)
            exit_price, exit_date, exit_reason = None, None, "Max Hold"
            
            max_hold_idx = min(len(df) - 1, signal_idx + 1 + p.hold_max_bars)
            for j in range(signal_idx + 2, max_hold_idx + 1):
                current_bar = df.iloc[j]
                current_date = df.index[j]
                
                if current_bar['low'] <= stop_price:
                    exit_price, exit_date, exit_reason = stop_price, current_date, "Stop Loss"
                    break
                if not pd.isna(current_bar['sar']) and current_bar['high'] >= current_bar['sar']:
                    exit_price, exit_date, exit_reason = current_bar['sar'], current_date, "SAR Exit"
                    break
            
            if exit_price is None:
                exit_price, exit_date = float(df.iloc[max_hold_idx]['close']), df.index[max_hold_idx]
            
            pnl = (exit_price - entry_price) * shares
            return_pct = 100 * (exit_price / entry_price - 1)
            equity += pnl
            
            trades.append({'entry_date': entry_date,'entry_price': entry_price,'exit_date': exit_date,'exit_price': exit_price,
                           'shares': shares,'pnl': pnl,'return_pct': return_pct,'exit_reason': exit_reason})
            equity_curve.append({'date': exit_date, 'equity': equity})
        except (IndexError, KeyError) as e:
            st.warning(f"Skipping trade simulation on {date} due to data error: {e}")
            continue

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    curve_df = pd.DataFrame(equity_curve).set_index('date').sort_index() if equity_curve else pd.DataFrame()
    curve_df['equity'] = curve_df['equity'].cummax()
    
    if not trades_df.empty:
        total_trades, winning_trades = len(trades_df), (trades_df['pnl'] > 0).sum()
        win_rate_pct = 100 * winning_trades / total_trades if total_trades > 0 else 0
        avg_return_pct = trades_df['return_pct'].mean()
        
        returns = curve_df['equity'].pct_change().dropna()
        sharpe_like = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        
        rolling_max = curve_df['equity'].cummax()
        drawdown = (curve_df['equity'] - rolling_max) / rolling_max
        max_drawdown_pct = abs(100 * drawdown.min())
        
        stats = {'trades': total_trades,'win_rate_pct': win_rate_pct,'avg_return_pct': avg_return_pct,
                 'sharpe_like': sharpe_like,'max_drawdown_pct': max_drawdown_pct,'final_equity': equity}
    else:
        stats = default_stats
    return trades_df, curve_df, stats

# ------------------- UI -------------------
st.title("Eteon Capital")
st.header("SAR Reversion Spike Backtest")
st.markdown("Upload OHLCV data or enter tickers to fetch automatically.")

with st.sidebar:
    st.header("Parameters")
    st.info("Parameters have been adjusted to capture setups similar to BMNR and VWAV.")
    p.flush_pct_threshold = st.slider("Flush Pct Threshold", -1.0, 0.0, p.flush_pct_threshold, 0.01)
    p.below_5day_avg_pct = st.slider("Below 5-Day Avg Pct", -1.0, 0.0, p.below_5day_avg_pct, 0.01)
    p.volume_spike_mult = st.slider("Volume Spike Multiplier", 1.0, 10.0, p.volume_spike_mult, 0.1)
    p.adx_threshold = st.slider("ADX Threshold", 10.0, 50.0, p.adx_threshold, 0.5)

mode = st.radio("Select data source:", ["Enter ticker(s)", "Upload CSV/ZIP"])
datasets = []

# ------------------- Data Loading -------------------
if mode == "Upload CSV/ZIP":
    uploaded_files = st.file_uploader("Upload CSV or ZIP files", type=["csv", "zip"], accept_multiple_files=True)

    def load_csv_file(f, filename) -> Tuple[str, pd.DataFrame]:
        df = pd.read_csv(f)
        df.columns = [c.strip().lower() for c in df.columns]
        required = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            raise ValueError(f"CSV must contain {required} columns.")
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date')
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].astype(float)
        symbol = os.path.splitext(filename)[0].upper()
        return symbol, df[numeric_cols]

    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                if uploaded_file.name.lower().endswith('.zip'):
                    with zipfile.ZipFile(uploaded_file) as z:
                        for filename in z.namelist():
                            if filename.lower().endswith('.csv'):
                                with z.open(filename) as f:
                                    symbol, df = load_csv_file(io.TextIOWrapper(f), filename)
                                    datasets.append((symbol, df))
                elif uploaded_file.name.lower().endswith('.csv'):
                    symbol, df = load_csv_file(uploaded_file, uploaded_file.name)
                    datasets.append((symbol, df))
            except Exception as e:
                st.error(f"Failed to load {uploaded_file.name}: {e}")

elif mode == "Enter ticker(s)":
    tickers = st.text_input("Enter comma-separated tickers:", "BMNR,VWAV,ALAB,SOUN")
    start_date = st.date_input("Start date", datetime(2024, 1, 1))
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
                
                # --- START OF BUG FIX ---
                # Correctly handle and standardize column names to prevent KeyError
                if isinstance(df.columns, pd.MultiIndex):
                    # For multi-level columns, take the first level (e.g., 'Open', 'High')
                    df.columns = df.columns.get_level_values(0)
                
                # Convert all column names to lowercase strings
                df.columns = df.columns.str.lower()

                # Handle yfinance's 'adj close' column
                if 'adj close' in df.columns:
                    df = df.rename(columns={'adj close': 'close'})
                
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required_cols):
                    st.error(f"Downloaded data for {ticker} is missing required columns. Got: {df.columns.tolist()}")
                    continue
                # --- END OF BUG FIX ---
                
                datasets.append((ticker, df))
            except Exception as e:
                st.error(f"Failed to fetch {ticker}: {e}")

# ------------------- Run backtest and display results -------------------
if datasets:
    st.header("Backtest Results")
    all_trades_list = []
    
    for symbol, df in datasets:
        # Ensure the dataframe has the required columns before simulation
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Cannot run simulation for {symbol}: DataFrame is missing columns. Required: {required_cols}")
            continue

        trades_df, curve_df, stats = simulate_trades(df, p, initial_equity)
        
        if not trades_df.empty:
            trades_df['symbol'] = symbol
            all_trades_list.append(trades_df)

        with st.expander(f"Results for {symbol} - {stats['trades']} Trades"):
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Final Equity", f"${stats['final_equity']:,.2f}")
            col2.metric("Win Rate", f"{stats['win_rate_pct']:.2f}%")
            col3.metric("Avg Return/Trade", f"{stats['avg_return_pct']:.2f}%")
            col4.metric("Sharpe-like", f"{stats['sharpe_like']:.2f}")
            col5.metric("Max Drawdown", f"{stats['max_drawdown_pct']:.2f}%")

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
            fig.suptitle(f'Equity Curve & Trades for {symbol}', fontsize=16)
            
            ax1.plot(curve_df.index, curve_df['equity'], label='Equity', color='blue')
            ax1.set_title('Equity Curve')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.grid(True, alpha=0.3)

            ax2.plot(df.index, df['close'], label='Close Price', color='black', alpha=0.6)
            if 'sar' in df.columns:
                ax2.plot(df.index, df['sar'], ':', color='purple', label='Parabolic SAR', alpha=0.5)
            
            if not trades_df.empty:
                buys = trades_df.set_index('entry_date')
                sells = trades_df.set_index('exit_date')
                ax2.plot(buys.index, buys['entry_price'], '^', markersize=8, color='green', label='Buy')
                ax2.plot(sells.index, sells['exit_price'], 'v', markersize=8, color='red', label='Sell')
            
            ax2.set_title('Price Chart & Trades')
            ax2.set_ylabel('Price ($)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            st.pyplot(fig)

            st.write("Trade Log:")
            st.dataframe(trades_df)

    if all_trades_list:
        st.header("Combined Portfolio Results")
        all_trades_df = pd.concat(all_trades_list).sort_values(by='entry_date').reset_index(drop=True)
        
        total_pnl = all_trades_df['pnl'].sum()
        final_portfolio_equity = initial_equity * len(datasets) + total_pnl
        total_trades = len(all_trades_df)
        winning_trades = (all_trades_df['pnl'] > 0).sum()
        agg_win_rate = 100 * winning_trades / total_trades if total_trades > 0 else 0
        agg_avg_return = all_trades_df['return_pct'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Trades", total_trades)
        col2.metric("Overall Final Equity (Sum)", f"${final_portfolio_equity:,.2f}")
        col3.metric("Aggregate Win Rate", f"{agg_win_rate:.2f}%")
        col4.metric("Aggregate Avg Return", f"{agg_avg_return:.2f}%")

        st.write("All Trades Combined:")
        st.dataframe(all_trades_df)
