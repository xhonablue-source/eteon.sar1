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

st.set_page_config(page_title='SAR Reversion Spike — Backtest', layout='wide')

# ------------------- Indicator helpers -------------------
def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range"""
    if df.empty:
        return pd.Series([], dtype=float, index=df.index)
    
    high = df['high']
    low = df['low'] 
    close = df['close']
    
    high_low = high - low
    high_close = (high - close.shift(1)).abs()
    low_close = (low - close.shift(1)).abs()
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period, min_periods=1).mean()
    
    return atr

def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average Directional Index - completely rewritten to avoid pandas issues"""
    if df.empty or len(df) < period:
        return pd.Series([np.nan] * len(df), index=df.index, dtype=float)
    
    # Convert to numpy arrays for calculation
    high = df['high'].values
    low = df['low'].values  
    close = df['close'].values
    n = len(df)
    
    # Initialize output array
    adx_values = np.full(n, np.nan)
    
    # Calculate directional movements
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    true_range = np.zeros(n)
    
    for i in range(1, n):
        # Directional movements
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        
        # True Range
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        true_range[i] = max(hl, hc, lc)
    
    # Calculate smoothed values using Wilder's smoothing
    for i in range(period, n):
        # Get the period data
        start_idx = max(0, i - period + 1)
        end_idx = i + 1
        
        # Calculate averages
        avg_plus_dm = np.mean(plus_dm[start_idx:end_idx])
        avg_minus_dm = np.mean(minus_dm[start_idx:end_idx]) 
        avg_tr = np.mean(true_range[start_idx:end_idx])
        
        if avg_tr > 0:
            # Calculate DI+ and DI-
            plus_di = 100 * (avg_plus_dm / avg_tr)
            minus_di = 100 * (avg_minus_dm / avg_tr)
            
            # Calculate DX
            di_sum = plus_di + minus_di
            if di_sum > 0:
                dx = 100 * abs(plus_di - minus_di) / di_sum
                
                # Calculate ADX (simple moving average of DX)
                if i >= period * 2 - 1:  # Need enough DX values
                    dx_start = max(period, i - period + 1)
                    # Calculate DX for previous periods to get ADX
                    dx_values = []
                    for j in range(dx_start, i + 1):
                        j_start = max(0, j - period + 1)
                        j_end = j + 1
                        j_avg_plus_dm = np.mean(plus_dm[j_start:j_end])
                        j_avg_minus_dm = np.mean(minus_dm[j_start:j_end])
                        j_avg_tr = np.mean(true_range[j_start:j_end])
                        if j_avg_tr > 0:
                            j_plus_di = 100 * (j_avg_plus_dm / j_avg_tr)
                            j_minus_di = 100 * (j_avg_minus_dm / j_avg_tr)
                            j_di_sum = j_plus_di + j_minus_di
                            if j_di_sum > 0:
                                j_dx = 100 * abs(j_plus_di - j_minus_di) / j_di_sum
                                dx_values.append(j_dx)
                    
                    if len(dx_values) > 0:
                        adx_values[i] = np.mean(dx_values)
    
    return pd.Series(adx_values, index=df.index, dtype=float)

def compute_parabolic_sar(df: pd.DataFrame, af: float = 0.02, max_af: float = 0.2) -> pd.Series:
    """Compute Parabolic SAR"""
    if df.empty:
        return pd.Series([], dtype=float, index=df.index)
    
    high = df['high'].values
    low = df['low'].values
    n = len(df)
    
    # Initialize arrays
    sar = np.zeros(n)
    trend = np.ones(n, dtype=bool)  # True = uptrend, False = downtrend
    ep = np.zeros(n)  # Extreme Point
    acc_factor = np.full(n, af)
    
    # Initial values
    sar[0] = low[0]
    trend[0] = True
    ep[0] = high[0]
    
    for i in range(1, n):
        # Calculate SAR
        sar[i] = sar[i-1] + acc_factor[i-1] * (ep[i-1] - sar[i-1])
        
        # Check for trend reversal
        if trend[i-1]:  # Was uptrend
            if low[i] <= sar[i]:  # Reversal to downtrend
                trend[i] = False
                sar[i] = ep[i-1]  # SAR becomes previous EP
                ep[i] = low[i]    # New EP is current low
                acc_factor[i] = af  # Reset acceleration factor
            else:  # Continue uptrend
                trend[i] = True
                # Adjust SAR to not exceed recent lows
                sar[i] = min(sar[i], low[i-1])
                if i > 1:
                    sar[i] = min(sar[i], low[i-2])
                
                # Update EP and acceleration factor
                if high[i] > ep[i-1]:
                    ep[i] = high[i]
                    acc_factor[i] = min(acc_factor[i-1] + af, max_af)
                else:
                    ep[i] = ep[i-1]
                    acc_factor[i] = acc_factor[i-1]
        else:  # Was downtrend
            if high[i] >= sar[i]:  # Reversal to uptrend
                trend[i] = True
                sar[i] = ep[i-1]  # SAR becomes previous EP
                ep[i] = high[i]   # New EP is current high
                acc_factor[i] = af  # Reset acceleration factor
            else:  # Continue downtrend
                trend[i] = False
                # Adjust SAR to not exceed recent highs
                sar[i] = max(sar[i], high[i-1])
                if i > 1:
                    sar[i] = max(sar[i], high[i-2])
                
                # Update EP and acceleration factor
                if low[i] < ep[i-1]:
                    ep[i] = low[i]
                    acc_factor[i] = min(acc_factor[i-1] + af, max_af)
                else:
                    ep[i] = ep[i-1]
                    acc_factor[i] = acc_factor[i-1]
    
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

# ------------------- Strategy Logic -------------------
def is_capitulation_flush(i: int, df: pd.DataFrame, p: Params) -> bool:
    """Check if bar i represents a capitulation flush"""
    if i >= len(df):
        return False
        
    row = df.iloc[i]
    
    # Check for big red candle
    pct_change = (row['close'] / row['open'] - 1.0) if row['open'] > 0 else 0
    if pct_change > p.flush_pct_threshold:
        return False
    
    # Check if close is well below 5-day average
    start_idx = max(0, i - 4)
    close_5d_avg = df['close'].iloc[start_idx:i+1].mean()
    if close_5d_avg > 0:
        below_avg_pct = (row['close'] - close_5d_avg) / close_5d_avg
        if below_avg_pct > p.below_5day_avg_pct:
            return False
    
    # Check for volume spike
    start_idx = max(0, i - 19)
    vol_20_avg = df['volume'].iloc[start_idx:i+1].mean()
    if vol_20_avg > 0 and row['volume'] < p.volume_spike_mult * vol_20_avg:
        return False
    
    return True

def sar_gap_check(close: float, sar_val: float, atr_val: float, p: Params) -> bool:
    """Check if SAR gap meets criteria"""
    if pd.isna(sar_val) or pd.isna(atr_val) or sar_val <= 0 or atr_val <= 0:
        return False
    
    # Check percentage gap
    gap_pct = (sar_val - close) / sar_val if sar_val > 0 else 0
    if gap_pct < p.sar_gap_pct:
        return False
    
    # Check ATR multiple gap
    atr_gap = (sar_val - close) >= p.sar_atr_mult * atr_val
    if not atr_gap:
        return False
    
    return True

def compute_trigger_price(flush_low: float, p: Params) -> float:
    """Compute the trigger price for entry"""
    return float(flush_low * p.reclaim_mult)

def scan_candidates(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Scan for trade candidates"""
    if df.empty:
        return pd.DataFrame(columns=['flush_low', 'trigger', 'close', 'sar', 'atr', 'adx'])
    
    # Calculate indicators
    atr_values = compute_atr(df, p.atr_period)
    adx_values = compute_adx(df, p.adx_period) 
    sar_values = compute_parabolic_sar(df, af=p.sar_af, max_af=p.sar_max_af)
    
    candidates = []
    
    for i in range(len(df)):
        if is_capitulation_flush(i, df, p):
            row = df.iloc[i]
            sar_val = sar_values.iloc[i]
            atr_val = atr_values.iloc[i] 
            adx_val = adx_values.iloc[i]
            
            # Check SAR gap and ADX threshold
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
    
    if candidates:
        return pd.DataFrame(candidates).set_index('date')
    else:
        return pd.DataFrame(columns=['flush_low', 'trigger', 'close', 'sar', 'atr', 'adx'])

def simulate_trades(df: pd.DataFrame, p: Params, initial_equity: float = 100000.0) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Simulate trading strategy"""
    default_stats = {
        'trades': 0, 
        'win_rate_pct': 0, 
        'avg_return_pct': 0, 
        'sharpe_like': np.nan, 
        'max_drawdown_pct': 0, 
        'final_equity': initial_equity
    }
    
    if df.empty or len(df) < p.adx_period + 1:
        return pd.DataFrame(), pd.DataFrame(), default_stats
    
    # Get candidates
    candidates = scan_candidates(df, p)
    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame(), default_stats
    
    # Calculate SAR for exit signals
    sar_values = compute_parabolic_sar(df, af=p.sar_af, max_af=p.sar_max_af)
    
    equity = initial_equity
    equity_curve = []
    trades = []
    
    for date, signal in candidates.iterrows():
        try:
            signal_idx = df.index.get_loc(date)
            entry_idx = signal_idx + 1
            
            if entry_idx >= len(df):
                continue
                
            entry_bar = df.iloc[entry_idx]
            
            # Check if price triggers entry
            if entry_bar['close'] >= signal['trigger']:
                entry_price = float(entry_bar['close'])
                position_size = equity * p.max_position_pct
                shares = int(position_size / entry_price) if entry_price > 0 else 0
                
                if shares <= 0:
                    continue
                
                # Set stop loss
                stop_price = signal['flush_low'] * (1 - p.stop_loss_pct)
                
                # Find exit
                exit_price = None
                exit_date = None
                exit_reason = "Max Hold"
                
                max_hold_idx = min(len(df) - 1, entry_idx + p.hold_max_bars)
                
                for j in range(entry_idx, max_hold_idx + 1):
                    current_bar = df.iloc[j]
                    
                    # Check stop loss
                    if current_bar['low'] <= stop_price:
                        exit_price = stop_price
                        exit_date = df.index[j]
                        exit_reason = "Stop Loss"
                        break
                    
                    # Check SAR exit
                    if j < len(sar_values) and not pd.isna(sar_values.iloc[j]):
                        if current_bar['high'] >= sar_values.iloc[j]:
                            exit_price = float(current_bar['close'])
                            exit_date = df.index[j]
                            exit_reason = "SAR Exit"
                            break
                
                # Default exit if no condition met
                if exit_price is None:
                    exit_price = float(df.iloc[max_hold_idx]['close'])
                    exit_date = df.index[max_hold_idx]
                
                # Calculate trade results
                pnl = (exit_price - entry_price) * shares
                return_pct = 100 * pnl / (entry_price * shares) if entry_price * shares > 0 else 0
                equity += pnl
                
                trades.append({
                    'entry_date': df.index[entry_idx],
                    'entry_price': entry_price,
                    'exit_date': exit_date, 
                    'exit_price': exit_price,
                    'shares': shares,
                    'pnl': pnl,
                    'return_pct': return_pct,
                    'exit_reason': exit_reason
                })
                
        except Exception as e:
            st.warning(f"Error processing trade at {date}: {e}")
            continue
    
    # Create results DataFrames
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    equity_curve = [{'date': df.index[0], 'equity': initial_equity}]  # Start with initial equity
    
    # Calculate equity curve
    current_equity = initial_equity
    for trade in trades:
        current_equity += trade['pnl'] 
        equity_curve.append({'date': trade['exit_date'], 'equity': current_equity})
    
    curve_df = pd.DataFrame(equity_curve).set_index('date') if equity_curve else pd.DataFrame()
    
    # Calculate statistics
    if len(trades_df) > 0:
        total_trades = len(trades_df)
        winning_trades = (trades_df['pnl'] > 0).sum()
        win_rate_pct = 100 * winning_trades / total_trades
        avg_return_pct = trades_df['return_pct'].mean()
        
        # Calculate Sharpe-like ratio
        if len(curve_df) > 1:
            returns = curve_df['equity'].pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                sharpe_like = np.sqrt(252) * returns.mean() / returns.std()
            else:
                sharpe_like = np.nan
        else:
            sharpe_like = np.nan
        
        # Calculate max drawdown
        if len(curve_df) > 0:
            equity_series = curve_df['equity']
            rolling_max = equity_series.cummax()
            drawdown = (equity_series - rolling_max) / rolling_max
            max_drawdown_pct = 100 * drawdown.min() if len(drawdown) > 0 else 0
        else:
            max_drawdown_pct = 0
        
        stats = {
            'trades': total_trades,
            'win_rate_pct': win_rate_pct,
            'avg_return_pct': avg_return_pct,
            'sharpe_like': sharpe_like,
            'max_drawdown_pct': max_drawdown_pct,
            'final_equity': equity
        }
    else:
        stats = default_stats
        
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
        """Load and validate CSV file"""
        df = pd.read_csv(f)
        df.columns = [c.lower() for c in df.columns]
        
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date')
        
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_columns] = df[numeric_columns].astype(float)
        
        return df[numeric_columns]

    files = []
    if upload_mode == "Single CSV" and uploaded:
        files = [uploaded]
    elif upload_mode == "Multiple CSVs" and uploaded:
        files = uploaded
    elif upload_mode == "Upload ZIP of CSVs" and uploaded:
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
            else:
                df = load_csv_file(item)
                symbol = os.path.splitext(item.name)[0].upper()
            
            datasets.append((symbol, df))
            
        except Exception as e:
            st.error(f"Failed to load file: {e}")

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
                
                # Rename columns to lowercase
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                datasets.append((ticker, df))
                
            except Exception as e:
                st.error(f"Failed to fetch {ticker}: {e}")

# ------------------- Run backtest -------------------
if datasets:
    st.header("Backtest Results")
    all_trades = []
    
    for symbol, df in datasets:
        with st.spinner(f"Running backtest for {symbol}..."):
            try:
                trades_df, curve_df, stats = simulate_trades(df, p, initial_equity)
                
                if stats['trades'] == 0:
                    st.warning(f"No trades generated for {symbol}")
                    continue
                
                # Add symbol to trades
                trades_df = trades_df.copy()
                trades_df['symbol'] = symbol
                all_trades.append(trades_df)
                
                # Display results
                st.subheader(f"Results for {symbol}")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Trades", stats['trades'])
                with col2:
                    st.metric("Win Rate", f"{stats['win_rate_pct']:.1f}%")
                with col3:
                    st.metric("Avg Return", f"{stats['avg_return_pct']:.2f}%")
                with col4:
                    st.metric("Final Equity", f"${stats['final_equity']:,.0f}")
                
                # Create chart
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot price and SAR
                ax.plot(df.index, df['close'], label='Close Price', linewidth=1.5)
                sar_values = compute_parabolic_sar(df, p.sar_af, p.sar_max_af)
                ax.plot(df.index, sar_values, '--', alpha=0.7, label='Parabolic SAR')
                
                # Plot trades
                if not trades_df.empty:
                    ax.scatter(trades_df['entry_date'], trades_df['entry_price'], 
                              marker='^', color='green', s=60, label='Entry', zorder=5)
                    ax.scatter(trades_df['exit_date'], trades_df['exit_price'], 
                              marker='v', color='red', s=60, label='Exit', zorder=5)
                
                ax.set_title(f"{symbol} - Price Chart with Trade Signals")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price ($)")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show trade details
                if not trades_df.empty:
                    st.subheader(f"Trade Details for {symbol}")
                    display_df = trades_df[['entry_date', 'entry_price', 'exit_date', 'exit_price', 'return_pct', 'exit_reason']].copy()
                    display_df['entry_date'] = display_df['entry_date'].dt.strftime('%Y-%m-%d')
                    display_df['exit_date'] = display_df['exit_date'].dt.strftime('%Y-%m-%d')
                    st.dataframe(display_df, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error running backtest for {symbol}: {e}")
    
    # Combined results
    if all_trades:
        st.header("Combined Results")
        combined_df = pd.concat(all_trades, ignore_index=True)
        
        total_trades = len(combined_df)
        total_wins = (combined_df['pnl'] > 0).sum()
        overall_win_rate = 100 * total_wins / total_trades if total_trades > 0 else 0
        overall_avg_return = combined_df['return_pct'].mean()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Trades", total_trades)
        with col2:
            st.metric("Overall Win Rate", f"{overall_win_rate:.1f}%")
        with col3:
            st.metric("Overall Avg Return", f"{overall_avg_return:.2f}%")
        
        st.subheader("All Trades")
        display_combined = combined_df[['symbol', 'entry_date', 'entry_price', 'exit_date', 'exit_price', 'return_pct', 'exit_reason']].copy()
        display_combined['entry_date'] = display_combined['entry_date'].dt.strftime('%Y-%m-%d')
        display_combined['exit_date'] = display_combined['exit_date'].dt.strftime('%Y-%m-%d')
        st.dataframe(display_combined, use_container_width=True)
        
        # Download button
        csv_buffer = io.StringIO()
        combined_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download All Trades as CSV",
            data=csv_buffer.getvalue(),
            file_name="sar_reversion_trades.csv",
            mime="text/csv"
        )
