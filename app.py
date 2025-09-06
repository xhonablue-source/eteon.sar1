# ------------------- UI -------------------
st.title("🏛️ Eteon Capital")
st.header("SAR Reversion Spike Backtest")
st.markdown("Upload OHLCV data or enter tickers to fetch automatically.")

# Locked proprietary parameters
p = Params(
    flush_pct_threshold=-0.25,
    below_5day_avg_pct=-0.40,
    volume_spike_mult=3.0,
    sar_gap_pct=0.20,
    sar_atr_mult=2.0,
    adx_threshold=25.0,
    reclaim_mult=1.05,
    stop_loss_pct=0.08,
    max_position_pct=0.05,
    atr_period=14,
    adx_period=14,
    sar_af=0.02,
    sar_max_af=0.2,
    hold_max_bars=60
)

initial_equity = 100000.0  # Fixed for consistency

with st.sidebar:
    st.header("Parameters (Locked)")
    st.markdown(
        """
        ⚙️ **Proprietary Strategy Parameters**  
        - Locked and non-editable  
        - Optimized internally (including BSLK detection)  
        - Shown here for transparency only  
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="color:gray;">All parameters are fixed and cannot be changed.</p>',
        unsafe_allow_html=True
    )
