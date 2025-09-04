
{*************************************
 SAR Reversion Spike — EasyLanguage
 Long-only strategy for equities (daily)
**************************************}

Inputs:
    // Detection thresholds
    FlushPctThreshold(-0.25),         // (Close/Open - 1) <= this (e.g. -0.25 = -25%)
    Below5DayAvgPct(-0.40),           // (Close - Avg(Close,5))/Avg(Close,5) <= this
    VolumeSpikeMult(3.0),             // Volume >= mult * Avg(Volume,20)

    // SAR distance checks
    SARGapPct(0.20),                  // (SAR - Close)/SAR >= this
    SAR_ATR_Mult(2.0),                // (SAR - Close) >= ATR * this

    // Confirmations
    ADXThreshold(25),                 // ADX(ADXPeriod) >= this
    ATRPeriod(14),
    ADXPeriod(14),
    SAR_AF(0.02),                     // Parabolic SAR acceleration factor (step)
    SAR_MaxAF(0.20),                  // Parabolic SAR maximum

    // Entry/Exit
    ReclaimMult(1.05),                // Trigger = Low * ReclaimMult
    StopLossPct(0.08),                // Hard stop below flush low (e.g., 0.08 = 8%)
    HoldMaxBars(60),                  // Max holding period (bars)

    // Sizing
    MaxPositionPct(0.05);             // % of equity to allocate per trade

Vars:
    ATR(0), ADXv(0), SARv(0),
    Vol20(0), Avg5Close(0),
    FlushBar(False),
    SARGapOK(False), ConfirmOK(False),
    TriggerPrice(0), FlushLow(0), StopPrice(0),
    Shares(0), EntryPending(False),
    BarsInTrade(0);

// --- Precompute indicators ---
ATR  = AvgTrueRange(ATRPeriod);
ADXv = ADX(ADXPeriod);
SARv = ParabolicSAR(SAR_AF, SAR_MaxAF);
Vol20 = Average(Volume, 20);
Avg5Close = Average(Close, 5);

// --- Detect Capitulation Flush (at bar close) ---
FlushBar = False;
If Vol20 > 0 then begin
    If ((Close / Open) - 1) <= FlushPctThreshold and
       ((Close - Avg5Close) / Avg5Close) <= Below5DayAvgPct and
       Volume >= VolumeSpikeMult * Vol20 then
    begin
        FlushBar = True;
    end;
end;

// --- SAR distance checks ---
SARGapOK = False;
If SARv > 0 and ATR > 0 then begin
    If ((SARv - Close) / SARv) >= SARGapPct and
       (SARv - Close) >= SAR_ATR_Mult * ATR then
        SARGapOK = True;
end;

// --- Confirmation (trend/vol) ---
ConfirmOK = (ADXv >= ADXThreshold);

// --- If a valid signal on this bar, arm a trigger for next bar ---
If FlushBar and SARGapOK and ConfirmOK and MarketPosition = 0 then begin
    TriggerPrice = Low * ReclaimMult;
    FlushLow     = Low;
    StopPrice    = FlushLow * (1 - StopLossPct);
    EntryPending = True;
end;

// --- Entry: next bar when Close >= TriggerPrice ---
If EntryPending and MarketPosition = 0 then begin
    If Close >= TriggerPrice then begin
        { sizing by % of equity }
        If Close > 0 then
            Shares = IntPortion( (GetAppInfo(aiStrategyEquity) * MaxPositionPct) / Close );
        If Shares < 1 then Shares = 1;

        SetPositionSize(Shares);
        Buy Next Bar at Market;

        EntryPending = False;
        BarsInTrade = 0;
    end;
    { Disarm after N bars without trigger (optional): }
    If BarsSinceEntry = 0 and BarsInTrade >= 5 then EntryPending = False;
end;

// --- In-trade management ---
If MarketPosition = 1 then begin
    BarsInTrade = BarsInTrade + 1;

    { 1) Hard stop below the original flush low (percentage) }
    StopPrice = FlushLow * (1 - StopLossPct);
    Sell Next Bar at Stop StopPrice;

    { 2) Exit when price tags/reclaims SAR (regression anchor) }
    If High >= SARv then
        Sell Next Bar at Market;

    { 3) Max holding period }
    If BarsInTrade >= HoldMaxBars then
        Sell Next Bar at Market;
end;

// --- Plotting (optional) ---
Plot1(SARv, "ParabolicSAR");
Plot2(TriggerPrice, "Trigger");
