#1.DFA(Detrended Fluctuation Analysis)-->Original for measurement of long-range correlations in time series
#2.Using DFA we are analyzing the trend-following and mean-reversion

#DFA can be used as a Regime detection tool:-
#3.a<0.5-->mean-reverting
#4.a=0.5-->uncorrelated
#5.a>0.5-->trend-following

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

# --------------------- User parameters ---------------------
CSV_PATH          = r"C:/Users/Mritunjay Maddhesiya/OneDrive/Desktop/MT5/XAUUSDm_1D.csv"
TIMEFRAME_MINUTES = 5                  # <-- set to 5 for 5-minute bars
ROLLING_DAYS      = 30                 # rolling window length in days for DFA
ALPHA_ENTER       = 0.60               # enter long when alpha > ALPHA_ENTER
STOP_LOSS_PCT     = 0.05               # 5% stop loss
TAKE_PROFIT_PCT   = 0.10               # 10% take profit
MIN_SCALE         = 4                  # minimal scale window for DFA
PLOT_RESULTS      = True
# -----------------------------------------------------------

# ---------- DFA implementation (robust) --------------------
def cumulative_sum(x):
    return np.cumsum(x - np.nanmean(x))

def calc_rms(x, scale):
    # x is 1D array; scale must be int and >=2
    scale = int(scale)
    if scale < 2:
        return np.array([])
    n = x.shape[0] // scale
    if n < 1:
        return np.array([])
    trimmed = x[: n * scale]
    X = trimmed.reshape(n, scale)
    scale_ax = np.arange(scale)
    rms = np.zeros(n)
    for i, xcut in enumerate(X):
        # linear detrend
        coeff = np.polyfit(scale_ax, xcut, 1)
        fit = np.polyval(coeff, scale_ax)
        rms[i] = np.sqrt(np.mean((xcut - fit) ** 2))
    return rms

def fluctuation(y, scales):
    # scales: iterable of ints
    out = []
    for scale in scales:
        rms_vals = calc_rms(y, scale)
        if rms_vals.size == 0:
            out.append(np.nan)
        else:
            out.append(np.sqrt(np.mean(rms_vals ** 2)))
    return np.array(out)

def dfa(x, scale_lim=[1,3], scale_dens=0.25, plot=False):
    x = np.asarray(x)
    if x.size < MIN_SCALE:
        return np.nan

    y = cumulative_sum(x)
    # create logspaced scales and convert to int; unique and filter
    raw = np.logspace(scale_lim[0], scale_lim[1],
                      num=max(3, int((scale_lim[1] - scale_lim[0]) / scale_dens)))
    scales = np.unique(np.array(raw).astype(int))
    scales = scales[(scales >= MIN_SCALE) & (scales <= (y.size // 2))]  # reasonable upper bound
    if scales.size == 0:
        return np.nan

    fluct = fluctuation(y, scales)
    # remove NaNs
    mask = np.isfinite(fluct) & (fluct > 0)
    if np.sum(mask) < 2:
        return np.nan

    sx = np.log10(scales[mask])
    sy = np.log10(fluct[mask])
    coeff = np.polyfit(sx, sy, 1)  # slope is alpha
    alpha = coeff[0]

    if plot:
        plt.figure(figsize=(6,4))
        plt.plot(sx, sy, 'o', label='Data')
        plt.plot(sx, np.polyval(coeff, sx), 'r', label=f'Fit: a={alpha:.3f}')
        plt.xlabel('log10(scale)')
        plt.ylabel('log10(F(scale))')
        plt.legend()
        plt.show()

    return alpha
# -----------------------------------------------------------

# ---------- Helper: performance metrics --------------------
def cagr_from_equity(equity_series, periods_per_year):
    start = equity_series.iloc[0]
    end = equity_series.iloc[-1]
    years = len(equity_series) / periods_per_year
    if start <= 0 or years <= 0:
        return np.nan
    return (end / start) ** (1.0 / years) - 1.0

def sharpe_annualized(returns_series, periods_per_year):
    # uses arithmetic mean excess returns (risk-free assumed 0)
    mu = returns_series.mean()
    sigma = returns_series.std(ddof=1)
    if sigma == 0 or np.isnan(mu) or np.isnan(sigma):
        return np.nan
    return (mu / sigma) * np.sqrt(periods_per_year)

def max_drawdown(equity_series):
    roll_max = equity_series.cummax()
    drawdown = (equity_series / roll_max) - 1.0
    return drawdown.min()
# -----------------------------------------------------------

# ----------------- Backtesting engine ----------------------
def backtest(df, alpha_series, entry_threshold=ALPHA_ENTER,
             stop_loss=STOP_LOSS_PCT, take_profit=TAKE_PROFIT_PCT,
             timeframe_minutes=TIMEFRAME_MINUTES):
    prices = df['close'].values
    has_HL = ('high' in df.columns) and ('low' in df.columns)
    highs = df['high'].values if has_HL else None
    lows  = df['low'].values if has_HL else None

    cash = 1.0          # start with 1 unit capital
    position = 0        # 0 or 1 (long)
    entry_price = None
    entry_idx = None

    equity_curve = []
    positions = []      # 0/1 per bar for plotting
    trades = []

    for i in range(len(df)):
        alpha = alpha_series.iloc[i] if i < len(alpha_series) else np.nan
        price = prices[i]

        # If currently no position, check entry condition: alpha > threshold
        if position == 0:
            if np.isfinite(alpha) and alpha > entry_threshold:
                # enter at next bar's open/close approximation: we use close as execution price
                entry_price = price
                entry_idx = i
                position = 1
                # record trade starter
                trade = {
                    'entry_idx': i,
                    'entry_price': entry_price,
                    'exit_idx': None,
                    'exit_price': None,
                    'pnl': None,
                    'status': 'open'
                }
            else:
                trade = None
        else:
            trade = trades[-1] if trades else None

        # If position open => check stop loss / take profit on this bar
        if position == 1:
            sl = entry_price * (1 - stop_loss)
            tp = entry_price * (1 + take_profit)

            exited = False
            exit_price = None
            exit_idx = i

            if has_HL:
                # If both SL and TP occur the same bar, conservative rule: SL triggers first
                # Check low first (SL), then high (TP)
                if lows[i] <= sl:
                    exit_price = sl
                    exited = True
                elif highs[i] >= tp:
                    exit_price = tp
                    exited = True
            else:
                # fallback: use close price checks - conservative: check close vs SL then TP
                if price <= sl:
                    exit_price = price
                    exited = True
                elif price >= tp:
                    exit_price = price
                    exited = True

            if exited:
                position = 0
                pnl = (exit_price / entry_price) - 1.0
                if trade is None:
                    trade = {}
                trade.update({
                    'exit_idx': exit_idx,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'status': 'closed'
                })
                trades.append(trade)
                entry_price = None
                entry_idx = None

        # If we created a trade entry this bar and it hasn't been appended yet, append it.
        if position == 1 and (not trades or trades[-1].get('status') != 'open'):
            # append open trade
            trades.append({
                'entry_idx': entry_idx,
                'entry_price': entry_price,
                'exit_idx': None,
                'exit_price': None,
                'pnl': None,
                'status': 'open'
            })

        # compute equity value this bar (mark-to-market)
        if position == 1:
            # mark to market at close price
            current_value = cash * (prices[i] / trades[-1]['entry_price'])
        else:
            current_value = cash
        equity_curve.append(current_value)
        positions.append(position)

    # Close any open position at last close (force exit)
    if trades and trades[-1]['status'] == 'open':
        last = trades[-1]
        exit_price = prices[-1]
        pnl = (exit_price / last['entry_price']) - 1.0
        last.update({'exit_idx': len(df)-1, 'exit_price': exit_price, 'pnl': pnl, 'status': 'closed'})
        equity_curve[-1] = cash * (1 + pnl)

    eq = pd.Series(equity_curve, index=df.index)
    pos_series = pd.Series(positions, index=df.index)
    return trades, eq, pos_series
# -----------------------------------------------------------

# ---------------- Main routine -----------------------------
def run():
    # load CSV - tries to parse common datetime column names
    df = pd.read_csv(CSV_PATH)
    # try detect datetime column
    date_cols = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower() or 'datetime' in c.lower()]
    if len(date_cols) == 0:
        raise ValueError("No datetime column found in CSV. Rename column to include 'time' or 'date' or 'datetime'.")
    dt_col = date_cols[0]
    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.set_index(dt_col).sort_index()

    # ensure close present
    if 'close' not in df.columns:
        raise ValueError("CSV must contain 'close' column.")
    # drop rows with NaN close
    df = df.dropna(subset=['close'])
    df['close'] = df['close'].astype(float)

    # compute returns for DFA: percentage returns (log or pct). Using pct_change
    returns = df['close'].pct_change().fillna(0).values

    # rolling DFA window (convert days -> number of bars)
    periods_per_day = (24 * 60) / TIMEFRAME_MINUTES
    rolling_bars = int(max(50, ROLLING_DAYS * periods_per_day))  # ensure at least 50 bars min

    print(f"Data rows: {len(df)}, rolling_bars: {rolling_bars} (approx {ROLLING_DAYS} days)")

    # Pre-allocate alpha series
    alphas = pd.Series(index=df.index, dtype=float)

    # choose scale limits: for short intraday we use scales [log10(10), log10(1000)] by default
    scale_lim = [np.log10(10), np.log10(max(32, min(2000, rolling_bars // 4)))]

    # compute rolling DFA (this can be a bit slow for large datasets)
    returns_series = pd.Series(returns, index=df.index)
    total = len(df)
    for i in range(total):
        # window: last rolling_bars ending at i (inclusive)
        if i + 1 < 50:
            alphas.iloc[i] = np.nan
            continue
        start = max(0, i + 1 - rolling_bars)
        window = returns_series.iloc[start:i+1].values
        a = dfa(window, scale_lim=scale_lim, scale_dens=0.25, plot=False)
        alphas.iloc[i] = a
        # optional: progress print every N
        if (i % 2000) == 0 and i > 0:
            print(f"Computed DFA up to {i}/{total}")

    # Backtest
    trades, equity, positions = backtest(df, alphas, entry_threshold=ALPHA_ENTER,
                                         stop_loss=STOP_LOSS_PCT, take_profit=TAKE_PROFIT_PCT,
                                         timeframe_minutes=TIMEFRAME_MINUTES)

    # Performance metrics
    periods_per_year = periods_per_day * 252  # approximate trading days = 252
    buyhold_returns = df['close'] / df['close'].iloc[0]
    buyhold_eq = buyhold_returns  # starting at 1

    cagr = cagr_from_equity(equity, periods_per_year)
    # compute per-period returns from equity
    eq_returns = equity.pct_change().fillna(0)
    sharpe = sharpe_annualized(eq_returns, periods_per_year)
    mdd = max_drawdown(equity)

    buy_cagr = cagr_from_equity(buyhold_eq, periods_per_year)
    buy_returns = buyhold_eq.pct_change().fillna(0)
    buy_sharpe = sharpe_annualized(buy_returns, periods_per_year)
    buy_mdd = max_drawdown(buyhold_eq)

    print("\n--- Strategy Performance ---")
    print(f"Final equity: {equity.iloc[-1]:.4f}")
    print(f"CAGR: {cagr:.2%}")
    print(f"Annualized Sharpe: {sharpe:.3f}")
    print(f"Max Drawdown: {mdd:.2%}")
    print(f"Number of trades: {len([t for t in trades if t.get('status')=='closed'])}")

    print("\n--- Buy & Hold ---")
    print(f"Final value: {buyhold_eq.iloc[-1]:.4f}")
    print(f"CAGR: {buy_cagr:.2%}")
    print(f"Annualized Sharpe: {buy_sharpe:.3f}")
    print(f"Max Drawdown: {buy_mdd:.2%}")

    # Per-trade PnL
    closed_trades = [t for t in trades if t.get('status') == 'closed' and t.get('pnl') is not None]
    trade_pnls = np.array([t['pnl'] for t in closed_trades])

    # plots
    if PLOT_RESULTS:
        fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        ax[0].plot(df.index, df['close'], label='Close')
        ax[0].set_title('Price')
        # mark entries/exits
        for t in closed_trades:
            ax[0].axvline(df.index[t['entry_idx']], color='g', alpha=0.3)
            ax[0].axvline(df.index[t['exit_idx']], color='r', alpha=0.3)

        ax[1].plot(df.index, alphas, label='DFA alpha')
        ax[1].axhline(ALPHA_ENTER, color='k', linestyle='--', label=f'Entry thresh {ALPHA_ENTER}')
        ax[1].set_ylabel('alpha')
        ax[1].legend()

        ax[2].plot(df.index, equity, label='Strategy Equity')
        ax[2].plot(df.index, buyhold_eq, label='Buy & Hold', alpha=0.7)
        ax[2].set_title('Equity Curve')
        ax[2].legend()

        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8,4))
        plt.hist(trade_pnls, bins=30)
        plt.title('Trade PnL distribution')
        plt.xlabel('PnL (fraction)')
        plt.show()

    # return objects for further analysis if needed
    return {
        'df': df,
        'alphas': alphas,
        'trades': trades,
        'equity': equity,
        'positions': positions,
        'metrics': {
            'strategy': {'CAGR': cagr, 'Sharpe': sharpe, 'MDD': mdd},
            'buyhold': {'CAGR': buy_cagr, 'Sharpe': buy_sharpe, 'MDD': buy_mdd}
        }
    }

# ---------------- run when executed -------------------------
if __name__ == "__main__":
    res = run()
