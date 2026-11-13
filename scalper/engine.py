"""Backtest engine for the gold scalper.

Signals are read at the close of an M5 bar and filled at the open of the next
M1 bar.  Prices in the data are BID; the ask is bid + the spread recorded from
real ticks, so every trade pays the real spread:

  long : buy at ask, stop/target are checked against bid
  short: sell at bid, stop/target are checked against ask

If a stop and a target are both inside the same M1 bar, the stop is assumed to
have been hit first (the pessimistic reading).
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

CONTRACT = 100.0   # ounces per 1.00 lot
LOT_STEP = 0.01


@dataclass
class Account:
    start_balance: float = 300.0
    risk_pct: float = 2.0            # % of equity risked at the stop per trade
    max_risk_pct: float = 4.0        # skip the trade if even 0.01 lot risks more
    max_lots: float = 0.10
    leverage: float = 500.0          # used only for the margin check
    daily_target: float = 10.0       # stop trading for the day at +$ (0 = off)
    daily_loss_pct: float = 5.0      # stop trading for the day at -% of balance
    max_trades_per_day: int = 6
    commission_per_lot: float = 0.0  # $ per lot per side
    slippage: float = 0.05           # $ per ounce, on entries and on stops
    max_spread: float = 1.2          # no entry when the spread is wider
    compounding: bool = False        # False: size from start balance


@dataclass
class Result:
    trades: pd.DataFrame
    daily: pd.Series
    equity: pd.Series
    stats: dict = field(default_factory=dict)


def load_m1(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col='time', parse_dates=True)
    return df


def to_m5(m1: pd.DataFrame) -> pd.DataFrame:
    g = m1.resample('5min', label='left', closed='left')
    m5 = pd.DataFrame({
        'open': g['open'].first(), 'high': g['high'].max(),
        'low': g['low'].min(), 'close': g['close'].last(),
        'spread': g['spread_mean'].mean(), 'ticks': g['ticks'].sum(),
    }).dropna(subset=['open'])
    return m5


def run(m1: pd.DataFrame, sig: pd.DataFrame, acc: Account,
        max_hold_min: int = 120, flat_hour_utc: int = 21,
        be_at_r: float = 0.0) -> Result:
    """sig: M5-indexed frame with columns side (+1/-1/0), sl (distance, $),
    tp (distance, $).  A signal on the bar starting at T is known at T+5min."""
    s = sig[sig['side'] != 0]
    known_at = s.index + pd.Timedelta(minutes=5)
    pos_in_m1 = m1.index.searchsorted(known_at, side='left')
    ok = pos_in_m1 < len(m1)
    # the fill bar must come within 2 minutes of the signal (no weekend gaps)
    ok &= (m1.index[np.minimum(pos_in_m1, len(m1) - 1)] - known_at) <= pd.Timedelta(minutes=2)
    entries = dict(zip(pos_in_m1[ok], zip(s['side'].values[ok], s['sl'].values[ok], s['tp'].values[ok])))

    o = m1['open'].values; h = m1['high'].values; l = m1['low'].values; c = m1['close'].values
    sp_o = m1['spread_open'].values; sp_m = m1['spread_mean'].values
    idx = m1.index
    day = idx.normalize().values
    hour = idx.hour.values

    bal = acc.start_balance
    trades = []
    cur_day = None; day_pnl = 0.0; day_n = 0; day_start_bal = bal
    in_pos = False
    side = 0; entry = sl = tp = lots = 0.0; i_in = 0; risk_d = 0.0; risk_dist = 0.0; moved_be = False

    def close(i, price, why):
        nonlocal bal, in_pos, day_pnl
        pnl = side * (price - entry) * lots * CONTRACT - 2 * acc.commission_per_lot * lots
        bal += pnl; day_pnl += pnl
        trades.append((idx[i_in], idx[i], side, lots, entry, price, pnl, why, risk_d, bal))
        in_pos = False

    for i in range(len(m1)):
        if day[i] != cur_day:
            cur_day = day[i]; day_pnl = 0.0; day_n = 0; day_start_bal = bal
        if in_pos:
            if side == 1:
                if o[i] <= sl:                       # gapped through the stop
                    close(i, o[i] - acc.slippage, 'sl')
                elif l[i] <= sl:
                    close(i, sl - acc.slippage, 'sl' if not moved_be else 'be')
                elif h[i] >= tp:
                    close(i, tp, 'tp')
            else:
                a_o = o[i] + sp_o[i]; a_h = h[i] + sp_m[i]; a_l = l[i] + sp_m[i]
                if a_o >= sl:
                    close(i, a_o + acc.slippage, 'sl')
                elif a_h >= sl:
                    close(i, sl + acc.slippage, 'sl' if not moved_be else 'be')
                elif a_l <= tp:
                    close(i, tp, 'tp')
            if in_pos:
                exit_px = c[i] if side == 1 else c[i] + sp_m[i]
                if be_at_r > 0 and not moved_be and side * (exit_px - entry) >= be_at_r * risk_dist:
                    sl = entry + side * 0.10; moved_be = True
                if (i - i_in) >= max_hold_min:
                    close(i, exit_px, 'time')
                elif hour[i] >= flat_hour_utc:
                    close(i, exit_px, 'eod')
                elif i + 1 < len(m1) and (idx[i + 1] - idx[i]) > pd.Timedelta(minutes=30):
                    close(i, exit_px, 'gap')
            continue
        e = entries.get(i)
        if e is None:
            continue
        if bal <= 0:
            break
        if acc.daily_target > 0 and day_pnl >= acc.daily_target:
            continue
        if day_pnl <= -acc.daily_loss_pct / 100.0 * day_start_bal:
            continue
        if day_n >= acc.max_trades_per_day or sp_o[i] > acc.max_spread or hour[i] >= flat_hour_utc:
            continue
        sd, sl_d, tp_d = e
        if not (sl_d > 0 and tp_d > 0):
            continue
        base = bal if acc.compounding else min(bal, acc.start_balance)
        # the real loss at the stop includes the spread and the slippage
        loss_per_oz = sl_d + acc.slippage
        lots_ = np.floor(base * acc.risk_pct / 100.0 / (loss_per_oz * CONTRACT) / LOT_STEP + 1e-9) * LOT_STEP
        if lots_ < LOT_STEP:
            if LOT_STEP * loss_per_oz * CONTRACT > base * acc.max_risk_pct / 100.0:
                continue
            lots_ = LOT_STEP
        lots_ = min(lots_, acc.max_lots)
        px = o[i] + sp_o[i] + acc.slippage if sd == 1 else o[i] - acc.slippage
        if lots_ * CONTRACT * px / acc.leverage > 0.5 * bal:     # margin check
            continue
        side = int(sd); lots = round(lots_, 2); entry = px
        # stop and target distances are measured from the mid price at entry,
        # so the spread is a real cost and not hidden inside the stop
        mid = o[i] + sp_o[i] / 2.0
        sl = mid - side * sl_d; tp = mid + side * tp_d
        risk_dist = abs(entry - sl)
        risk_d = risk_dist * lots * CONTRACT
        i_in = i; in_pos = True; moved_be = False; day_n += 1

    cols = ['t_in', 't_out', 'side', 'lots', 'entry', 'exit', 'pnl', 'why', 'risk', 'balance']
    tr = pd.DataFrame(trades, columns=cols)
    return summarize(tr, m1, acc)


def summarize(tr: pd.DataFrame, m1: pd.DataFrame, acc: Account) -> Result:
    all_days = pd.DatetimeIndex(sorted(set(m1.index.normalize())))
    all_days = all_days[all_days.dayofweek < 5]
    if len(tr):
        daily = tr.groupby(tr['t_out'].dt.normalize())['pnl'].sum().reindex(all_days, fill_value=0.0)
    else:
        daily = pd.Series(0.0, index=all_days)
    equity = acc.start_balance + daily.cumsum()
    st = {'trades': len(tr), 'days': len(daily)}
    if len(tr):
        w = tr['pnl'] > 0
        gp = tr.loc[w, 'pnl'].sum(); gl = -tr.loc[~w, 'pnl'].sum()
        peak = equity.cummax()
        st.update({
            'net': tr['pnl'].sum(),
            'win_rate': w.mean(),
            'profit_factor': gp / gl if gl > 0 else np.inf,
            'avg_win': tr.loc[w, 'pnl'].mean() if w.any() else 0.0,
            'avg_loss': tr.loc[~w, 'pnl'].mean() if (~w).any() else 0.0,
            'avg_trade': tr['pnl'].mean(),
            'avg_per_day': daily.mean(),
            'median_day': daily.median(),
            'green_days': (daily > 0).mean(),
            'red_days': (daily < 0).mean(),
            'days_hit_target': (daily >= acc.daily_target).mean() if acc.daily_target > 0 else np.nan,
            'worst_day': daily.min(), 'best_day': daily.max(),
            'max_dd': (peak - equity).max(),
            'max_dd_pct': ((peak - equity) / peak).max(),
            'min_equity': equity.min(),
            'final': equity.iloc[-1],
            'trades_per_day': len(tr) / len(daily),
            'avg_lots': tr['lots'].mean(),
        })
    return Result(tr, daily, equity, st)
