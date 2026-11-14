"""Multi-timeframe Smart-Money setup for gold (4H -> 1H -> 5m).

  4H : market direction (break of swing structure), key levels (last 4H swing
       high / low = the dealing range, used for the premium / discount filter)
  1H : trend (must agree with 4H), zones to trade from: Fair Value Gaps and
       Order Blocks left by a displacement, liquidity (last 1H swing = target)
  5m : confirmation.  After price trades into a 1H zone, wait for a 5m change
       of character: a body close through the last 5m swing in the trade
       direction.  The stop goes behind the extreme made inside the zone.

No look-ahead: a 1H / 4H bar is used only after it has closed, a swing only
after `k` bars have confirmed it, and the signal on the 5m bar starting at T
uses data up to the close of that bar (filled on the next bar's open).
"""
import numpy as np
import pandas as pd


def atr(df, n=14):
    pc = df['close'].shift(1)
    tr = pd.concat([df['high'] - df['low'], (df['high'] - pc).abs(), (df['low'] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()


def resample(m5, rule):
    g = m5.resample(rule, label='left', closed='left')
    return pd.DataFrame({'open': g['open'].first(), 'high': g['high'].max(),
                         'low': g['low'].min(), 'close': g['close'].last()}).dropna()


def structure(df, k):
    """Swing structure.  Returns, as of the close of each bar: trend (+1 / -1 /
    0), the last confirmed swing high and swing low."""
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    n = len(df)
    trend = np.zeros(n, dtype=int); sh = np.full(n, np.nan); sl = np.full(n, np.nan)
    cur_t = 0; cur_sh = cur_sl = np.nan; sh_live = sl_live = False
    for i in range(n):
        if i >= 2 * k:
            p = i - k
            if h[p] == h[i - 2 * k:i + 1].max():
                cur_sh = h[p]; sh_live = True
            if l[p] == l[i - 2 * k:i + 1].min():
                cur_sl = l[p]; sl_live = True
        if sh_live and c[i] > cur_sh:
            cur_t = 1; sh_live = False
        elif sl_live and c[i] < cur_sl:
            cur_t = -1; sl_live = False
        trend[i] = cur_t; sh[i] = cur_sh; sl[i] = cur_sl
    return trend, sh, sl


def zones_1h(h1, fvg_min_atr=0.3, ob_look=4, ob_max_atr=3.0, use_fvg=True, use_ob=True):
    """Fair Value Gaps and Order Blocks on closed 1H bars.
    Bullish FVG at bar i: low[i] > high[i-2]  -> zone [high[i-2], low[i]].
    Bullish OB: the last down candle before that displacement, if the
    displacement closed above its high -> zone [low, high] of that candle."""
    o, h, l, c = (h1[x].values for x in ('open', 'high', 'low', 'close'))
    a = atr(h1).values
    close_t = h1.index + pd.Timedelta(hours=1)
    out = []
    for i in range(15, len(h1)):
        av = a[i - 1]
        for d in (1, -1):
            gap = l[i] - h[i - 2] if d == 1 else l[i - 2] - h[i]
            if gap < fvg_min_atr * av:
                continue
            if use_fvg:
                top, bot = (l[i], h[i - 2]) if d == 1 else (l[i - 2], h[i])
                out.append({'dir': d, 'kind': 'FVG', 'top': top, 'bot': bot, 't': close_t[i]})
            if use_ob:
                for j in range(i - 2, max(i - 2 - ob_look, 0), -1):
                    if d * (c[j] - o[j]) < 0:                       # opposite candle
                        broke = c[i - 1] > h[j] if d == 1 else c[i - 1] < l[j]
                        if broke and h[j] - l[j] <= ob_max_atr * av:
                            out.append({'dir': d, 'kind': 'OB', 'top': h[j], 'bot': l[j], 't': close_t[i]})
                        break
    return out


def mtf_smc(m5, k4=2, k1=2, k5=2, confirm=True, body_atr=0.3, max_wait=24,
            fvg_min_atr=0.3, use_fvg=True, use_ob=True, zone_life_h=48,
            need_h1_trend=True, pd_filter=False, need_sweep=False, sweep_bars=48,
            sl_buf_atr=0.3, min_sl=2.0, max_sl=15.0, rr=2.0, tp_mode='rr',
            start_h=7, end_h=19, details=False):
    """Returns an M5-indexed frame: side (+1 / -1 / 0), sl, tp ($ per ounce)."""
    h4 = resample(m5, '4h'); h1 = resample(m5, '1h')
    t4, sh4, sl4 = structure(h4, k4)
    t1, sh1, sl1 = structure(h1, k1)
    _, sh5, sl5 = structure(m5, k5)
    zs = sorted(zones_1h(h1, fvg_min_atr, use_fvg=use_fvg, use_ob=use_ob), key=lambda z: z['t'])
    o, h, l, c = (m5[x].values for x in ('open', 'high', 'low', 'close'))
    a5 = atr(m5).values
    n = len(m5)
    tc = m5.index + pd.Timedelta(minutes=5)                         # close time of each 5m bar
    # index of the last CLOSED 1H / 4H bar at the close of each 5m bar
    p1 = np.searchsorted(h1.index + pd.Timedelta(hours=1), tc, side='right') - 1
    p4 = np.searchsorted(h4.index + pd.Timedelta(hours=4), tc, side='right') - 1
    hour = m5.index.hour + m5.index.minute / 60.0
    ses = (hour >= start_h) & (hour < end_h)
    life = pd.Timedelta(hours=zone_life_h)

    side = np.zeros(n, dtype=int); sl_d = np.full(n, np.nan); tp_d = np.full(n, np.nan)
    info = [None] * n
    active = []; zi = 0; arm = None
    for j in range(60, n):
        while zi < len(zs) and zs[zi]['t'] <= tc[j]:
            active.append(zs[zi]); zi += 1
        if p4[j] < 0 or p1[j] < 0:
            continue
        # a close through the far side of a zone kills it
        keep = []
        for z in active:
            dead = c[j] < z['bot'] if z['dir'] == 1 else c[j] > z['top']
            if dead or tc[j] - z['t'] > life:
                if arm is not None and arm['z'] is z:
                    arm = None
            else:
                keep.append(z)
        active = keep
        bias = t4[p4[j]]
        ok_dir = bias != 0 and (not need_h1_trend or t1[p1[j]] == bias)
        if pd_filter and ok_dir:                                    # longs in discount, shorts in premium
            mid = (sh4[p4[j]] + sl4[p4[j]]) / 2.0
            ok_dir = (c[j] < mid) if bias == 1 else (c[j] > mid)
        if arm is not None and arm['z']['dir'] != bias:
            arm = None
        if ok_dir and ses[j]:
            for z in active:
                if z['dir'] != bias:
                    continue
                hit = l[j] <= z['top'] if bias == 1 else h[j] >= z['bot']
                if hit and (arm is None or (z is not arm['z'] and
                            (z['bot'] < arm['z']['bot'] if bias == 1 else z['top'] > arm['z']['top']))):
                    arm = {'z': z, 'j0': arm['j0'] if arm else j,
                           'ext': arm['ext'] if arm else (l[j] if bias == 1 else h[j])}
        if arm is None:
            continue
        d = arm['z']['dir']
        arm['ext'] = min(arm['ext'], l[j]) if d == 1 else max(arm['ext'], h[j])
        if j - arm['j0'] > max_wait:
            arm = None
            continue
        if confirm:
            lvl = sh5[j] if d == 1 else sl5[j]
            go = (d * (c[j] - lvl) > 0 and d * (c[j - 1] - lvl) <= 0
                  and d * (c[j] - o[j]) >= body_atr * a5[j])
            stop = arm['ext'] - d * sl_buf_atr * a5[j]
        else:
            go = True
            stop = (arm['z']['bot'] if d == 1 else arm['z']['top']) - d * sl_buf_atr * a5[j]
        if go and need_sweep:
            lo = max(arm['j0'] - sweep_bars, 0); hi = max(arm['j0'] - 3, lo + 1)
            go = arm['ext'] < l[lo:hi].min() if d == 1 else arm['ext'] > h[lo:hi].max()
        if not (go and ses[j]):
            continue
        dist = max(d * (c[j] - stop), min_sl)
        z = arm['z']
        active = [x for x in active if x is not z]                  # a zone is traded once
        arm = None
        if dist > max_sl:
            continue
        tgt = rr * dist
        if tp_mode == 'liq':                                        # 1H liquidity, if it pays at least rr
            liq = sh1[p1[j]] if d == 1 else sl1[p1[j]]
            if not np.isnan(liq) and d * (liq - c[j]) >= rr * dist:
                tgt = d * (liq - c[j])
        side[j] = d; sl_d[j] = dist; tp_d[j] = tgt
        if details:
            info[j] = {'zone': z['kind'], 'top': z['top'], 'bot': z['bot'], 'bias4h': int(bias), 'trend1h': int(t1[p1[j]])}
    out = pd.DataFrame({'side': side, 'sl': sl_d, 'tp': tp_d}, index=m5.index)
    if details:
        out['info'] = info
    return out
