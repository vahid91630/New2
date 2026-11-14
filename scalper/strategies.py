"""Scalping setups on M5 bars.  Every function returns an M5-indexed frame with
side (+1 long / -1 short / 0 none), sl and tp (distances in $ per ounce).
Only closed bars are used: the value on row T uses data up to the close of T."""
import numpy as np
import pandas as pd


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def atr(df, n=14):
    pc = df['close'].shift(1)
    tr = pd.concat([df['high'] - df['low'], (df['high'] - pc).abs(), (df['low'] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1.0 / n, adjust=False).mean()
    dn = (-d).clip(lower=0).ewm(alpha=1.0 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _session(idx, start_h, end_h):
    h = idx.hour + idx.minute / 60.0
    return (h >= start_h) & (h < end_h)


def _frame(df, long_, short_, sl, tp):
    side = np.where(long_, 1, np.where(short_, -1, 0))
    out = pd.DataFrame({'side': side, 'sl': sl, 'tp': tp}, index=df.index)
    out.loc[out['sl'].isna() | out['tp'].isna(), 'side'] = 0
    return out


def trend_pullback(df, fast=20, slow=100, sl_atr=1.5, rr=1.5, start_h=7, end_h=20):
    """Trend by two EMAs, entry when price dips to the fast EMA and closes back
    above the previous bar's high (mirror for shorts)."""
    a = atr(df); ef = ema(df['close'], fast); es = ema(df['close'], slow)
    up = (ef > es) & (es > es.shift(6))
    dn = (ef < es) & (es < es.shift(6))
    long_ = up & (df['low'] <= ef) & (df['close'] > df['high'].shift(1)) & (df['close'] > ef)
    short_ = dn & (df['high'] >= ef) & (df['close'] < df['low'].shift(1)) & (df['close'] < ef)
    ses = _session(df.index, start_h, end_h)
    return _frame(df, long_ & ses, short_ & ses, sl_atr * a, sl_atr * rr * a)


def breakout(df, n=24, sl_atr=1.5, rr=1.5, start_h=7, end_h=20, vol_mult=1.2):
    """Close beyond the high/low of the last n bars on a bar larger than usual."""
    a = atr(df)
    hh = df['high'].rolling(n).max().shift(1); ll = df['low'].rolling(n).min().shift(1)
    big = (df['high'] - df['low']) > vol_mult * a.shift(1)
    long_ = (df['close'] > hh) & big & (df['close'].shift(1) <= hh.shift(1))
    short_ = (df['close'] < ll) & big & (df['close'].shift(1) >= ll.shift(1))
    ses = _session(df.index, start_h, end_h)
    return _frame(df, long_ & ses, short_ & ses, sl_atr * a, sl_atr * rr * a)


def mean_reversion(df, n=20, k=2.5, rsi_n=7, lo=20, hi=80, sl_atr=1.5, rr=1.0, start_h=0, end_h=7):
    """Fade a stretch outside the Bollinger band when RSI is extreme and the
    bar closes back in the direction of the mean."""
    a = atr(df); m = df['close'].rolling(n).mean(); sd = df['close'].rolling(n).std()
    r = rsi(df['close'], rsi_n)
    long_ = (df['low'] < m - k * sd) & (r.shift(1) < lo) & (df['close'] > df['open'])
    short_ = (df['high'] > m + k * sd) & (r.shift(1) > hi) & (df['close'] < df['open'])
    ses = _session(df.index, start_h, end_h)
    return _frame(df, long_ & ses, short_ & ses, sl_atr * a, sl_atr * rr * a)


def opening_range(df, open_h=13.5, range_min=30, window_h=2.5, sl_frac=0.5, rr=1.5, min_atr=0.8, max_atr=4.0):
    """Breakout of the first `range_min` minutes after a session open (UTC).
    Stop inside the range (sl_frac of its height), one trade per side per day."""
    a = atr(df)
    h = df.index.hour + df.index.minute / 60.0
    d = df.index.normalize()
    in_rng = (h >= open_h) & (h < open_h + range_min / 60.0)
    rh = df['high'].where(in_rng).groupby(d).transform('max')
    rl = df['low'].where(in_rng).groupby(d).transform('min')
    after = (h >= open_h + range_min / 60.0) & (h < open_h + window_h)
    height = rh - rl
    okh = (height >= min_atr * a) & (height <= max_atr * a)
    long_raw = after & okh & (df['close'] > rh)
    short_raw = after & okh & (df['close'] < rl)
    first_long = long_raw & (long_raw.groupby(d).cumsum() == 1)
    first_short = short_raw & (short_raw.groupby(d).cumsum() == 1)
    sl = sl_frac * height + (df['close'] - rh).clip(lower=0).where(first_long, 0) + (rl - df['close']).clip(lower=0).where(first_short, 0)
    return _frame(df, first_long, first_short, sl, rr * sl_frac * height)


from .smc import mtf_smc

STRATEGIES = {
    'mtf_smc': mtf_smc,
    'trend_pullback': trend_pullback,
    'breakout': breakout,
    'mean_reversion': mean_reversion,
    'opening_range': opening_range,
}
