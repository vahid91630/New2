"""Signal bot: watches gold on M5 and sends a signal (entry, stop, target) for
you to trade by hand.  It places no orders.

Only setups whose past win rate is above signals.min_win_rate in both the
pick period and the unseen period are allowed to send (see signal_stats.py).

    python -m scalper.signals                 # live, prices from MetaTrader 5
    python -m scalper.signals --replay 10     # print the signals of the last 10 days of the data file
    python -m scalper.signals --test          # send a test message to Telegram
"""
import argparse, json, os, time, urllib.parse, urllib.request, yaml
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import pandas as pd
from .strategies import STRATEGIES
from .signal_stats import measure

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load():
    cfg = yaml.safe_load(open(os.path.join(ROOT, 'config', 'scalper.yaml'), encoding='utf-8'))
    path = os.path.join(ROOT, 'reports', 'signal_stats.json')
    stats = json.load(open(path, encoding='utf-8')) if os.path.exists(path) else measure(cfg)
    setups = [s for s in cfg['signals']['setups'] if stats.get(s['id'], {}).get('enabled')]
    return cfg, stats, setups


def send(cfg, text):
    print(text + "\n", flush=True)
    tg = cfg['signals']['telegram']
    token = tg['bot_token'] or os.getenv('TELEGRAM_BOT_TOKEN'); chat = tg['chat_id'] or os.getenv('TELEGRAM_CHAT_ID')
    if not (token and chat):
        return
    try:
        data = urllib.parse.urlencode({'chat_id': chat, 'text': text}).encode()
        urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data=data, timeout=15).read()
    except Exception as e:                       # a failed message must not stop the bot
        print('telegram failed:', e, flush=True)


def message(cfg, stats, setup, side, t_utc, bid, ask, sl_d, tp_d):
    sg, st = cfg['signals'], stats[setup['id']]
    tz = ZoneInfo(sg['timezone'])
    mid = (bid + ask) / 2
    entry = ask if side == 1 else bid
    sl = mid - side * sl_d; tp = mid + side * tp_d
    risk = abs(entry - sl); reward = abs(tp - entry)
    local = t_utc.astimezone(tz)
    until = local + timedelta(minutes=sg['valid_minutes'])
    u, a = st['unseen'], st['all']
    return "\n".join([
        f"{'🟢 خرید' if side == 1 else '🔴 فروش'} طلا XAUUSD — {setup['id']}",
        f"{setup['title']}",
        f"ورود: {entry:.2f}  (اسپرد {ask - bid:.2f})",
        f"حد ضرر: {sl:.2f}  (−${risk:.2f} برای هر ۰٫۰۱ لات)",
        f"حد سود: {tp:.2f}  (+${reward:.2f} برای هر ۰٫۰۱ لات)",
        f"زمان: {local:%H:%M} — معتبر تا {until:%H:%M}، خروج حداکثر {cfg['strategy']['max_hold_min']} دقیقه بعد",
        f"سابقه: برد {u['win_rate']:.0%} از {u['signals']} سیگنال دادهٔ دیده‌نشده ({a['win_rate']:.0%} از {a['signals']} در کل ۲۰۲۶)",
    ])


def replay(cfg, stats, setups, days):
    from .engine import load_m1, to_m5
    m1 = load_m1(os.path.join(ROOT, cfg['data']))
    m1 = m1[m1.index[-1] - pd.Timedelta(days=days):]
    m5 = to_m5(m1); rows = []
    for s in setups:
        sig = STRATEGIES[s['strategy']](m5, **s['params'])
        for t, r in sig[sig['side'] != 0].iterrows():
            rows.append((t + pd.Timedelta(minutes=5), s, r))
    for t, s, r in sorted(rows, key=lambda x: x[0]):
        i = m1.index.searchsorted(t)
        if i >= len(m1) or m1['spread_open'].iloc[i] > cfg['signals']['max_spread']:
            continue
        bid = m1['open'].iloc[i]; ask = bid + m1['spread_open'].iloc[i]
        print(message(cfg, stats, s, int(r['side']), m1.index[i].tz_localize('UTC').to_pydatetime(), bid, ask, r['sl'], r['tp']) + "\n")
    print(f"{len(rows)} signals in the last {days} days of data")


def live(cfg, stats, setups):
    import MetaTrader5 as mt5
    m = cfg['mt5']; sym = cfg['symbol']; off = timedelta(hours=m['server_utc_offset_hours'])
    login = m['login'] or os.getenv('MT5_LOGIN')
    kw = dict(login=int(login), password=m['password'] or os.getenv('MT5_PASSWORD'),
              server=m['server'] or os.getenv('MT5_SERVER')) if login else {}
    if not mt5.initialize(**kw):
        raise SystemExit(f"MT5 initialize failed: {mt5.last_error()}")
    if not mt5.symbol_select(sym, True):
        raise SystemExit(f"symbol {sym} not available (your broker may call it XAUUSDm, GOLD, ...)")
    print('watching', sym, 'setups:', [s['id'] for s in setups], flush=True)
    last_bar = None
    while True:
        time.sleep(3)
        rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M5, 1, 400)       # closed bars only
        if rates is None or len(rates) < 150:
            continue
        df = pd.DataFrame(rates)
        df.index = pd.to_datetime(df['time'], unit='s') - off                 # server time -> UTC
        if df.index[-1] == last_bar:
            continue
        first = last_bar is None
        last_bar = df.index[-1]
        if first:
            continue                                                          # no signal from a bar that closed before start
        tick = mt5.symbol_info_tick(sym)
        if tick.ask - tick.bid > cfg['signals']['max_spread']:
            continue
        for s in setups:
            r = STRATEGIES[s['strategy']](df[['open', 'high', 'low', 'close']], **s['params']).iloc[-1]
            if r['side'] != 0:
                send(cfg, message(cfg, stats, s, int(r['side']), datetime.now(timezone.utc), tick.bid, tick.ask, r['sl'], r['tp']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--replay', type=int, help='print the signals of the last N days of the data file')
    ap.add_argument('--test', action='store_true', help='send a test message')
    a = ap.parse_args()
    cfg, stats, setups = load()
    if not setups:
        raise SystemExit('no setup passes the win-rate rule; nothing to send')
    if a.test:
        send(cfg, 'ربات سیگنال طلا وصل است ✅')
    elif a.replay:
        replay(cfg, stats, setups, a.replay)
    else:
        live(cfg, stats, setups)


if __name__ == '__main__':
    main()
