"""Live runner for MetaTrader 5 (Windows, terminal open and logged in).

    python -m scalper.live_mt5

Uses the same strategy function and the same money rules as the backtest.
Starts in dry_run mode (config/scalper.yaml): it prints the order it would
send and sends nothing.  Run it on a demo account before any real money.
"""
import os, time, yaml
from datetime import datetime, timedelta, timezone
import numpy as np, pandas as pd
from .strategies import STRATEGIES
from .engine import CONTRACT, LOT_STEP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log(*a):
    print(datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'), *a, flush=True)


def main():
    import MetaTrader5 as mt5
    cfg = yaml.safe_load(open(os.path.join(ROOT, 'config', 'scalper.yaml'), encoding='utf-8'))
    m, acc, st = cfg['mt5'], cfg['account'], cfg['strategy']
    sym = cfg['symbol']; off = timedelta(hours=m['server_utc_offset_hours'])
    login = m['login'] or os.getenv('MT5_LOGIN')
    kw = dict(login=int(login), password=m['password'] or os.getenv('MT5_PASSWORD'),
              server=m['server'] or os.getenv('MT5_SERVER')) if login else {}
    if not mt5.initialize(**kw):
        raise SystemExit(f"MT5 initialize failed: {mt5.last_error()}")
    if not mt5.symbol_select(sym, True):
        raise SystemExit(f"symbol {sym} not available (your broker may call it XAUUSDm, GOLD, ...)")
    info = mt5.symbol_info(sym)
    fn = STRATEGIES[st['name']]
    last_bar = None
    log(f"started on {sym}, dry_run={m['dry_run']}")

    def my_positions():
        return [p for p in (mt5.positions_get(symbol=sym) or []) if p.magic == m['magic']]

    def close(p, why):
        tick = mt5.symbol_info_tick(sym)
        req = dict(action=mt5.TRADE_ACTION_DEAL, symbol=sym, volume=p.volume, position=p.ticket,
                   type=mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                   price=tick.bid if p.type == mt5.POSITION_TYPE_BUY else tick.ask,
                   deviation=m['deviation_points'], magic=m['magic'], comment=why,
                   type_filling=mt5.ORDER_FILLING_IOC)
        log('CLOSE', why, req if m['dry_run'] else mt5.order_send(req))

    def today_stats(now_utc):
        day0 = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        deals = mt5.history_deals_get(day0 + off, now_utc + off + timedelta(hours=1)) or []
        mine = [d for d in deals if d.magic == m['magic'] and d.symbol == sym]
        pnl = sum(d.profit + d.commission + d.swap for d in mine)
        n = sum(1 for d in mine if d.entry == mt5.DEAL_ENTRY_IN)
        return pnl, n

    while True:
        time.sleep(5)
        now = datetime.now(timezone.utc)
        # manage the open trade: time limit and end of session
        for p in my_positions():
            opened = datetime.fromtimestamp(p.time, timezone.utc) - off
            if now - opened >= timedelta(minutes=st['max_hold_min']):
                close(p, 'time')
            elif now.hour >= st['flat_hour_utc']:
                close(p, 'eod')
        rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M5, 1, 400)   # closed bars only
        if rates is None or len(rates) < 150:
            continue
        df = pd.DataFrame(rates)
        df.index = pd.to_datetime(df['time'], unit='s') - off             # server time -> UTC
        if df.index[-1] == last_bar:
            continue
        last_bar = df.index[-1]
        sig = fn(df[['open', 'high', 'low', 'close']], **st['params']).iloc[-1]
        if sig['side'] == 0 or my_positions():
            continue
        bal = mt5.account_info().balance
        pnl, n = today_stats(now)
        tick = mt5.symbol_info_tick(sym); spread = tick.ask - tick.bid
        if acc['daily_target'] > 0 and pnl >= acc['daily_target']:
            log('skip: daily target reached', round(pnl, 2)); continue
        if pnl <= -acc['daily_loss_pct'] / 100 * bal:
            log('skip: daily loss limit', round(pnl, 2)); continue
        if n >= acc['max_trades_per_day'] or spread > acc['max_spread'] or now.hour >= st['flat_hour_utc']:
            log('skip: trades', n, 'spread', round(spread, 2)); continue
        base = bal if acc['compounding'] else min(bal, acc['start_balance'])
        loss_oz = sig['sl'] + acc['slippage']
        contract = info.trade_contract_size or CONTRACT
        step = info.volume_step or LOT_STEP
        lots = np.floor(base * acc['risk_pct'] / 100 / (loss_oz * contract) / step + 1e-9) * step
        if lots < info.volume_min:
            if info.volume_min * loss_oz * contract > base * acc['max_risk_pct'] / 100:
                log('skip: stop too wide for this account', round(sig['sl'], 2)); continue
            lots = info.volume_min
        lots = round(min(lots, acc['max_lots']), 2)
        side = int(sig['side']); mid = (tick.ask + tick.bid) / 2
        req = dict(action=mt5.TRADE_ACTION_DEAL, symbol=sym, volume=lots,
                   type=mt5.ORDER_TYPE_BUY if side == 1 else mt5.ORDER_TYPE_SELL,
                   price=tick.ask if side == 1 else tick.bid,
                   sl=round(mid - side * sig['sl'], info.digits), tp=round(mid + side * sig['tp'], info.digits),
                   deviation=m['deviation_points'], magic=m['magic'], comment='gold-scalper',
                   type_filling=mt5.ORDER_FILLING_IOC)
        if m['dry_run']:
            log('DRY RUN order:', req)
        else:
            r = mt5.order_send(req)
            log('ORDER', r.retcode, r.comment, req)


if __name__ == '__main__':
    main()
