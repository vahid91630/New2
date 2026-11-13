"""Backtest the configured scalper on real 2026 tick-derived bars.

    python -m scalper.run_backtest            # report + chart in reports/
    python -m scalper.run_backtest --risk 20  # same rules, bigger size
"""
import argparse, os, yaml
import numpy as np, pandas as pd
from .engine import load_m1, to_m5, run, Account
from .strategies import STRATEGIES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fmt(name, r, target):
    s = r.stats
    if not s.get('trades'):
        return f"{name}: no trades"
    return "\n".join([
        f"== {name}: {r.daily.index[0].date()} -> {r.daily.index[-1].date()} ({s['days']} trading days)",
        f"  balance          ${r.equity.iloc[0] - r.daily.iloc[0]:.0f} -> ${s['final']:.2f}   (net {s['net']:+.2f})",
        f"  avg per day      ${s['avg_per_day']:+.2f}   (target ${target:.0f}; median day ${s['median_day']:+.2f})",
        f"  days >= target   {s['days_hit_target']:.0%}   green {s['green_days']:.0%} / red {s['red_days']:.0%} / no trade {1 - s['green_days'] - s['red_days']:.0%}",
        f"  trades           {s['trades']} ({s['trades_per_day']:.2f}/day), avg lot {s['avg_lots']:.2f}",
        f"  win rate         {s['win_rate']:.0%}   avg win ${s['avg_win']:.2f} / avg loss ${s['avg_loss']:.2f}",
        f"  profit factor    {s['profit_factor']:.2f}",
        f"  max drawdown     ${s['max_dd']:.2f} ({s['max_dd_pct']:.0%})   lowest balance ${s['min_equity']:.2f}",
        f"  best / worst day ${s['best_day']:+.2f} / ${s['worst_day']:+.2f}",
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=os.path.join(ROOT, 'config', 'scalper.yaml'))
    ap.add_argument('--risk', type=float, help='override risk_pct (and allow it as max_risk_pct)')
    ap.add_argument('--out', default=os.path.join(ROOT, 'reports'))
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding='utf-8'))
    acc = Account(**cfg['account'])
    tag = ''
    if a.risk:
        acc.risk_pct = a.risk; acc.max_risk_pct = max(acc.max_risk_pct, a.risk); acc.max_lots = 1.0
        tag = f'_risk{a.risk:g}'
    m1 = load_m1(os.path.join(ROOT, cfg['data']))
    st = cfg['strategy']; fn = STRATEGIES[st['name']]
    split = pd.Timestamp(cfg['in_sample_end']) + pd.Timedelta(days=1)
    parts = {
        'IN-SAMPLE (used to pick the rule)': m1[:split - pd.Timedelta(minutes=1)],
        'OUT-OF-SAMPLE (never seen while picking)': m1[split:],
        'FULL PERIOD': m1,
    }
    os.makedirs(a.out, exist_ok=True)
    lines = [f"Gold scalper backtest - {st['name']} {st['params']}",
             f"account ${acc.start_balance:.0f}, risk {acc.risk_pct}%/trade, daily target ${acc.daily_target}, real tick spread + ${acc.slippage} slippage", ""]
    res = {}
    for name, d in parts.items():
        r = run(d, fn(to_m5(d), **st['params']), acc, st['max_hold_min'], st['flat_hour_utc'])
        res[name] = r
        lines += [fmt(name, r, acc.daily_target), ""]
    full = res['FULL PERIOD']
    monthly = full.daily.groupby(full.daily.index.to_period('M')).sum().round(2)
    lines += ["monthly net ($): " + ", ".join(f"{k}: {v:+.0f}" for k, v in monthly.items())]
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(a.out, f'backtest{tag}.txt'), 'w').write(txt + "\n")
    full.trades.to_csv(float_format='%.3f', path_or_buf=os.path.join(a.out, f'trades{tag}.csv'), index=False)
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.plot(full.equity.index, full.equity.values, color='#1f6f8b', lw=1.6, label='balance')
        ideal = acc.start_balance + acc.daily_target * np.arange(1, len(full.equity) + 1)
        ax.plot(full.equity.index, ideal, color='#999', lw=1, ls='--', label=f'${acc.daily_target:.0f}/day target')
        ax.axvline(split, color='#c0392b', lw=1, ls=':'); ax.text(split, ax.get_ylim()[1], ' unseen data ->', color='#c0392b', va='top', fontsize=8)
        ax.axhline(acc.start_balance, color='#bbb', lw=0.8)
        ax.set_yscale('symlog', linthresh=1000) if ideal[-1] > 20 * max(full.equity.max(), 1) else None
        ax.set_title(f"XAUUSD scalper, ${acc.start_balance:.0f} account, risk {acc.risk_pct:g}%/trade"); ax.set_ylabel('USD'); ax.legend(loc='upper left', frameon=False)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, f'equity{tag}.png'), dpi=130)
    except ImportError:
        pass


if __name__ == '__main__':
    main()
