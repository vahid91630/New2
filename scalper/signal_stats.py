"""Measure every signal setup on the real 2026 data and decide which ones are
allowed to send signals: win rate above the threshold in the period used to
pick the rules AND in the period kept unseen.

    python -m scalper.signal_stats      # writes reports/signal_stats.json / .md
"""
import json, os, yaml
import pandas as pd
from .engine import load_m1, to_m5, run, Account
from .strategies import STRATEGIES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def measure(cfg):
    m1 = load_m1(os.path.join(ROOT, cfg['data']))
    split = pd.Timestamp(cfg['in_sample_end']) + pd.Timedelta(days=1)
    st, sg = cfg['strategy'], cfg['signals']
    # every signal is followed with a fixed 0.01 lot, no daily limits
    raw = Account(start_balance=1e6, leverage=1e9, risk_pct=1e-9, max_risk_pct=1e9, daily_target=0, daily_loss_pct=1e9,
                  max_trades_per_day=99, max_spread=sg['max_spread'])
    out = {}
    for s in sg['setups']:
        r = run(m1, STRATEGIES[s['strategy']](to_m5(m1), **s['params']), raw, s.get('max_hold_min', st['max_hold_min']), st['flat_hour_utc'])
        t = r.trades
        row = {'title': s['title']}
        for key, part in (('pick', t[t.t_in < split]), ('unseen', t[t.t_in >= split]), ('all', t)):
            w = part.pnl > 0
            gl = -part.pnl[~w].sum()
            row[key] = {
                'signals': int(len(part)), 'win_rate': round(float(w.mean()), 3) if len(part) else None,
                'net_usd_001lot': round(float(part.pnl.sum()), 1),
                'profit_factor': round(float(part.pnl[w].sum() / gl), 2) if gl > 0 else None,
                'avg_win': round(float(part.pnl[w].mean()), 2) if w.any() else None,
                'avg_loss': round(float(part.pnl[~w].mean()), 2) if (~w).any() else None,
            }
        ok = all(row[k]['signals'] >= sg['min_trades'] and row[k]['win_rate'] > sg['min_win_rate'] for k in ('pick', 'unseen'))
        row['passes_rule'] = bool(ok)
        row['enabled'] = bool(ok or s.get('always_send'))     # always_send: you asked for this setup; its record is shown in every message
        row['per_week'] = round(len(t) / (r.stats['days'] / 5.0), 1)
        out[s['id']] = row
    return out


def main():
    cfg = yaml.safe_load(open(os.path.join(ROOT, 'config', 'scalper.yaml'), encoding='utf-8'))
    res = measure(cfg)
    os.makedirs(os.path.join(ROOT, 'reports'), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, 'reports', 'signal_stats.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    lines = ['| setup | signals/week | win rate (pick) | win rate (unseen) | avg win / avg loss, 0.01 lot | net unseen, 0.01 lot | sends signals |', '|---|---|---|---|---|---|---|']
    for k, r in res.items():
        a = r['all']
        lines.append(f"| {k} | {r['per_week']} | {r['pick']['win_rate']:.0%} ({r['pick']['signals']}) | {r['unseen']['win_rate']:.0%} ({r['unseen']['signals']}) | ${a['avg_win']} / ${a['avg_loss']} | ${r['unseen']['net_usd_001lot']:+} | {'YES' if r['passes_rule'] else 'yes (always_send, does not pass the rule)' if r['enabled'] else 'no'} |")
    md = "\n".join(lines)
    open(os.path.join(ROOT, 'reports', 'signal_stats.md'), 'w', encoding='utf-8').write(md + "\n")
    print(md)


if __name__ == '__main__':
    main()
