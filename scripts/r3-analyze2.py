#!/usr/bin/env python3
"""R3: noise-aware analysis. Needs runs/ (chain1) and runs2/ (fresh processes). Prints and writes OUT_JSON:
per prompt: A1-A2, B1-B2 (cross-process noise), A-B pairs (A1B1, A2B2, A1B2, A2B1), on all targets and on 'predictable' targets (mean NLL_A < 8),
and the pooled 2x2 estimate of mean dNLL = mean(B)-mean(A) with its standard error. Usage: r3-analyze2.py RUNS1 RUNS2 OUT_JSON"""
import json, math, sys
from pathlib import Path
r1, r2, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
L = lambda d, arm, cls: {r['name']: r for r in json.load(open(d / f'{arm}-{cls}' / 'nll.json'))['records']}
D = {(d, a, c): L(p, a, c) for d, p in (('1', r1), ('2', r2)) for a in 'AB' for c in ('short', 'long')}
mean = lambda v: sum(v) / len(v) if v else float('nan')
def stats(x, y, keep=None):
    idx = [i for i in range(len(x)) if keep is None or keep[i]]
    d = [y[i] - x[i] for i in idx]
    if not d: return None
    sd = math.sqrt(sum((v - mean(d)) ** 2 for v in d) / (len(d) - 1)) if len(d) > 1 else float('nan')
    return {'n': len(d), 'mean': mean(d), 'se': sd / math.sqrt(len(d)), 'rms': math.sqrt(mean([v * v for v in d])), 'max_abs': max(abs(v) for v in d), 'n_gt_0.5': sum(abs(v) > 0.5 for v in d)}
res = []
for cls in ('short', 'long'):
    for name in D[('1', 'A', cls)]:
        if name.endswith('-rep'): continue
        g = lambda d, a: D[(d, a, cls)][name]['nll']
        A1, A2, B1, B2 = g('1', 'A'), g('2', 'A'), g('1', 'B'), g('2', 'B')
        am = [(x + y) / 2 for x, y in zip(A1, A2)]
        keep = [v < 8 for v in am]
        row = {'name': name, 'n_targets': len(A1), 'n_predictable': sum(keep), 'NLL_A_mean': mean(am), 'NLL_B_mean': mean([(x + y) / 2 for x, y in zip(B1, B2)])}
        for lab, (x, y) in {'A1_vs_A2': (A1, A2), 'B1_vs_B2': (B1, B2), 'A1_vs_B1': (A1, B1), 'A2_vs_B2': (A2, B2), 'A1_vs_B2': (A1, B2), 'A2_vs_B1': (A2, B1)}.items():
            row[lab] = stats(x, y); row[lab + '_pred'] = stats(x, y, keep)
        pooled = [((b1 + b2) / 2) - ((a1 + a2) / 2) for a1, a2, b1, b2 in zip(A1, A2, B1, B2)]
        row['pooled_dNLL'] = stats([0] * len(pooled), pooled); row['pooled_dNLL_pred'] = stats([0] * len(pooled), pooled, keep)
        res.append(row)
json.dump(res, open(out, 'w'), indent=1)
f = lambda s: 'n/a' if not s else f"{s['mean']:+.3f}±{s['se']:.3f} rms{s['rms']:.2f} max{s['max_abs']:.1f}"
for r in res:
    print(f"{r['name'][:26]:26} A={r['NLL_A_mean']:.2f} B={r['NLL_B_mean']:.2f} npred={r['n_predictable']}")
    print('   noise  A1-A2 all:', f(r['A1_vs_A2']), '| B1-B2 all:', f(r['B1_vs_B2']))
    print('   A-B    A1-B1 all:', f(r['A1_vs_B1']), '| A2-B2:', f(r['A2_vs_B2']))
    print('   pooled B-A all:', f(r['pooled_dNLL']), '| predictable:', f(r['pooled_dNLL_pred']), '| noise pred A1-A2:', f(r['A1_vs_A2_pred']), 'B1-B2:', f(r['B1_vs_B2_pred']))
