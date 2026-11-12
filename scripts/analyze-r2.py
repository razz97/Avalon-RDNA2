#!/usr/bin/env python3
"""R2 analysis: determinism, A/B divergence vs top1-top2 margins, forced-prefix logprob deltas, MoE counters. Writes
results/r2-analysis.json and prints markdown tables. Usage: analyze-r2.py"""
import json, re, glob, os, itertools, hashlib
from pathlib import Path

R = Path(__file__).parent.parent / 'results'
P = R / 'probes'


def load(n):
    f = P / (n + '.json')
    return json.loads(f.read_text()) if f.exists() else None


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None


def top1(d, k):
    t = d['steps'][k]['top5']
    return t[0][1] if t else None


out = {}
# 1. determinism / A vs B (no MTP, 40K)
groups = {g: [load(f'{g}-free-{i}') for i in range(1, 6)] for g in ('A', 'B')}
for g in groups:
    groups[g] = [x for x in groups[g] if x and 'error' not in x]
out['prompt_sha256'] = sorted({x['prompt_sha256'] for g in groups.values() for x in g})
for g, xs in groups.items():
    seqs = [tuple(x['tokens']) for x in xs]
    maxd = 0.0
    for a, b in itertools.combinations(xs, 2):
        for k in range(min(len(a['steps']), len(b['steps']))):
            ta, tb = a['steps'][k]['top5'], b['steps'][k]['top5']
            if ta and tb and ta[0][0] == tb[0][0]:
                maxd = max(maxd, abs(ta[0][1] - tb[0][1]))
    out[f'{g}_free_40k'] = {'runs': len(xs), 'distinct_sequences': len(set(seqs)), 'max_abs_top1_logprob_diff_between_runs': maxd}
def pair_max(x, y):
    m, n = 0.0, 0
    for k in range(min(len(x['steps']), len(y['steps']))):
        if x['tokens'][:k] != y['tokens'][:k]:
            break                       # only positions with an identical prefix are comparable
        ta, tb = x['steps'][k]['top5'], y['steps'][k]['top5']
        lp = dict(map(tuple, ta))
        for i, l in tb:
            if i in lp:
                m = max(m, abs(l - lp[i])); n += 1
    return m


pw = {}
for name, (g1, g2) in {'AA': ('A', 'A'), 'BB': ('B', 'B'), 'AB': ('A', 'B')}.items():
    vals = []
    for i, x in enumerate(groups[g1]):
        for j, y in enumerate(groups[g2]):
            if (g1 == g2 and j <= i):
                continue
            vals.append(round(pair_max(x, y), 3))
    pw[name] = {'n_pairs': len(vals), 'max_abs_logprob_delta_per_pair': sorted(vals)}
out['pairwise_noise_floor_40k_nomtp'] = pw
if groups['A'] and groups['B']:
    a0 = groups['A'][0]
    rows = []
    for i, b in enumerate(groups['B']):
        k = first_diff(a0['tokens'], b['tokens'])
        rows.append({'B_run': i + 1, 'first_diff': k,
                     'A_margin_at_k': a0['steps'][k]['margin'] if k is not None else None,
                     'B_margin_at_k': b['steps'][k]['margin'] if k is not None else None})
    out['A1_vs_B_runs'] = rows
def gstats(names):
    xs = [load(n) for n in names]
    xs = [x for x in xs if x and 'error' not in x]
    vals = [round(pair_max(x, y), 3) for i, x in enumerate(xs) for y in xs[i+1:]]
    return {'runs': len(xs), 'distinct_sequences': len({tuple(x['tokens']) for x in xs}),
            'token_at_23': [x['tokens'][23] if len(x['tokens']) > 23 else None for x in xs],
            'pair_max_abs_logprob_delta': sorted(vals)}


out['ring0_determinism'] = {g: gstats([f'{g}-{i}' for i in range(1, 4)]) for g in ('A-ring0', 'B-ring0', 'B-fp32-ring0')}
out['ring0_all_AB'] = {'A_vs_B_pairs': sorted(round(pair_max(load(f'A-ring0-{i}'), load(f'B-ring0-{j}')), 3) for i in range(1, 4) for j in range(1, 4)
                                              if load(f'A-ring0-{i}') and load(f'B-ring0-{j}'))}
out['127k_noise_floor'] = {'A': gstats(['A-free-127k', 'A-free-127k-2']), 'B': gstats(['B-free-127k', 'B-free-127k-2']),
                           'AB': sorted(round(pair_max(load(a), load(b)), 3) for a in ('A-free-127k', 'A-free-127k-2') for b in ('B-free-127k', 'B-free-127k-2')
                                        if load(a) and load(b))}
# 2. forced prefix (A1 tokens) deltas
a1 = load('A-free-1')


def forced_stats(name, ref):
    d = load(name)
    if not d or 'error' in d or not ref:
        return None
    dl, flips, n = [], [], 0
    for k in range(min(len(d['steps']), len(ref['steps']))):
        ta, tb = ref['steps'][k]['top5'], d['steps'][k]['top5']
        if not ta or not tb:
            continue
        n += 1
        lp = {i: l for i, l in ta}
        common = [(lp[i], l) for i, l in tb if i in lp]
        for x, y in common:
            dl.append(abs(x - y))
        if ta[0][0] != tb[0][0]:
            flips.append({'k': k, 'ref_margin': ref['steps'][k]['margin'], 'this_margin': d['steps'][k]['margin']})
    return {'positions': n, 'argmax_flips': flips, 'max_abs_logprob_delta_top5': max(dl) if dl else None,
            'mean_abs_logprob_delta_top5': sum(dl)/len(dl) if dl else None}


out['forced_B_vs_A1'] = forced_stats('B-forcedA1', a1)
out['forced_B_fp32_vs_A1'] = forced_stats('B-fp32-forcedA1', a1)
if load('B-fp32-free-1') and a1:
    b = load('B-fp32-free-1')
    out['fp32_free_vs_A1_first_diff'] = first_diff(a1['tokens'], b['tokens'])
    out['fp32_free_vs_B_free1_first_diff'] = first_diff(groups['B'][0]['tokens'], b['tokens']) if groups['B'] else None
u = load('U-free-1')
if u and a1 and 'error' not in u:
    out['unpatched_vs_A1'] = {'tokens_identical': u['tokens'] == a1['tokens'], 'logprobs_identical': all(
        (x['top5'] == y['top5']) for x, y in zip(u['steps'], a1['steps'])), 'argv_bin': u['argv'][0]}
# 3. 127K
a127, b127 = load('A-free-127k'), load('B-free-127k')
if a127 and b127 and 'error' not in a127 and 'error' not in b127:
    k = first_diff(a127['tokens'], b127['tokens'])
    out['127k_free'] = {'prompt_sha256_equal': a127['prompt_sha256'] == b127['prompt_sha256'], 'first_diff': k,
                        'A_margin_at_k': a127['steps'][k]['margin'] if k is not None else None,
                        'B_margin_at_k': b127['steps'][k]['margin'] if k is not None else None}
out['127k_forced_B_vs_A'] = forced_stats('B-forcedA-127k', a127)
# 4. MTP on (harness) with n_probs
mtp = {}
for d in sorted(glob.glob(str(R / 'runs' / '[AB]-40960-b8192-ub4096-*'))):
    ev = Path(d) / 'events-0.jsonl'
    pp = Path(d) / 'prompt-plan.json'
    if not ev.exists():
        continue
    toks, marg = [], []
    for l in ev.read_text().splitlines():
        e = json.loads(l)['event']
        for c in e.get('completion_probabilities') or []:
            toks.append(c.get('id'))
            tl = c.get('top_logprobs') or []
            marg.append(tl[0]['logprob'] - tl[1]['logprob'] if len(tl) > 1 else None)
        if not e.get('completion_probabilities') and e.get('tokens'):
            for t in e['tokens']:
                toks.append(t); marg.append(None)
    if any(m is not None for m in marg):
        mtp[os.path.basename(d)] = {'prompt_sha256': json.loads(pp.read_text())['token_ids_sha256'] if pp.exists() else None, 'tokens': toks[:128], 'margins': marg[:128]}
out['mtp_runs'] = {k: {'prompt_sha256': v['prompt_sha256'], 'n': len(v['tokens'])} for k, v in mtp.items()}
amt = [v for k, v in mtp.items() if k.startswith('A-')]
bmt = [v for k, v in mtp.items() if k.startswith('B-')]
if amt and bmt:
    rows = []
    for i, b in enumerate(bmt):
        k = first_diff(amt[0]['tokens'], b['tokens'])
        rows.append({'B_run': i + 1, 'first_diff_vs_A1': k, 'A_margin': amt[0]['margins'][k] if k is not None else None,
                     'B_margin': b['margins'][k] if k is not None else None})
    out['mtp_A_vs_B'] = rows
    out['mtp_A_distinct'] = len({tuple(v['tokens']) for v in amt})
    out['mtp_B_distinct'] = len({tuple(v['tokens']) for v in bmt})
# 5. MoE counters from the engine logs
moe = []
for d in sorted(glob.glob(str(R / 'runs' / '[AB]-*'))):
    log = Path(d) / 'engine.log'
    if not log.exists():
        continue
    t = log.read_text(errors='replace')
    m1 = re.search(r'prefill ring: (\d+) experts copied, (\d+) read in place \(([\d.]+) % in place\)', t)
    m2 = re.search(r'rewarm: (\d+) evicted entries and (\d+) ranked pairs', t)
    m3 = re.search(r'rewarm: (\d+) entries restored', t)
    moe.append({'run': os.path.basename(d), 'copied': int(m1.group(1)) if m1 else None, 'in_place': int(m1.group(2)) if m1 else None,
                'in_place_pct': float(m1.group(3)) if m1 else None, 'evicted': int(m2.group(1)) if m2 else None,
                'ranked_pairs': int(m2.group(2)) if m2 else None, 'restored': int(m3.group(1)) if m3 else None})
out['moe_counters'] = moe
(R / 'r2-analysis.json').write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps({k: v for k, v in out.items() if k != 'moe_counters'}, indent=1)[:6000])
print('MoE counters')
for m in moe:
    print(m)
