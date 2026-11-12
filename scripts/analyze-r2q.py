#!/usr/bin/env python3
"""R2 final analysis. Probes in results/probes ran on the PATCHED binary (names q-*); results/probes-baseline-binary holds the
earlier probes, which all ran the UNPATCHED binary (baseline run-to-run noise only). Writes results/r2-final-analysis.json."""
import json, itertools
from pathlib import Path

R = Path(__file__).parent.parent / 'results'


def load(n, d='probes'):
    f = R / d / (n + '.json')
    if not f.exists():
        return None
    x = json.loads(f.read_text())
    return None if 'error' in x else x


def lp(step):
    return {i: l for i, l in step['top5']}


def pair(x, y):
    """max |dlogprob| over top-5 ids common to both, only over positions with identical prefix; and first token divergence."""
    m, fd = 0.0, None
    for k in range(min(len(x['steps']), len(y['steps']))):
        if x['tokens'][:k] != y['tokens'][:k]:
            break
        a, b = lp(x['steps'][k]), lp(y['steps'][k])
        for i in a:
            if i in b:
                m = max(m, abs(a[i] - b[i]))
    for k, (s, t) in enumerate(zip(x['tokens'], y['tokens'])):
        if s != t:
            fd = k
            break
    return round(m, 3), fd


def group(names, d='probes'):
    xs = [load(n, d) for n in names]
    xs = [x for x in xs if x]
    return xs


def stats(xs):
    pr = [pair(a, b) for a, b in itertools.combinations(xs, 2)]
    return {'runs': len(xs), 'distinct_sequences': len({tuple(x['tokens']) for x in xs}),
            'token_at_23': [x['tokens'][23] if len(x['tokens']) > 23 else None for x in xs],
            'pair_max_dlogprob': sorted(p[0] for p in pr), 'pair_first_token_diff': [p[1] for p in pr]}


def cross(xs, ys):
    pr = [pair(a, b) for a in xs for b in ys]
    return {'pair_max_dlogprob': sorted(p[0] for p in pr), 'pair_first_token_diff': [p[1] for p in pr]}


def forced(x, ref):
    if not x or not ref:
        return None
    d, flips = [], []
    for k in range(min(len(x['steps']), len(ref['steps']))):
        a, b = lp(ref['steps'][k]), lp(x['steps'][k])
        d += [abs(a[i] - b[i]) for i in a if i in b]
        if ref['steps'][k]['top5'][0][0] != x['steps'][k]['top5'][0][0]:
            flips.append({'k': k, 'ref_margin': ref['steps'][k]['margin'], 'margin': x['steps'][k]['margin']})
    return {'max_dlogprob': round(max(d), 3), 'mean_dlogprob': round(sum(d) / len(d), 4), 'argmax_flips': flips}


out = {}
A = group([f'q-A-free-{i}' for i in range(1, 4)])
B = group([f'q-B-free-{i}' for i in range(1, 6)])
out['prompt_sha256'] = sorted({x['prompt_sha256'] for x in A + B})
out['binary_A_B'] = sorted({[a for a in x['argv'] if a.endswith('llama-server')][0] for x in A + B})
out['A_A_40k'], out['B_B_40k'], out['A_B_40k'] = stats(A), stats(B), cross(A, B)
if A:
    out['forced_B_vs_A1'] = forced(load('q-B-forcedA1'), A[0])
    out['forced_B_fp32_vs_A1'] = forced(load('q-B-fp32-forcedA1'), A[0])
    f = load('q-B-fp32-free-1')
    if f:
        out['fp32_free_vs_A'] = cross([f], A)
A7 = group(['q-A-free-127k-1', 'q-A-free-127k-2'])
B7 = group(['q-B-free-127k-1', 'q-B-free-127k-2'])
out['127k'] = {'A_A': stats(A7), 'B_B': stats(B7), 'A_B': cross(A7, B7),
               'forced_B_vs_A': forced(load('q-B-forcedA-127k'), A7[0] if A7 else None),
               'prompt_sha256_equal': len({x['prompt_sha256'] for x in A7 + B7}) == 1}
# baseline-binary noise (all probes ran the unpatched binary)
base = group([f'A-free-{i}' for i in range(1, 6)], 'probes-baseline-binary') + group([f'B-free-{i}' for i in range(1, 6)], 'probes-baseline-binary')
out['unpatched_binary_noise_40k'] = stats(base)
out['unpatched_binary_ring0_noise_40k'] = stats(group([f'A-ring0-{i}' for i in range(1, 4)] + [f'B-ring0-{i}' for i in range(1, 4)], 'probes-baseline-binary'))
(R / 'r2-final-analysis.json').write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps(out, indent=1))
