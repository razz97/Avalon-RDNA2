#!/usr/bin/env python3
"""R3: compare A/B llama-qsa-nll outputs, check consistency and apply the G1 alarm thresholds. Usage: r3-compare.py RUNS_DIR PROMPTS_DIR OUT_JSON"""
import csv, hashlib, json, math, sys
from pathlib import Path
runs, pdir, outj = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
meta = {m['name']: m for m in json.load(open(pdir / 'prompts.json'))}
THR = {'S1': 0.02, 'S2': 0.02, 'S3': 0.02, 'S4': 0.03}
res = {'prompts': [], 'checks': {}, 'problems': []}
def load(arm, cls):
    d = runs / f'{arm}-{cls}'
    return json.load(open(d / 'nll.json')), json.load(open(d / 'meta.json')), d
def kernel_calls(d):
    f = next(iter((d / 'prof').rglob('k_kernel_stats.csv')), None)
    if not f: return None
    n = 0
    for r in csv.DictReader(open(f)):
        if 'k_fattn_qsa_rdna2' in r.get('Name', ''): n += int(float(r.get('Calls', 0)))
    return n
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
for cls in ('short', 'long'):
    try:
        A, mA, dA = load('A', cls); B, mB, dB = load('B', cls)
    except Exception as e:
        res['problems'].append(f'{cls}: missing outputs ({e})'); continue
    for k in ('binary_sha256', 'libggml_hip_sha256', 'commit', 'spec_sha256', 'model_manifest_sha256', 'n_ubatch', 'n_batch', 'kv_types'):
        if mA[k] != mB[k]: res['problems'].append(f'{cls}: A/B differ in {k}')
    pa = {k: v for k, v in mA['flags'].items() if k.startswith('LLAMA_QSA')}; pb = {k: v for k, v in mB['flags'].items() if k.startswith('LLAMA_QSA')}
    if pa.get('LLAMA_QSA_RDNA2_PREFILL') != '0' or pb.get('LLAMA_QSA_RDNA2_PREFILL') != '1' or pa.get('LLAMA_QSA_TILES') != '0' or pb.get('LLAMA_QSA_TILES') != '0': res['problems'].append(f'{cls}: wrong A/B flags {pa} {pb}')
    if '/build-main/' not in mA['binary_realpath'] or mA['binary_realpath'] != mB['binary_realpath']: res['problems'].append(f'{cls}: binary is not the patched build')
    if mA['rc'] or mB['rc']: res['problems'].append(f'{cls}: nonzero rc')
    ka, kb = (kernel_calls(runs.parent / 'dispatch' / 'A'), kernel_calls(runs.parent / 'dispatch' / 'B')) if cls == 'short' else (None, None)
    res['checks'][cls] = {'A_flags': pa, 'B_flags': pb, 'binary_sha256': mB['binary_sha256'], 'binary_realpath': mB['binary_realpath'], 'commit': mB['commit'],
                          'kernel_calls_A': ka, 'kernel_calls_B': kb, 'vram_min_free_mib_A': mA['vram_min_free_mib'], 'vram_min_free_mib_B': mB['vram_min_free_mib'],
                          'journal_A': mA['journal_kernel_suspicious_lines'], 'journal_B': mB['journal_kernel_suspicious_lines'], 'elapsed_A': mA['elapsed_total_s'], 'elapsed_B': mB['elapsed_total_s']}
    if cls == 'short':
        if not kb: res['problems'].append('short: B did not run k_fattn_qsa_rdna2 -> INVALID')
        if ka: res['problems'].append('short: A ran k_fattn_qsa_rdna2')
    if mA['journal_kernel_suspicious_lines'] or mB['journal_kernel_suspicious_lines']: res['problems'].append(f'{cls}: kernel journal lines')
    for ra, rb in zip(A['records'], B['records']):
        assert ra['name'] == rb['name'] and ra['positions'] == rb['positions'] and ra['target_ids'] == rb['target_ids'], 'A/B inputs differ'
        if ra['name'].endswith('-rep'):
            res.setdefault('repeat_records', []).append((ra['name'], cls)); continue
        m = meta[ra['name']]
        assert hashlib.sha256(json.dumps(ra['target_ids'], separators=(',', ':')).encode()).hexdigest() == m['target_ids_sha256'], 'target hash'
        nA, nB = ra['nll'], rb['nll']
        bad = [i for i, (x, y) in enumerate(zip(nA, nB)) if x is None or y is None or not math.isfinite(x) or not math.isfinite(y)]
        if bad or ra['error'] or rb['error']: res['problems'].append(f"{ra['name']}: NaN/Inf/error at {bad}")
        ok = [i for i in range(len(nA)) if i not in bad]
        d = [nB[i] - nA[i] for i in ok]
        n = len(d); mean = lambda v: sum(v) / len(v) if v else float('nan')
        pos = ra['positions']; tail = [k for k in range(len(pos)) if pos[k] >= ra['n_tokens'] - 32]; spread = [k for k in range(len(pos)) if k not in tail]
        sd = math.sqrt(sum((x - mean(d)) ** 2 for x in d) / (n - 1)) if n > 1 else float('nan')
        row = {'name': ra['name'], 'kind': m['kind'], 'group': m['group'], 'n_tokens': ra['n_tokens'], 'n_targets': n, 'prompt_tokens_sha256': m['prompt_tokens_sha256'],
               'NLL_A': mean([nA[i] for i in ok]), 'NLL_B': mean([nB[i] for i in ok]), 'dNLL': mean(d), 'dNLL_se': sd / math.sqrt(n) if n > 1 else None,
               'dNLL_spread': mean([nB[k] - nA[k] for k in spread if k in ok]), 'dNLL_tail': mean([nB[k] - nA[k] for k in tail if k in ok]),
               'NLL_A_spread': mean([nA[k] for k in spread if k in ok]), 'NLL_A_tail': mean([nA[k] for k in tail if k in ok]),
               'max_abs_d': max(abs(x) for x in d) if d else None, 'outliers_gt_0.5': [{'pos': pos[i], 'target': ra['target_ids'][i], 'nll_A': nA[i], 'nll_B': nB[i]} for i in ok if abs(nB[i] - nA[i]) > 0.5],
               'n_d_gt_0.1': sum(abs(x) > 0.1 for x in d), 'elapsed_A_s': ra['elapsed_s'], 'elapsed_B_s': rb['elapsed_s'], 'threshold': THR[m['group']]}
        row['gate_mean'] = 'PASS' if row['dNLL'] <= row['threshold'] else 'ALARM'
        res['prompts'].append(row)
res['all_gate_mean_pass'] = all(r['gate_mean'] == 'PASS' for r in res['prompts']) and len(res['prompts']) == 6
res['any_outlier'] = any(r['outliers_gt_0.5'] for r in res['prompts'])
json.dump(res, open(outj, 'w'), indent=1)
for r in res['prompts']:
    print(f"{r['name']:28} n={r['n_tokens']:6} A={r['NLL_A']:.4f} B={r['NLL_B']:.4f} d={r['dNLL']:+.4f}±{(r['dNLL_se'] or 0):.4f} max|d|={r['max_abs_d']:.3f} d>0.1:{r['n_d_gt_0.1']} outl={len(r['outliers_gt_0.5'])} {r['gate_mean']}")
print('problems:', res['problems']); print('checks:', json.dumps({k: {a: b for a, b in v.items() if a.startswith(('kernel', 'vram'))} for k, v in res['checks'].items()}))
