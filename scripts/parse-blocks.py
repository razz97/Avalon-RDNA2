#!/usr/bin/env python3
"""Per-block prefill speed and memory facts of a oneshot run dir -> JSON line. Usage: parse-blocks.py RUN_DIR..."""
import json, re, sys, pathlib
for d in map(pathlib.Path, sys.argv[1:]):
    log = (d / 'engine.log').read_text(errors='replace')
    blocks, prev_n, prev_t = [], 0, 0.0
    for m in re.finditer(r'prompt processing, n_tokens =\s*(\d+), progress = [\d.]+, t =\s*([\d.]+) s', log):
        n, t = int(m.group(1)), float(m.group(2))
        blocks.append({'start': prev_n, 'end': n, 'seconds': round(t - prev_t, 3), 'tok_s': round((n - prev_n) / (t - prev_t), 1)})
        prev_n, prev_t = n, t
    def g(p):
        m = re.search(p, log); return m.groups() if m else None
    comp = g(r'prompt compute buffer: ([\d.]+) MiB at (\d+) tokens')
    cap = g(r'capacity: policy=\S+ granted=(\d+) MiB free=(\d+) MiB reserve=(\d+) MiB')
    ev = [json.loads(l) for l in (d / 'events-0.jsonl').read_text().splitlines() if l.strip()] if (d / 'events-0.jsonl').exists() else []
    toks = [e['event'].get('tokens', [None])[0] for e in ev if 'event' in e and e['event'].get('tokens')]
    minfree = peak = None
    try:
        smp = [json.loads(l) for l in (d / 'samples.jsonl').read_text().splitlines() if l.strip()]
        used = [int(x['gpu_vram_used']) for x in smp if x.get('gpu_vram_used') not in (None, '')]
        tot = int(smp[0]['gpu_vram_total'])
        minfree = round((tot - max(used)) / 2**20, 1) if used else None
        peak = round(max(used) / 2**20, 1) if used else None
    except Exception: pass
    print(json.dumps({'run': d.name, 'blocks': blocks,
                      'prompt_mean_tok_s': round(prev_n / prev_t, 1) if prev_t else None,
                      'last_block_tok_s': blocks[-1]['tok_s'] if blocks else None,
                      'compute_buffer_mib_at_ubatch': comp, 'moe_capacity_granted_free_reserve': cap,
                      'peak_vram_used_mib_samples': peak, 'min_free_vram_mib_samples': minfree, 'generated_tokens_head': toks[:128]}))
