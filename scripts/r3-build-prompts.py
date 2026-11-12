#!/usr/bin/env python3
"""R3: build the token-id spec for llama-qsa-nll. R2 prompts (40960 / 126972) are rebuilt exactly (hash-checked against R2),
the others are synthetic / repo-source text (no personal data). Usage: r3-build-prompts.py OUTDIR
Writes OUTDIR/{short,long}.spec (REC name n_tokens n_pos / ids / positions) and OUTDIR/prompts.json (names, groups, hashes)."""
import hashlib, importlib.util, json, os, random, subprocess, sys
from pathlib import Path
here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('ab', here / 'oneshot-ab.py'); ab = importlib.util.module_from_spec(spec); spec.loader.exec_module(ab)
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
BIN = os.environ['QSA_NLL_BIN']; MODEL = os.environ['QSA_MODEL']
env = {**os.environ, 'QSA_NLL_MODE': 'tok'}
def tokenise(text, name):
    fi, fo = out / f'{name}.txt', out / f'{name}.ids'
    fi.write_text(text)
    subprocess.run([BIN, '-m', MODEL], env={**env, 'QSA_NLL_IN': str(fi), 'QSA_NLL_OUT': str(fo)}, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return [int(x) for x in fo.read_text().split()]
SYS = 'You are a helpful programming assistant. Answer the user directly and completely.'
def render(user):   # Qwen chat template with enable_thinking=False (verified against R2 hash below)
    return f'<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
def r2_prompt(n):
    toks = tokenise(render(ab.long_task(max(4000, (n + 24) // 25 + 100))), f'r2tpl-{n}')
    return toks[:n - 200] + toks[-200:]
h = lambda ids: hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
prompts = []
def add(name, kind, ids, group): prompts.append({'name': name, 'kind': kind, 'ids': ids, 'group': group})
p40 = r2_prompt(40960)
assert h(p40).startswith('70ad59c7c7c79d7f8b7bf87fbe81cdbac45c3a031ccf23b9cc7a1c44b116ba52'), 'R2 40K prompt not reproduced: ' + h(p40)
# retrieval: facts table, then the question and the (teacher-forced) answer at the end
def retrieval(n, seed, where):
    rnd = random.Random(seed)
    base = 'Inventory ledger. Each line states the registration code of one device.\n'
    codes, lines = {}, []
    for i in range(n // 18):
        c = '%s-%04d-%s' % (rnd.choice(['KR', 'ZT', 'VX', 'QM']), rnd.randrange(10000), ''.join(rnd.choice('abcdefghjkmnpqrstuvwxyz') for _ in range(5)))
        codes[i] = c
        lines.append(f'Device D{i:05d} (rack {rnd.randrange(1, 90)}, row {rnd.randrange(1, 40)}) has registration code {c}.')
    target = {'start': 3, 'middle': int(len(lines) * 0.3)}[where]
    q = f'\nQuestion: what is the registration code of device D{target:05d}?\nAnswer: Device D{target:05d} has registration code {codes[target]}.'
    qi = tokenise(q, f'ret-q-{seed}')
    body = tokenise(base + '\n'.join(lines) + '\n', f'ret-body-{seed}')
    ids = body[:n - len(qi)] + qi
    assert len(ids) == n
    marker = tokenise(f'Device D{target:05d} (rack', f'ret-m-{seed}')
    assert any(body[k:k + len(marker)] == marker for k in range(0, n - len(qi))), 'evidence line was truncated away'
    return ids
add('S3b-retrieval-40K', 'retrieval(evidence at start)', retrieval(40960, 11, 'start'), 'S3')
add('S1a-code-8K', 'code task (R2 long_task, truncated to 8192)', None, 'S1')
add('S1b-retrieval-8K', 'retrieval(evidence in middle)', retrieval(8192, 12, 'middle'), 'S1')
add('S2-structured-16K', 'structured JSON lines', None, 'S2')
# code 8K: repo CUDA source; structured 16K: synthetic jsonl
src = ''.join(open(f).read() for f in sorted(Path(here.parent / 'ggml/src/ggml-cuda').glob('*.cu')))
prompts[1]['ids'] = r2_prompt(8192)   # first 7992 + last 200 tokens of the R2 long_task prompt (repo source code is memorised by the model: NLL~0, insensitive)
rnd = random.Random(5)
js = '\n'.join(json.dumps({'id': i, 'ts': 1700000000 + i * 37, 'user': 'u%04d' % rnd.randrange(3000), 'event': rnd.choice(['login', 'upload', 'purchase', 'logout', 'error']), 'amount': round(rnd.random() * 100, 2), 'tags': [rnd.choice(['a', 'b', 'c', 'd']) for _ in range(rnd.randrange(1, 4))]}) for i in range(2500))
j = tokenise(js, 'json'); assert len(j) >= 16384; prompts[3]['ids'] = j[:16384]
add('S3a-code-40K(R2 prompt)', 'code task (R2 long_task, hash-checked)', p40, 'S3')
add('S4-127K(R2 prompt)', 'code task (R2 long_task)', r2_prompt(126972), 'S4')
def positions(n):
    pos = []
    for frac in (0.25, 0.5, 0.75):
        c0 = int(n * frac); pos += [c0 + k * 97 for k in range(8)]
    pos += [n - 4096 + k * 500 + 7 for k in range(8)]
    pos += list(range(n - 32, n))      # last 32 targets: fixed continuation
    return sorted(set(pos))
order = ['S1a-code-8K', 'S1b-retrieval-8K', 'S2-structured-16K', 'S3a-code-40K(R2 prompt)', 'S3b-retrieval-40K', 'S4-127K(R2 prompt)']
meta = []
for arm_file, names in (('short', order[:3] + ['S1a-code-8K@rep']), ('long', order[3:])):
    with open(out / f'{arm_file}.spec', 'w') as f:
        for nm in names:
            base_nm, _, rep = nm.partition('@')
            p = next(x for x in prompts if x['name'] == base_nm); ids = p['ids']; pos = positions(len(ids))
            nm2 = base_nm.replace('(', '_').replace(')', '').replace(' ', '_') + ('-' + rep if rep else '')
            f.write(f'REC {nm2} {len(ids)} {len(pos)}\n' + ' '.join(map(str, ids)) + '\n' + ' '.join(map(str, pos)) + '\n')
            meta.append({'name': nm2, 'file': arm_file, 'kind': p['kind'], 'group': p['group'], 'n_tokens': len(ids), 'prompt_tokens_sha256': h(ids),
                         'target_ids_sha256': h([ids[q] for q in pos]), 'n_targets': len(pos), 'positions': pos})
(out / 'prompts.json').write_text(json.dumps(meta, indent=1))
print(json.dumps([{k: v for k, v in m.items() if k != 'positions'} for m in meta], indent=1))
