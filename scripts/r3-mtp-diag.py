#!/usr/bin/env python3
"""R3 MTP diagnosis: for each of A/B x MTP off/on start ONE patched llama-server (port ab.PORT, never production), send the SAME natural prompt twice
(temperature 0, seed 1, cache_prompt=false, id_slot 0), then a different short prompt, then the first prompt again. Records text, stop reason, tokens, draft stats, top-3 logprobs of the first tokens.
Usage: r3-mtp-diag.py OUT_DIR"""
import importlib.util, json, os, subprocess, sys, time, urllib.request, hashlib, signal
from pathlib import Path
here = Path(__file__).resolve().parent; repo = here.parent
spec = importlib.util.spec_from_file_location('ab', here / 'oneshot-ab.py'); ab = importlib.util.module_from_spec(spec); spec.loader.exec_module(ab)
out = Path(sys.argv[1]).resolve(); out.mkdir(parents=True, exist_ok=True)
bind = repo / 'build-hip-gfx1030-qsa/bin'
md = ab.D / 'models/Qwen3.8-Flash-Next-Avalon-IQ3KS'; model = md / 'Qwen3.8-Flash-Next-Avalon-IQ3KS-00001-of-00033.gguf'
root = subprocess.check_output([str(ab.P / 'rocm-7.14.1-venv/bin/rocm-sdk'), 'path', '--root'], text=True).strip()
text = open(repo / 'README.md').read()[:7000]   # natural English prose from the repo README, no filler code
P1 = 'Read the following text and then answer.\n\n' + text + '\n\nQuestion: summarise the text above in three short sentences.\nAnswer:'
P2 = 'Write one sentence about why the sky is blue.'
def req(body):
    r = urllib.request.Request(f'http://127.0.0.1:{ab.PORT}/completion', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(r, timeout=900))
allres = []
for arm, mtp in (('A', False), ('A', True), ('B', False), ('B', True)):
    ab.AB_ENV.clear(); ab.AB_ENV.update({'GGML_SCHED_BORROW_COMPUTE': '1', 'LLAMA_QSA_TILES': '0', 'LLAMA_QSA_TILES_ALL': '0'})
    ab.AB_ENV.update({'LLAMA_QSA_RDNA2_PREFILL': '0'} if arm == 'A' else {'LLAMA_QSA_RDNA2_PREFILL': '1', 'LLAMA_QSA_RDNA2_MIN_KV': '1024'}); ab.BIN_DIR = bind
    env = ab.child_environment(os.environ, None, None)
    env.update(ROCM_PATH=root, HIP_PATH=root, LD_LIBRARY_PATH=str(bind) + ':' + root + '/lib:' + root + '/lib/llvm/lib', GPU_MAX_HW_QUEUES='2', HSA_ENABLE_SDMA='1')
    argv = [str(bind / 'llama-server'), '-m', str(model), '-c', '262144', '-b', '8192', '-ub', '4096', '-fa', 'on', '-ctk', 'q8_0', '-ctv', 'q4_0', '--parallel', '1', '--host', '127.0.0.1', '--port', str(ab.PORT), '-t', '6', '-lv', '4', '--jinja']
    if mtp: argv += ['-md', str(md / 'MTP/mtp-Qwen3.8-Flash-Next-shared-IQ3K-IQ4NL-FR65k.gguf'), '--spec-type', 'draft-mtp', '--spec-draft-n-max', '2', '-ubd', '256', '-ctkd', 'q8_0', '-ctvd', 'q8_0']
    tag = f"{arm}-mtp{'ON' if mtp else 'OFF'}"; log = open(out / f'{tag}.server.log', 'w')
    p = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=md, start_new_session=True)
    rows = []; err = None
    try:
        t0 = time.time()
        while True:
            if p.poll() is not None: raise RuntimeError('server exited')
            try:
                if urllib.request.urlopen(f'http://127.0.0.1:{ab.PORT}/health', timeout=3).status == 200: break
            except Exception: time.sleep(3)
            if time.time() - t0 > 1500: raise RuntimeError('health timeout')
        for i, pr in enumerate((P1, P1, P2, P1)):
            r = req({'prompt': pr, 'n_predict': 48, 'temperature': 0, 'seed': 1, 'cache_prompt': False, 'id_slot': 0, 'n_probs': 3})
            tm = r.get('timings', {})
            first = [[(c['token'], round(c['logprob'], 3)) for c in t.get('top_logprobs', [])[:3]] for t in r.get('completion_probabilities', [])[:4]]
            rows.append({'req': i + 1, 'prompt': 'P1' if pr is P1 else 'P2', 'prompt_n': tm.get('prompt_n'), 'predicted_n': tm.get('predicted_n'), 'stop_type': r.get('stop_type'), 'stop_word': r.get('stopping_word'),
                         'draft_n': tm.get('draft_n'), 'draft_acc': tm.get('draft_n_accepted'), 'text_sha': hashlib.sha256(r['content'].encode()).hexdigest()[:12], 'text': r['content'][:160], 'first4_top3': first})
            print(tag, rows[-1]['req'], rows[-1]['prompt'], rows[-1]['prompt_n'], rows[-1]['predicted_n'], rows[-1]['stop_type'], rows[-1]['text_sha'], repr(rows[-1]['text'][:50]), flush=True)
    except Exception as e: err = repr(e); print(tag, 'ERROR', err, flush=True)
    finally:
        try: os.killpg(p.pid, signal.SIGINT); p.wait(timeout=60)
        except Exception: os.killpg(p.pid, signal.SIGKILL)
        time.sleep(5)
    allres.append({'config': tag, 'env': {k: env.get(k) for k in env if k.startswith(('LLAMA_QSA', 'GGML_SCHED'))}, 'argv': argv, 'rows': rows, 'error': err})
json.dump(allres, open(out / 'mtp-diag.json', 'w'), indent=1)
