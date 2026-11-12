#!/usr/bin/env python3
"""R3 S6: 6 chained short requests (1K-8K tokens, MTP ON) against ONE patched llama-server (arm B default, or A via 2nd arg; B uses MIN_KV=1024, port 18081, never production).
Request 6 repeats request 1 with cache_prompt=false: outputs must be identical (no state leak). Checks finish, tokens, MTP draft stats, VRAM free, kernel log.
Usage: r3-smoke.py OUT_DIR"""
import importlib.util, json, os, subprocess, sys, time, urllib.request, hashlib, signal, threading
from pathlib import Path
here = Path(__file__).resolve().parent; repo = here.parent
spec = importlib.util.spec_from_file_location('ab', here / 'oneshot-ab.py'); ab = importlib.util.module_from_spec(spec); spec.loader.exec_module(ab)
ARM = sys.argv[2] if len(sys.argv) > 2 else 'B'
out = Path(sys.argv[1]).resolve(); out.mkdir(parents=True, exist_ok=True)
bind = repo / 'build-main/bin'
ab.AB_ENV.clear(); ab.AB_ENV.update({'GGML_SCHED_BORROW_COMPUTE': '1', 'LLAMA_QSA_TILES': '0', 'LLAMA_QSA_TILES_ALL': '0', **({'LLAMA_QSA_RDNA2_PREFILL': '1', 'LLAMA_QSA_RDNA2_MIN_KV': '1024'} if ARM == 'B' else {'LLAMA_QSA_RDNA2_PREFILL': '0'})}); ab.BIN_DIR = bind
if ab.games(): sys.exit('competing processes')
env = ab.child_environment(os.environ, None, None)
root = subprocess.check_output([str(ab.P / 'rocm-7.14.1-venv/bin/rocm-sdk'), 'path', '--root'], text=True).strip()
env.update(ROCM_PATH=root, HIP_PATH=root, LD_LIBRARY_PATH=str(bind) + ':' + root + '/lib:' + root + '/lib/llvm/lib', GPU_MAX_HW_QUEUES='2', HSA_ENABLE_SDMA='1')
md = ab.D / 'models/Qwen3.8-Flash-Next-Avalon-IQ3KS'; model = md / 'Qwen3.8-Flash-Next-Avalon-IQ3KS-00001-of-00033.gguf'
PORT = ab.PORT
argv = [str(bind / 'llama-server'), '-m', str(model), '-c', '262144', '-b', '8192', '-ub', '4096', '-fa', 'on', '-ctk', 'q8_0', '-ctv', 'q4_0', '--parallel', '1',
        '--host', '127.0.0.1', '--port', str(PORT), '-t', '6', '-lv', '4', '--jinja', '-md', str(md / 'MTP/mtp-Qwen3.8-Flash-Next-shared-IQ3K-IQ4NL-FR65k.gguf'),
        '--spec-type', 'draft-mtp', '--spec-draft-n-max', '2', '-ubd', '256', '-ctkd', 'q8_0', '-ctvd', 'q8_0']
def card():
    for c in Path('/sys/class/drm').glob('card[0-9]'):
        if (c / 'device/mem_info_vram_used').exists(): return c
vc = card(); total = int((vc / 'device/mem_info_vram_total').read_text()); mf = [total]; stop = threading.Event()
def poll():
    while not stop.is_set():
        try: mf[0] = min(mf[0], total - int((vc / 'device/mem_info_vram_used').read_text()))
        except Exception: pass
        time.sleep(1)
threading.Thread(target=poll, daemon=True).start()
since = time.strftime('%Y-%m-%d %H:%M:%S')
log = open(out / 'server.log', 'w')
p = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=md, start_new_session=True)
def req(path, body, timeout=900):
    r = urllib.request.Request(f'http://127.0.0.1:{PORT}{path}', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(r, timeout=timeout))
res = []; err = None
try:
    t0 = time.time()
    while True:
        if p.poll() is not None: raise RuntimeError('server exited rc=%s' % p.returncode)
        try:
            if urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=3).status == 200: break
        except Exception: time.sleep(3)
        if time.time() - t0 > 1500: raise RuntimeError('health timeout')
    src = ''.join(open(f).read() for f in sorted((repo / 'ggml/src/ggml-cuda').glob('fattn*.cu')))   # non-sensitive repo code as filler text
    sizes = [1000, 2500, 4000, 8000, 1500]   # approx tokens (chars/3.3)
    prompts = [src[i * 1000: i * 1000 + int(n * 3.3)] + '\n\n// Question: what does the code above do? Answer in two sentences.\n' for i, n in enumerate(sizes)]
    prompts.append(prompts[0])       # request 6 == request 1
    for i, pr in enumerate(prompts):
        t = time.time()
        r = req('/completion', {'prompt': pr, 'n_predict': 32, 'temperature': 0, 'seed': 1, 'cache_prompt': False, 'id_slot': 0})
        tm = r.get('timings', {})
        res.append({'req': i + 1, 'prompt_n': tm.get('prompt_n'), 'predicted_n': tm.get('predicted_n'), 'stop_type': r.get('stop_type'), 'draft_n': tm.get('draft_n'), 'draft_accepted': tm.get('draft_n_accepted'),
                    'prompt_per_s': tm.get('prompt_per_second'), 'text_sha256': hashlib.sha256(r['content'].encode()).hexdigest()[:16], 'text_head': r['content'][:80], 's': round(time.time() - t, 1), 'vram_min_free_mib_so_far': mf[0] >> 20})
        print(res[-1], flush=True)
except Exception as e:
    err = repr(e)
finally:
    stop.set()
    try: os.killpg(p.pid, signal.SIGINT); p.wait(timeout=60)
    except Exception: os.killpg(p.pid, signal.SIGKILL)
jk = subprocess.run(['journalctl', '-k', '--since', since, '--no-pager'], capture_output=True, text=True).stdout
(out / 'journalctl-k.txt').write_text(jk)
bad = [l for l in jk.splitlines() if any(k in l.lower() for k in ('amdgpu', 'gpu reset', 'hang', 'oom', 'page fault'))]
checks = {'all_finished': bool(res) and len(res) == 6 and all(r['stop_type'] in ('limit', 'eos', 'word') and r['predicted_n'] for r in res),
          'req1_equals_req6': len(res) == 6 and res[0]['text_sha256'] == res[5]['text_sha256'], 'mtp_drafts_seen': any(r['draft_n'] for r in res),
          'prefill_ge_64_each': all((r['prompt_n'] or 0) >= 64 for r in res), 'kernel_log_issues': bad, 'vram_min_free_mib': mf[0] >> 20}
json.dump({'argv': argv, 'env_flags': {k: env.get(k) for k in env if k.startswith(('LLAMA_QSA', 'GGML_SCHED'))}, 'binary_realpath': str((bind / 'llama-server').resolve()),
           'binary_sha256': hashlib.sha256(open(bind / 'llama-server', 'rb').read()).hexdigest(), 'requests': res, 'checks': checks, 'error': err}, open(out / 'smoke.json', 'w'), indent=1)
print(json.dumps(checks), err)
