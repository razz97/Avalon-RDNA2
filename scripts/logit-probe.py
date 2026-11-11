#!/usr/bin/env python3
"""Greedy / teacher-forced logit probe on the real model (isolated server, port 18081, does not touch services).

Starts a fresh llama-server (arm A = baseline, B = LLAMA_QSA_RDNA2_PREFILL=1, extra env via --env JSON), builds the
same prompt as scripts/oneshot-ab.py (hash recorded), prefills it once and then asks for ONE token at a time with
n_probs=5 (cache_prompt reuse, so only the new token is evaluated: decode path).
  - free run: the next prefix token is the arm's own argmax
  - --force FILE: the prefix is taken from the token list of FILE (a previous probe's "tokens"), so two arms are compared
    on exactly the same prefix and their top-5 logprobs/margins are comparable position by position.
Output JSON: prompt hash, tokens, per position top-5 [id, logprob], margin top1-top2.
Usage: logit-probe.py --arm A|B --prompt-tokens N --steps S --out FILE [--force FILE] [--mtp] [--ubatch N] [--env JSON]
"""
import argparse, hashlib, importlib.util, json, os, signal, subprocess, sys, time, types, urllib.request
from pathlib import Path

here = Path(__file__).parent
spec = importlib.util.spec_from_file_location('ab', here / 'oneshot-ab.py')
ab = importlib.util.module_from_spec(spec); spec.loader.exec_module(ab)

a = argparse.ArgumentParser()
a.add_argument('--arm', choices=['A', 'B'], required=True)
a.add_argument('--prompt-tokens', type=int, default=40960)
a.add_argument('--steps', type=int, default=33)
a.add_argument('--out', required=True)
a.add_argument('--force')
a.add_argument('--mtp', action='store_true')
a.add_argument('--batch', type=int, default=8192)
a.add_argument('--ubatch', type=int, default=4096)
a.add_argument('--env', default='{}')
a.add_argument('--bin-dir')
x = a.parse_args()

common = {'GGML_SCHED_BORROW_COMPUTE': '1', 'LLAMA_QSA_TILES': '0', 'LLAMA_QSA_TILES_ALL': '0'}
armenv = {'LLAMA_QSA_RDNA2_PREFILL': '0'} if x.arm == 'A' else {'LLAMA_QSA_RDNA2_PREFILL': '1', 'LLAMA_QSA_RDNA2_MIN_KV': '32768'}
ab.AB_ENV.clear(); ab.AB_ENV.update(common); ab.AB_ENV.update(armenv); ab.AB_ENV.update(json.loads(x.env))
if x.bin_dir:
    ab.BIN_DIR = Path(x.bin_dir)

cfg = types.SimpleNamespace(flash_attn=True, ctk='q8_0', ctv='q4_0', ctkd='q8_0', ctvd='q8_0', jinja=True, fit_target=None, cache_ram=None,
                            offline=False, no_warmup=False, draft_nmax=2, draft_ubatch=256, moe_ring_prefill=None, moe_reserve_mib=None)
if ab.games():
    sys.exit('competing processes: ' + repr(ab.games()))
env = ab.child_environment(os.environ, None, None)
root = subprocess.check_output([str(ab.P / 'rocm-7.14.1-venv/bin/rocm-sdk'), 'path', '--root'], text=True).strip()
env.update(ROCM_PATH=root, HIP_PATH=root, LD_LIBRARY_PATH=root + '/lib:' + root + '/lib/llvm/lib', GPU_MAX_HW_QUEUES='2', HSA_ENABLE_SDMA='1')
model = ab.D / 'models/Qwen3.8-Flash-Next-Avalon-IQ3KS/Qwen3.8-Flash-Next-Avalon-IQ3KS-00001-of-00033.gguf'
argv = ab.build_server_argv(model, 6, x.batch, x.ubatch, x.mtp, cfg)
outp = Path(x.out); outp.parent.mkdir(parents=True, exist_ok=True)
log = open(str(outp) + '.engine.log', 'w')
srv = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=model.parent, start_new_session=True)
PORT = ab.PORT


def post(path, obj, timeout=None):
    r = urllib.request.Request(f'http://127.0.0.1:{PORT}{path}', data=json.dumps(obj).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.load(f)


result = {'arm': x.arm, 'mtp': x.mtp, 'ubatch': x.ubatch, 'env': {**armenv, **json.loads(x.env)}, 'argv': [str(v) for v in argv]}
try:
    t0 = time.time()
    while True:
        if srv.poll() is not None:
            raise RuntimeError('server exited rc=%s' % srv.returncode)
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=1) as f:
                if f.status == 200:
                    break
        except Exception:
            time.sleep(1)
        if time.time() - t0 > 600:
            raise RuntimeError('server start timeout')
    records = max(4000, (x.prompt_tokens + 24) // 25 + 100)
    msgs = [{'role': 'system', 'content': 'You are a helpful programming assistant. Answer the user directly and completely.'},
            {'role': 'user', 'content': ab.long_task(records)}]
    rendered = post('/apply-template', {'messages': msgs, 'chat_template_kwargs': {'enable_thinking': False}})['prompt']
    toks = post('/tokenize', {'content': rendered, 'add_special': False, 'parse_special': True})['tokens']
    prompt = toks[:x.prompt_tokens - 200] + toks[-200:]
    assert len(prompt) == x.prompt_tokens
    result['prompt_sha256'] = hashlib.sha256(json.dumps(prompt, separators=(',', ':')).encode()).hexdigest()
    forced = json.load(open(x.force))['tokens'] if x.force else None
    gen, steps = [], []
    for k in range(x.steps):
        pre = prompt + (forced[:k] if forced else gen)
        t1 = time.time()
        r = post('/completion', {'prompt': pre, 'n_predict': 1, 'temperature': 0, 'cache_prompt': True, 'n_probs': 5,
                                 'return_tokens': True, 'ignore_eos': True, 'samplers': ['top_k'], 'top_k': 1})
        cp = r.get('completion_probabilities') or []
        top = []
        if cp:
            e = cp[0]
            tl = e.get('top_logprobs') or e.get('top_probs') or []
            top = [[t.get('id'), t.get('logprob', t.get('prob'))] for t in tl]
        tok = (r.get('tokens') or [None])[0]
        gen.append(tok)
        steps.append({'k': k, 'token': tok, 'top5': top, 'margin': (top[0][1] - top[1][1]) if len(top) > 1 and top[0][1] is not None else None,
                      'prompt_n': (r.get('timings') or {}).get('prompt_n'), 'cache_n': (r.get('timings') or {}).get('cache_n'), 's': round(time.time() - t1, 2)})
        print(k, tok, steps[-1]['margin'], steps[-1]['prompt_n'], steps[-1]['s'], flush=True)
    result.update(tokens=gen, steps=steps, forced=bool(forced))
except Exception as e:
    result['error'] = repr(e)
    print('ERROR', repr(e), file=sys.stderr)
finally:
    try:
        os.killpg(srv.pid, signal.SIGTERM); srv.wait(timeout=60)
    except Exception:
        try:
            os.killpg(srv.pid, signal.SIGKILL)
        except Exception:
            pass
    log.close()
    outp.write_text(json.dumps(result, indent=1) + '\n')
sys.exit(1 if 'error' in result else 0)
