#!/usr/bin/env python3
"""R3: run llama-qsa-nll once for one arm (one process, several prompt records) with the same engine environment as the R2 harness.
Usage: r3-run.py --arm A|B --spec FILE --out DIR [--min-kv N] [--wrap 'rocprofv3 ... --'] [--meta prompts.json]
Always uses the PATCHED P2 binary dir (R3_BIN_DIR, default build-hip-gfx1030-qsa/bin); refuses anything else."""
import argparse, hashlib, importlib.util, json, os, signal, subprocess, shlex, sys, threading, time, types
from pathlib import Path
here = Path(__file__).resolve().parent; repo = here.parent
spec = importlib.util.spec_from_file_location('ab', here / 'oneshot-ab.py'); ab = importlib.util.module_from_spec(spec); spec.loader.exec_module(ab)
a = argparse.ArgumentParser()
a.add_argument('--arm', required=True, choices='AB'); a.add_argument('--spec', required=True); a.add_argument('--out', required=True)
a.add_argument('--min-kv', default='32768'); a.add_argument('--wrap'); a.add_argument('--tag', default=''); a.add_argument('--ub', default='4096')
x = a.parse_args()
binp = Path(os.environ.get('R3_BIN_DIR', repo / 'build-hip-gfx1030-qsa/bin')).resolve() / 'llama-qsa-nll'
assert str(binp).startswith(str(repo / 'build-hip-gfx1030-qsa')), 'must be the patched build'
out = Path(x.out).resolve(); out.mkdir(parents=True, exist_ok=True)
common = {'GGML_SCHED_BORROW_COMPUTE': '1', 'LLAMA_QSA_TILES': '0', 'LLAMA_QSA_TILES_ALL': '0'}
armenv = {'LLAMA_QSA_RDNA2_PREFILL': '0'} if x.arm == 'A' else {'LLAMA_QSA_RDNA2_PREFILL': '1', 'LLAMA_QSA_RDNA2_MIN_KV': x.min_kv}
ab.AB_ENV.clear(); ab.AB_ENV.update(common); ab.AB_ENV.update(armenv); ab.BIN_DIR = binp.parent
if ab.games(): sys.exit('competing processes: ' + repr(ab.games()))
env = ab.child_environment(os.environ, None, None)
root = subprocess.check_output([str(ab.P / 'rocm-7.14.1-venv/bin/rocm-sdk'), 'path', '--root'], text=True).strip()
env.update(ROCM_PATH=root, HIP_PATH=root, LD_LIBRARY_PATH=str(binp.parent) + ':' + root + '/lib:' + root + '/lib/llvm/lib', GPU_MAX_HW_QUEUES='2', HSA_ENABLE_SDMA='1',
           QSA_NLL_MODE='run', QSA_NLL_IN=str(Path(x.spec).resolve()), QSA_NLL_OUT=str(out / 'nll.json'))
model = ab.D / 'models/Qwen3.8-Flash-Next-Avalon-IQ3KS/Qwen3.8-Flash-Next-Avalon-IQ3KS-00001-of-00033.gguf'
argv = [str(binp), '-m', str(model), '-c', '262144', '-b', '8192', '-ub', x.ub, '-fa', 'on', '-ctk', 'q8_0', '-ctv', 'q4_0', '--parallel', '1', '-t', '6']
if x.wrap: argv = shlex.split(x.wrap) + argv
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
man = ab.D / 'models/Qwen3.8-Flash-Next-Avalon-IQ3KS/download-manifest.json'
def card():
    for c in Path('/sys/class/drm').glob('card[0-9]'):
        f = c / 'device/mem_info_vram_used'
        if f.exists(): return c
vc = card(); total = int((vc / 'device/mem_info_vram_total').read_text()); minfree = [total]; stop = threading.Event()
def poll():
    while not stop.is_set():
        try: minfree[0] = min(minfree[0], total - int((vc / 'device/mem_info_vram_used').read_text()))
        except Exception: pass
        time.sleep(1)
since = time.strftime('%Y-%m-%d %H:%M:%S')
pl = threading.Thread(target=poll, daemon=True); pl.start()
t0 = time.time()
log = open(out / 'engine.log', 'w')
p = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=model.parent, start_new_session=True)
try: rc = p.wait()
except KeyboardInterrupt:
    os.killpg(p.pid, signal.SIGTERM); rc = -1
stop.set(); log.close()
jk = subprocess.run(['journalctl', '-k', '--since', since, '--no-pager'], capture_output=True, text=True).stdout
(out / 'journalctl-k.txt').write_text(jk)
bad = [l for l in jk.splitlines() if any(k in l.lower() for k in ('amdgpu', 'gpu reset', 'page fault', 'ring', 'oom'))]
meta = {'arm': x.arm, 'tag': x.tag, 'rc': rc, 'elapsed_total_s': time.time() - t0, 'commit': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
        'binary_realpath': str(binp), 'binary_sha256': sha(binp), 'libggml_hip_sha256': sha(binp.parent / 'libggml-hip.so.0.28.0'),
        'model_first_shard': str(model), 'model_manifest_sha256': sha(man), 'argv': argv, 'mtp': False, 'kv_types': 'K=q8_0 V=q4_0',
        'n_ctx': 262144, 'n_batch': 8192, 'n_ubatch': int(x.ub), 'gpu': 'gfx1030', 'rocm': '7.14.1',
        'flags': {k: v for k, v in env.items() if k.startswith(('LLAMA_', 'GGML_', 'GPU_', 'HSA_'))},
        'vram_min_free_mib': minfree[0] // 2**20, 'journal_kernel_suspicious_lines': bad, 'spec_sha256': sha(x.spec), 'wrapped': bool(x.wrap)}
(out / 'meta.json').write_text(json.dumps(meta, indent=1))
print(json.dumps({k: meta[k] for k in ('arm', 'tag', 'rc', 'elapsed_total_s', 'vram_min_free_mib', 'journal_kernel_suspicious_lines', 'flags')}, indent=1))
