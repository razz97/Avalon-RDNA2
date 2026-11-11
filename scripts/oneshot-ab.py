#!/usr/bin/env python3
"""One-shot Avalon smoke, derived from coder-q8-smoke emergency guards."""
import argparse, hashlib, importlib.util, json, os, re, select, signal, socket, subprocess, sys, threading, time, urllib.request
from collections import deque
from pathlib import Path
D=Path('/home/alex/.local/share/lifeos-inference')
P=D/'avalon-q8-prep-e09e775'
BIN_DIR=Path(os.environ.get('AB_BIN_DIR', str(P/'build-hip-trace-o3-gfx1030/bin')))
AB_ENV=json.loads(os.environ.get('AB_ENV','{}'))
O=D/'qwen-v31-baseline/diagnostics/observe.py'
spec=importlib.util.spec_from_file_location('observe',O); obs=importlib.util.module_from_spec(spec); spec.loader.exec_module(obs)
PORT=18081
LONG_TASK=('Read the following synthetic work-queue records and summarize the important safety invariants.\n'+''.join(f'Record {i:03d}: The worker stores id, retry count, timestamp, payload and acknowledgement before committing a durable queue update.\n' for i in range(73))+'Describe recovery after a crash and give practical pseudocode.')
TASKS=[('performance',0)]
PROMPT=LONG_TASK
CTX=262144
VRAM_MIN_MIB=100

def long_task(records):
 return ('Read the following synthetic work-queue records and summarize the important safety invariants.\n'+''.join(f'Record {i:03d}: The worker stores id, retry count, timestamp, payload and acknowledgement before committing a durable queue update.\n' for i in range(records))+'Describe recovery after a crash and give practical pseudocode.')

def find_record_count(token_count, target=99000):
 low,high=1,4000
 while low<=high:
  mid=(low+high)//2; n=token_count(mid)
  if n<target-50: low=mid+1
  elif n>target: high=mid-1
  else: return mid,n
 raise ValueError('unable to construct prompt within 50 tokens of target')

def validate_final(kind, final, mtp_enabled, expected=512):
 count=final.get('tokens_predicted'); reason=final.get('stop_type'); timings=final.get('timings') or {}; settings=final.get('generation_settings') or {}
 if not isinstance(count,int) or count<0 or timings.get('predicted_n')!=count or reason not in ('eos','limit','word'): raise ValueError('invalid final event')
 if settings.get('ignore_eos') is not (kind=='performance'): raise ValueError('effective ignore_eos mismatch')
 if kind=='performance' and (count!=expected or reason!='limit'): raise ValueError('invalid forced throughput sample')
 d,a=timings.get('draft_n'),timings.get('draft_n_accepted')
 if (d is None)!=(a is None) or (d is not None and (not isinstance(d,int) or not isinstance(a,int) or not 0<=a<=d)): raise ValueError('invalid MTP counters')
 return reason
FAULT=re.compile(r'(?i)(\bamdgpu\b[^\n]*\b(fault|ring[^\n]*\btimeout|reset)\b|\bGPU\b[^\n]*\b(fault|reset|timeout)\b|\bout of memory\b|\boom.kill\b|\bpage allocation failure\b)')

def stop(p):
 if p and p.poll() is None:
  try: os.killpg(p.pid,signal.SIGTERM)
  except ProcessLookupError: pass
  try: p.wait(timeout=8)
  except subprocess.TimeoutExpired:
   try: os.killpg(p.pid,signal.SIGKILL)
   except ProcessLookupError: pass
   p.wait(timeout=8)

def stop_owned_group(p):
 if not p: return True
 try: os.killpg(p.pid,signal.SIGKILL)
 except ProcessLookupError: pass
 try: p.wait(timeout=8)
 except subprocess.TimeoutExpired: return False
 for _ in range(40):
  ps=subprocess.run(['ps','-eo','pgid=,stat='],capture_output=True,text=True,check=True)
  alive=[x for x in ps.stdout.splitlines() if int(x.split()[0])==p.pid and not x.split()[1].startswith('Z')]
  if not alive: return True
  time.sleep(.1)
 return False

def games(owned_pid=None):
 p=subprocess.run(['ps','-eo','pid=,comm='],capture_output=True,text=True,check=True)
 found=[]
 for x in p.stdout.splitlines():
  if re.search(r'(?i)\b(steam|gamescope|proton|wine|lutris|hyprland|sway|xorg|llama-server|strata)\b',x):
   pid=int(x.split()[0])
   try:
    if owned_pid and os.getpgid(pid)==owned_pid: continue
   except ProcessLookupError: continue
   found.append(x)
 return found

def observe_prefill(state, line, now_ns):
 m=re.search(r'prompt processing, n_tokens\s*=\s*(\d+)',line)
 if m and int(m.group(1))>state['prefill_tokens']:
  state['prefill_tokens']=int(m.group(1)); state['prefill_progress_ns']=max(state['prefill_progress_ns'],now_ns)
 m=re.search(r'\[TRACE_(?:SGEMM|FATTN)\] start_us=(\d+) end_us=(\d+) ',line)
 if m:
  start,end=map(int,m.groups())
  if start*1000>=state['request_started_ns'] and start<=end and end*1000<=now_ns and end>state['prefill_trace_us']:
   state['prefill_trace_us']=end; state['prefill_progress_ns']=max(state['prefill_progress_ns'],end*1000)

def validate_context(text, ctx, state, out=None):
 actual=[int(x) for x in re.findall(r'llama_context: n_ctx\s*=\s*(\d+)',text)]
 slot=[int(x) for x in re.findall(r'n_ctx_slot\s*=\s*(\d+)',text)]
 batches=[int(x) for x in re.findall(r'llama_context: n_batch\s*=\s*(\d+)',text)]
 ubatches=[int(x) for x in re.findall(r'llama_context: n_ubatch\s*=\s*(\d+)',text)]
 kv_types=re.findall(r'llama_kv_cache: size[^\n]*K \(([^)]+)\):[^\n]*V \(([^)]+)\):',text)
 flash=re.findall(r'llama_context: flash_attn\s*=\s*(\w+)',text)
 parsed={'requested':ctx,'context_values':actual,'slot_values':slot,'batch_values':batches,'ubatch_values':ubatches,'kv_types':kv_types,'flash_attention':flash}
 if out is not None: (out/'context-validation.json').write_text(json.dumps(parsed)+'\n')
 if len(actual)!=(2 if state['mtp_enabled'] else 1) or not slot or any(x!=ctx for x in actual+slot): raise RuntimeError('context mismatch or missing target/draft evidence')
 if not batches or not ubatches or batches[0]!=state['batch'] or ubatches[0]!=state['ubatch']: raise RuntimeError('target batch/microbatch mismatch')
 expected_kv=[(state['ctk'],state['ctv'])]*2+([(state['ctkd'],state['ctvd'])]*2 if state['mtp_enabled'] else [])
 if kv_types!=expected_kv: raise RuntimeError('target or draft KV type mismatch')
 expected_flash='enabled' if state.get('flash_attn',True) else 'disabled'
 if len(flash)!=(2 if state['mtp_enabled'] else 1) or any(x!=expected_flash for x in flash): raise RuntimeError('flash attention target/draft mismatch or missing evidence')
 return parsed

def request(state, out, reuse):
 try:
  while not state['stop']:
   try:
    with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health',timeout=1) as r:
     if r.status==200: break
   except Exception: time.sleep(.5)
  if state['stop']: return
  state['health_ns']=time.monotonic_ns()
  if not state['synthetic']:
   text=(out/'engine.log').read_text(errors='replace')
   validate_context(text,CTX,state,out)
  state['phase']='inference'
  def prepare():
   if state['stop']: raise RuntimeError('supervisor stopped')
   records=max(4000,(state['prompt_tokens']+24)//25+100)
   messages=[{'role':'system','content':'You are a helpful programming assistant. Answer the user directly and completely.'},{'role':'user','content':long_task(records)}]
   template_request={'messages':messages,'chat_template_kwargs':{'enable_thinking':False}}
   tr=urllib.request.Request(f'http://127.0.0.1:{PORT}/apply-template',data=json.dumps(template_request).encode(),headers={'Content-Type':'application/json'})
   with urllib.request.urlopen(tr,timeout=15) as r: rendered=json.load(r)['prompt']
   if not isinstance(rendered,str) or '<|im_start|>assistant' not in rendered: raise ValueError('template missing assistant generation prompt')
   token_req=urllib.request.Request(f'http://127.0.0.1:{PORT}/tokenize',data=json.dumps({'content':rendered,'add_special':False,'parse_special':True}).encode(),headers={'Content-Type':'application/json'})
   with urllib.request.urlopen(token_req,timeout=15) as tr: tokens=json.load(tr).get('tokens')
   target=state['prompt_tokens']
   if not isinstance(tokens,list) or len(tokens)<target or any(not isinstance(x,int) for x in tokens): raise ValueError('tokenizer returned insufficient or invalid tokens')
   prompt=tokens[:target-200]+tokens[-200:]
   if len(prompt)!=target: raise ValueError('prompt token count mismatch')
   return template_request,prompt,len(tokens),records
  for i,(kind,_) in enumerate(TASKS):
   if state['stop']: return
   template_request,prompt,original_count,records=prepare()
   token_count=len(prompt)
   predict=state['predict']
   if token_count+predict>CTX: raise ValueError('prompt count/context mismatch')
   (out/f'template-request-{i}.json').write_text(json.dumps(template_request,indent=2,ensure_ascii=False)+'\n')
   prompt_hash=hashlib.sha256(json.dumps(prompt,separators=(',',':')).encode()).hexdigest()
   (out/'prompt-plan.json').write_text(json.dumps({'source_records':records,'source_tokens':original_count,'input_tokens':token_count,'construction':f'first {token_count-200} and final 200 tokenizer tokens; retained chat template suffix','token_ids_sha256':prompt_hash,'completion_requests':1,'n_predict':predict},indent=2)+'\n')
   print(f'Prefill: enviando {token_count} tokens; generación: {predict} tokens',flush=True)
   payload={'prompt':prompt,'n_predict':predict,'temperature':0,'cache_prompt':reuse,'stream':True,'return_tokens':True,'ignore_eos':True}
   (out/f'payload-{i}.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
   offset=(out/'engine.log').stat().st_size
   started=time.monotonic_ns(); state.update(request_index=i,request_started_ns=started,last_token_ns=None,prefill_progress_ns=started,prefill_tokens=0,prefill_trace_us=started//1000,progress_log_offset=offset)
   record={'index':i,'kind':kind,'prompt_tokens_tokenizer':token_count,'prompt_sha256':prompt_hash,'start_monotonic_ns':started,'valid':False,'events':[],'response':None}
   current=0; texts=[]; final=None; failure=None
   try:
    req=urllib.request.Request(f'http://127.0.0.1:{PORT}/completion',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with (out/f'events-{i}.jsonl').open('x') as raw, urllib.request.urlopen(req,timeout=None) as r:
     data=[]
     while not state['stop']:
      line=r.readline()
      if not line: break
      line=line.decode('utf-8').rstrip('\r\n')
      if line.startswith('data:'): data.append(line[5:].lstrip())
      elif not line and data:
       obj=json.loads('\n'.join(data)); data=[]; now=time.monotonic_ns()
       raw.write(json.dumps({'received_monotonic_ns':now,'event':obj},ensure_ascii=False)+'\n'); raw.flush()
       if obj.get('stop') is True: final=obj; break
       if obj.get('stop') is not False: raise ValueError('unexpected stream event')
       n=obj.get('tokens_predicted')
       if not isinstance(n,int) or n<current: raise ValueError('invalid token progress')
       if n>current:
        previous=state['last_token_ns']; state['last_token_ns']=now
        record['events'].append({'received_monotonic_ns':now,'tokens_predicted':n,'token_delta':n-current,'gap_ms':None if previous is None else (now-previous)/1e6})
        current=n
       if isinstance(obj.get('content'),str): texts.append(obj['content'])
    if final is None: raise ValueError('stream ended without final event')
    count=final.get('tokens_predicted'); reason=validate_final(kind,final,reuse,predict)
    cached=(final.get('timings') or {}).get('cache_n'); processed=(final.get('timings') or {}).get('prompt_n')
    if cached!=0 or processed!=state['prompt_tokens']: raise ValueError('server did not process the requested fresh prompt tokens')
    record.update(response=final,finish_reason=reason,tokens_generated=count,quality_complete=(reason=='eos') if kind=='natural' else None)
    record['valid']=True
   except Exception as e: failure=e
   finally:
    record['content']=''.join(texts); record['end_monotonic_ns']=time.monotonic_ns(); record['error']=repr(failure) if failure else None
    record['ttft_ms']=(record['events'][0]['received_monotonic_ns']-started)/1e6 if record['events'] else None
    (out/f'request-{i}.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
    state['request_started_ns']=None
   if failure: raise failure
   state['completed']=i+1
  state['done']=True
 except Exception as e: state['error']=repr(e)

def print_summary(out, result):
 print(f"Estado: {result['reason']} | tiempo total: {result['seconds']:.2f} s",flush=True)
 p=out/'request-0.json'
 if not p.exists():
  print('Sin estadísticas completas: no se completó la solicitud.',flush=True); return
 try:
  r=json.loads(p.read_text()); t=(r.get('response') or {}).get('timings') or {}
  if not r.get('valid') or result['reason']!='suite complete':
   print('Solicitud incompleta o interrumpida; no hay velocidades finales verificadas.',flush=True); return
  def speed(n,ms):
   return f'{1000*n/ms:.2f} tok/s' if isinstance(n,(int,float)) and isinstance(ms,(int,float)) and ms>0 else 'N/D'
  ttft=r.get('ttft_ms')
  print(f"Tokens: procesados={t.get('prompt_n', 'N/D')}, caché={t.get('cache_n', 'N/D')}, generados={t.get('predicted_n', 'N/D')}",flush=True)
  print(f"Prefill: {speed(t.get('prompt_n'),t.get('prompt_ms'))} | Decode: {speed(t.get('predicted_n'),t.get('predicted_ms'))} | TTFT: {f'{ttft/1000:.2f} s' if isinstance(ttft,(int,float)) else 'N/D'}",flush=True)
 except (OSError,ValueError,TypeError) as e: print(f'Estadísticas no disponibles: {e}',flush=True)

def child_environment(parent, ring_prefill, reserve_mib):
 env={k:v for k,v in parent.items() if not k.startswith(('LLAMA_ARG_','GGML_','STRATA_')) and k not in ('HSA_ENABLE_SDMA','GPU_MAX_HW_QUEUES','HIP_ENABLE_DEFERRED_LOADING')}
 env.update(AB_ENV)
 if ring_prefill is not None: env['GGML_CUDA_MOE_RING_PREFILL']=str(ring_prefill)
 if reserve_mib is not None: env['GGML_CUDA_MOE_CACHE_RESERVE_MB']=str(reserve_mib)
 return env

def build_server_argv(model, threads, batch, ubatch, mtp_reuse, cfg):
 argv=[str(BIN_DIR/'llama-server'),'-m',str(model),'-c',str(CTX),'-b',str(batch),'-ub',str(ubatch),'-fa','on' if cfg.flash_attn else 'off','-ctk',cfg.ctk,'-ctv',cfg.ctv,'--parallel','1','--host','127.0.0.1','--port',str(PORT),'-t',str(threads),'-lv','4']
 argv.append('--jinja' if cfg.jinja else '--no-jinja')
 if cfg.fit_target is not None: argv += ['--fit-target',str(cfg.fit_target)]
 if cfg.cache_ram is not None: argv += ['--cache-ram',str(cfg.cache_ram)]
 if cfg.offline: argv.append('--offline')
 if cfg.no_warmup: argv.append('--no-warmup')
 if mtp_reuse: argv += ['-md',str(model.parent/'MTP/mtp-Qwen3.8-Flash-Next-shared-IQ3K-IQ4NL-FR65k.gguf'),'--spec-type','draft-mtp','--spec-draft-n-max',str(cfg.draft_nmax),'-ubd',str(cfg.draft_ubatch),'-ctkd',cfg.ctkd,'-ctvd',cfg.ctvd]
 return argv

def run(out, mode, threads, batch, ubatch, arm, mtp_reuse=False, options=None):
 cfg=options
 out.mkdir(parents=True,exist_ok=False)
 s=socket.socket(); s.settimeout(.2)
 if s.connect_ex(('127.0.0.1',PORT))==0: raise RuntimeError('port already occupied')
 s.close()
 competitors=games()
 if competitors: raise RuntimeError('competing processes: '+repr(competitors))
 pre=obs.sample(None,'nvme0n1'); mem=pre['meminfo_kib']
 if mem.get('MemAvailable',0)<2*1024**2 or mem.get('SwapFree',0)<512*1024 or not pre['gpu_vram_total'] or not pre['gpu_vram_used'] or int(pre['gpu_vram_total'])-int(pre['gpu_vram_used'])<VRAM_MIN_MIB*1024**2: raise RuntimeError('preflight RAM, swap or VRAM guard')
 env=child_environment(os.environ,cfg.moe_ring_prefill,cfg.moe_reserve_mib)
 root=subprocess.check_output([str(P/'rocm-7.14.1-venv/bin/rocm-sdk'),'path','--root'],text=True).strip()
 env.update(ROCM_PATH=root,HIP_PATH=root,LD_LIBRARY_PATH=root+'/lib:'+root+'/lib/llvm/lib')
 hip={'baseline':{},'sdma':{'HSA_ENABLE_SDMA':'0'},'queues':{'GPU_MAX_HW_QUEUES':'2','HSA_ENABLE_SDMA':'1'},'both':{'HSA_ENABLE_SDMA':'0','GPU_MAX_HW_QUEUES':'2'}}[arm]; env.update(hip)
 model=D/'models/Qwen3.8-Flash-Next-Avalon-IQ3KS/Qwen3.8-Flash-Next-Avalon-IQ3KS-00001-of-00033.gguf'
 argv=build_server_argv(model,threads,batch,ubatch,mtp_reuse,cfg)
 server_argv=argv[:]
 if mode!='real': argv=[sys.executable,'-u',str(Path(__file__).with_name('avalon-fake-oneshot.py')),str(PORT),mode]
 command=argv
 (out/'command.json').write_text(json.dumps({'argv':command,'server_argv':server_argv,'ROCM_PATH':root,'LD_LIBRARY_PATH':env['LD_LIBRARY_PATH'],'moe_cache_reserve_env':env.get('GGML_CUDA_MOE_CACHE_RESERVE_MB','unset'),'moe_ring_prefill_env':env.get('GGML_CUDA_MOE_RING_PREFILL','unset'),'HIP_variables':hip,'mode':mode,'prompt_tokens':cfg.prompt_tokens,'performance_predict':cfg.predict,'temperature':0,'chat_template_kwargs':{'enable_thinking':False},'stream':True,'ignore_eos':True,'prefill_timeout_s':None,'decode_timeout_s':None,'generation_socket_timeout_s':None,'suite_timeout_s':None,'vram_guard_mib':VRAM_MIN_MIB,'process_ram_soft_limit_bytes':None,'process_swap_soft_limit_bytes':2*1024**3,'debugger':None},indent=2)+'\n')
 log=(out/'engine.log').open('x'); samples=(out/'samples.jsonl').open('x'); kernel=(out/'kernel.log').open('x')
 child=watch=child_debugger=None; cg=None; reader=None; reason=''; start=time.monotonic(); state={'phase':'load','stop':False,'done':False,'synthetic':mode!='real','mtp_enabled':mtp_reuse,'flash_attn':cfg.flash_attn,'batch':batch,'ubatch':ubatch,'ctk':cfg.ctk,'ctv':cfg.ctv,'ctkd':cfg.ctkd,'ctvd':cfg.ctvd,'prompt_tokens':cfg.prompt_tokens,'predict':cfg.predict}; tail=''; pressure=extreme=elevated=swap_pressure=None; oom_base=None; swaps=deque(); last_progress=start; last_size=0; progress_request=None; progress_pos=0; progress_pending=b''; stall_key=None; stall_number=0; snapshots=0; stack_started=False; episode=None; printed_progress=0
 print(f"Carga del servidor aislado; resultados: {out}"+(' [SIMULADO]' if mode!='real' else ''),flush=True)
 try:
  if mode=='fault': watch=subprocess.Popen([sys.executable,'-u','-c','import sys,time; time.sleep(2); sys.stdout.write("page allocation "); sys.stdout.flush(); time.sleep(.2); sys.stdout.write("failure\\n"); sys.stdout.flush(); time.sleep(100)'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
  else:
   cursor=subprocess.check_output(['journalctl','-k','-n','0','--show-cursor','--no-pager'],text=True).strip().split('-- cursor: ')[-1]
   if not cursor or '\n' in cursor: raise RuntimeError('no kernel cursor')
   watch=subprocess.Popen(['journalctl','-k','-f','--after-cursor',cursor,'-o','cat'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
  time.sleep(.4)
  if watch.poll() is not None: raise RuntimeError('kernel watcher failed')
  child_started_ns=time.monotonic_ns()
  child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=model.parent,start_new_session=True)
  cg=obs.cgroups(child.pid)
  if not cg or os.getpgid(child.pid)!=child.pid: raise RuntimeError('owned process group not established')
  (out/'cgroups.json').write_text(json.dumps(cg,indent=2)+'\n')
  finite=[(c['path'],k,(c.get(k) or '').strip()) for c in cg for k in ('memory.max','memory.swap.max') if (c.get(k) or '').strip() not in ('','max')]
  if finite: raise RuntimeError('finite inherited memory or swap limit: '+repr(finite))
  reader=threading.Thread(target=request,args=(state,out,True),daemon=True); reader.start()
  engine_fault_pos=0; engine_fault_tail=''
  while child.poll() is None:
   if mode=='observer-fail': raise RuntimeError('synthetic collector failure')
   sample=obs.sample(child.pid,'nvme0n1'); mem=sample['meminfo_kib']; proc=sample['process']; now=time.monotonic()
   if not sample['diskstats'] or not sample['psi_memory'] or not sample['psi_io'] or not proc['status'] or not proc['cgroups'] or not sample['vmstat_pages']: raise RuntimeError('collector missing mandatory fields')
   sample['phase']=state['phase']; sample['request_index']=state.get('request_index'); sample['completed']=state.get('completed',0); sample['request_started_ns']=state.get('request_started_ns'); sample['last_token_ns']=state.get('last_token_ns'); sample['prefill_tokens_observed']=state.get('prefill_tokens'); sample['prefill_progress_ns']=state.get('prefill_progress_ns')
   hw=next(iter(Path('/sys/class/drm/card1/device/hwmon').glob('hwmon*')),None)
   if hw:
    sample['gpu_hwmon']={x:obs.read(hw/x) for x in ('temp1_input','temp2_input','power1_average','fan1_input')}
   sample['gpu_sclk']=obs.read('/sys/class/drm/card1/device/pp_dpm_sclk'); sample['gpu_mclk']=obs.read('/sys/class/drm/card1/device/pp_dpm_mclk')
   sample['cpu_freq_khz']=[obs.read(f'/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq') for i in range(os.cpu_count() or 0)]
   samples.write(json.dumps(sample,separators=(',',':'))+'\n'); samples.flush()
   full=float(re.search(r'full avg10=([0-9.]+)',sample['psi_memory']).group(1)); freegpu=int(sample['gpu_vram_total'])-int(sample['gpu_vram_used'])
   if mode=='vram-fail': freegpu=99*1024**2
   if watch.poll() is not None: reason='kernel watcher died'
   if select.select([watch.stdout],[],[],0)[0]:
    data=os.read(watch.stdout.fileno(),65536).decode(errors='replace'); kernel.write(data); kernel.flush()
    if not data: reason='kernel watcher EOF'
    if FAULT.search(tail+data): reason='kernel fault, allocation failure or OOM'
    tail=(tail+data)[-512:]
   with (out/'engine.log').open('rb') as ef:
    ef.seek(engine_fault_pos); engine_data=ef.read().decode(errors='replace'); engine_fault_pos=ef.tell()
   if FAULT.search(engine_fault_tail+engine_data): reason='engine GPU or allocation fault'
   engine_fault_tail=(engine_fault_tail+engine_data)[-512:]
   if mem['MemAvailable']<2*1024**2: reason='less than 2 GiB available RAM'
   if mem['SwapFree']<512*1024: reason='less than 512 MiB swap headroom'
   rss=re.search(r'^VmRSS:\s+(\d+)\s+kB',proc['status'],re.M)
   if not rss: reason='missing process RAM telemetry'
   vmswap=re.search(r'^VmSwap:\s+(\d+)\s+kB',proc['status'],re.M)
   if not vmswap: reason='missing process swap telemetry'
   elif int(vmswap.group(1))*1024>2*1024**3: reason='process swap exceeded 2 GiB'
   swaps.append((now,sample['vmstat_pages']['pswpout']))
   while swaps and now-swaps[0][0]>10: swaps.popleft()
   swap_bytes=(swaps[-1][1]-swaps[0][1])*os.sysconf('SC_PAGE_SIZE')
   if swap_bytes>128*1024**2:
    swap_pressure=swap_pressure or now
    if now-swap_pressure>10: reason='sustained rapid swap activity'
   else: swap_pressure=None
   events=dict(line.split() for line in proc['cgroups'][0]['memory.events.local'].splitlines()); oom=int(events['oom'])+int(events['oom_kill'])
   if oom_base is None: oom_base=oom
   if oom>oom_base: reason='inherited cgroup OOM event'
   if state['phase']=='load':
    if full>80:
     extreme=extreme or now
     if now-extreme>20: reason='extreme full PSI above 80% for 20 seconds'
    else: extreme=None
    if full>40 and (mem['MemAvailable']<8*1024**2 or swap_bytes>128*1024**2):
     elevated=elevated or now
     if now-elevated>20: reason='high PSI with low RAM or rapid swap'
    else: elevated=None
   else:
    if full>10:
     pressure=pressure or now
     if now-pressure>20: reason='sustained inference full PSI above 10%'
    else: pressure=None
   if freegpu<VRAM_MIN_MIB*1024**2: reason='less than 100 MiB free VRAM'
   if games(child.pid) and not reason: reason='competing desktop, game or inference process'
   if state.get('error') and not reason: reason='request failure: '+state['error']
   if not reason and state.get('request_started_ns') and state.get('last_token_ns') is None:
    request_started=state['request_started_ns']
    if progress_request!=request_started:
     progress_request=request_started; progress_pos=state['progress_log_offset']; progress_pending=b''
    with (out/'engine.log').open('rb') as pf:
     pf.seek(progress_pos); chunk=pf.read(); progress_pos=pf.tell()
    parts=(progress_pending+chunk).split(b'\n'); progress_pending=parts.pop()
    if len(progress_pending)>1024*1024: raise RuntimeError('oversized incomplete engine log line')
    for line in parts: observe_prefill(state,line.decode(errors='replace'),time.monotonic_ns())
    seen=state['prefill_tokens']
    if seen>printed_progress:
     printed_progress=seen
     print(f'Prefill observado: {seen}/{cfg.prompt_tokens} tokens',flush=True)
   if state['done'] and not reason: reason='suite complete'
   if reason: break
   time.sleep(.5)
 except Exception as e: reason='supervision failure: '+repr(e)
 finally:
  state['stop']=True
  if child and not stop_owned_group(child): reason='cleanup failure: owned process group not empty'
  stop(watch)
  if reader: reader.join(timeout=3)
  result={'reason':reason or 'process exited','child_exit':child.returncode if child else None,'seconds':round(time.monotonic()-start,2),'child_started_ns':locals().get('child_started_ns'),'health_ns':state.get('health_ns'),'reader_alive':reader.is_alive() if reader else None,'completed_requests':state.get('completed',0)}
  log.close(); samples.close(); kernel.close(); (out/'result.json').write_text(json.dumps(result,indent=2)+'\n'); print_summary(out,result)
 return result

if __name__=='__main__':
 a=argparse.ArgumentParser(description='Avalon: prefill y decode con una petición fresca; supervisión de seguridad activa.')
 a.add_argument('--ctx',type=int,default=262144,help='ventana de contexto (múltiplo de 256; defecto: 262144)')
 a.add_argument('--prompt-tokens',type=int,default=100000,help='tokens de entrada fresca (defecto: 100000)')
 a.add_argument('--predict',type=int,default=128,help='tokens de salida forzados (defecto: 128)')
 a.add_argument('--batch',type=int,default=4096); a.add_argument('--ubatch',type=int,default=4096)
 a.add_argument('--threads',type=int,default=6)
 a.add_argument('--flash-attn',action=argparse.BooleanOptionalAction,default=True,help='Flash Attention (por defecto activo); al desactivarlo V usa f16 salvo elección explícita')
 for flag,default in [('ctk','q8_0'),('ctv',None),('ctkd','q8_0'),('ctvd',None)]:
  a.add_argument('--'+flag,choices=['q4_0','q8_0','f16','bf16'],default=default)
 a.add_argument('--mtp-reuse',dest='mtp_reuse',action='store_true',default=True,help='activar MTP (defecto)')
 a.add_argument('--no-mtp',dest='mtp_reuse',action='store_false')
 a.add_argument('--draft-ubatch',type=int,default=256); a.add_argument('--draft-nmax',type=int,default=2)
 a.add_argument('--jinja',action=argparse.BooleanOptionalAction,default=True)
 a.add_argument('--fit-target',type=int,default=None,help='MiB; si se omite usa el valor por defecto del motor')
 a.add_argument('--cache-ram',type=int,default=None,help='MiB; si se omite usa el valor por defecto del motor')
 a.add_argument('--moe-reserve-mib',type=int,default=None,help='reserva MoE explícita en MiB; por defecto no se pasa')
 a.add_argument('--moe-ring-prefill',type=int,default=None,metavar='MIB',help='0 desactiva el ring de prefill; valor >=2 fija su tamaño en MiB; omitido usa el valor por defecto del motor')
 a.add_argument('--offline',action='store_true'); a.add_argument('--no-warmup',action='store_true')
 a.add_argument('--arm',choices=['queues'],default='queues'); a.add_argument('--out',help='directorio nuevo; por defecto se genera automáticamente')
 a.add_argument('--mode',choices=['real','fake','fake-delay','fault','observer-fail','vram-fail'],default='real',help=argparse.SUPPRESS)
 x=a.parse_args()
 if x.ctv is None: x.ctv='q4_0' if x.flash_attn else 'f16'
 if x.ctvd is None: x.ctvd='q8_0' if x.flash_attn else 'f16'
 if not x.flash_attn and (x.ctv in ('q4_0','q8_0') or (x.mtp_reuse and x.ctvd in ('q4_0','q8_0'))): a.error('sin Flash Attention, V target y V draft activo deben ser f16 o bf16')
 if x.ctx<256 or x.ctx%256: a.error('context must be a positive multiple of 256')
 if x.prompt_tokens<256 or x.predict<1 or x.prompt_tokens+x.predict>x.ctx: a.error('prompt tokens and output tokens must fit in context; prompt minimum 256')
 if min(x.batch,x.ubatch,x.threads,x.draft_ubatch,x.draft_nmax)<1 or x.ubatch>x.batch: a.error('batch, ubatch, threads and draft values must be positive; ubatch must not exceed batch')
 if x.moe_ring_prefill is not None and (x.moe_ring_prefill<0 or x.moe_ring_prefill==1): a.error('moe-ring-prefill must be 0 or at least 2 MiB')
 if x.fit_target is not None and x.fit_target<0 or x.cache_ram is not None and x.cache_ram< -1 or x.moe_reserve_mib is not None and x.moe_reserve_mib<0: a.error('invalid memory option')
 CTX=x.ctx
 print(f"Flash Attention: {'on' if x.flash_attn else 'off'} | KV target: {x.ctk}/{x.ctv}"+(f" | KV draft: {x.ctkd}/{x.ctvd}" if x.mtp_reuse else ''),flush=True)
 output=Path(x.out).expanduser() if x.out else Path(__file__).parent/('oneshot-'+str(time.time_ns()))
 r=run(output,x.mode,x.threads,x.batch,x.ubatch,x.arm,x.mtp_reuse,x)
 expected={'fake':'suite complete','fake-delay':'suite complete','fault':'kernel fault, allocation failure or OOM','observer-fail':'supervision failure:','vram-fail':'less than 100 MiB free VRAM'}.get(x.mode,'suite complete')
 sys.exit(0 if r['reason'].startswith(expected) and not r['reader_alive'] and r['child_exit'] is not None else 1)
