"""Run pending AF3 folds on ONE GPU, only while that GPU is idle (nobody else using it).
usage: python3 sched_gpu.py <gpu>
Groups run in priority order. Chunks of CHUNK folds; GPU re-checked before each chunk so we yield
if someone else starts. Cross-worker claims via flock. Resumable: finished folds are skipped."""
import fcntl, glob, os, shutil, subprocess, sys, time
ROOT = os.environ.get('MSA_DECISION_ROOT', os.getcwd())
AF3 = f'{ROOT}/alphafold3'
GROUPS = [('mk_in', 'mk_out'), ('e3b_in', 'e3b_out')]      # priority order
CHUNK, MEM_MAX, UTIL_MAX = 120, 3000, 10
SAMPLES = int(os.environ.get('WAIT_SAMPLES', 5))
g = sys.argv[1]
os.makedirs(f'{ROOT}/sched/claims', exist_ok=True)
log = lambda m: print(time.strftime('%m-%d %H:%M:%S'), f'[gpu{g}]', m, flush=True)

def gpu_idle():
    o = subprocess.run(['nvidia-smi', '-i', g, '--query-gpu=memory.used,utilization.gpu',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout.split(',')
    return int(o[0]) < MEM_MAX and int(o[1]) < UTIL_MAX

def wait_idle():
    ok = 0
    while ok < SAMPLES:
        ok = ok + 1 if gpu_idle() else 0
        time.sleep(60)

def done(out, name): return bool(glob.glob(f'{out}/{name}/*_confidences.json'))

def claim():
    with open(f'{ROOT}/sched/lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for ind, outd in GROUPS:
            todo = []
            for f in sorted(glob.glob(f'{ROOT}/{ind}/*.json')):
                n = os.path.basename(f)[:-5]
                if done(f'{ROOT}/{outd}', n) or os.path.exists(f'{ROOT}/sched/claims/{outd}__{n}'): continue
                todo.append((f, outd, n))
                if len(todo) == CHUNK: break
            if todo:
                for f, o, n in todo: open(f'{ROOT}/sched/claims/{o}__{n}', 'w').close()
                return todo
    return []

fails = 0
while True:
    if fails >= 3: log('3 consecutive failed chunks; stopping'); break
    todo = claim()
    if not todo: log('no pending work; exiting'); break
    wait_idle()
    shard = f'{ROOT}/sched/shard_gpu{g}'; shutil.rmtree(shard, ignore_errors=True); os.makedirs(shard)
    for f, o, n in todo:
        d = f'{ROOT}/{o}/{n}'
        if os.path.isdir(d) and not done(f'{ROOT}/{o}', n): shutil.rmtree(d)      # half-written leftover
        os.symlink(f, f'{shard}/{n}.json')
    outs = {o for _, o, _ in todo}
    log(f'running {len(todo)} folds -> {sorted(outs)}')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=g)
    for o in outs: os.makedirs(f'{ROOT}/{o}', exist_ok=True)
    # one chunk never mixes groups: claim() draws from a single group
    r = subprocess.run([f'{AF3}/.venv/bin/python', f'{AF3}/run_alphafold.py', f'--input_dir={shard}',
                        f'--output_dir={ROOT}/{sorted(outs)[0]}', f'--model_dir={ROOT}', '--norun_data_pipeline'],
                       env=env, stdout=open(f'{ROOT}/sched/gpu{g}.af3.log', 'a'), stderr=subprocess.STDOUT)
    bad = 0
    for f, o, n in todo:                       # drop empty/corrupt outputs so they are retried
        d = f'{ROOT}/{o}/{n}'
        if os.path.isdir(d) and any(os.path.getsize(p) == 0 for p in glob.glob(f'{d}/*.json')):
            shutil.rmtree(d); bad += 1
        try: os.remove(f'{ROOT}/sched/claims/{o}__{n}')
        except FileNotFoundError: pass
    nd = sum(done(f'{ROOT}/{o}', n) for _, o, n in todo)
    fails = fails + 1 if nd == 0 else 0
    log(f'chunk finished rc={r.returncode}, {nd}/{len(todo)} folds done, removed {bad} corrupt')
