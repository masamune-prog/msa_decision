"""Compile every AF3 fold (all experiments) into one long-format CSV: all_af3_folds.csv"""
import csv, glob, json, os, sys, re
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipsae_recompute import ipsae_from_confidences
ROOT = os.environ.get('MSA_DECISION_ROOT', os.getcwd())
csv.field_size_limit(10**9)
EXPS = ['e1_out', 'e1_out_s2', 'e1_retest_out', 'e2_out', 'e2_seeds_out', 'e3_out', 'ex_out', 'tm_out']

def parse(exp, name):
    seed = 1
    m = re.search(r'__s(\d+)$', name)
    if m: seed = int(m.group(1)); name = name[:m.start()]
    if exp == 'e1_out_s2': seed = 2
    binder, cond = name.rsplit('__', 1)
    return binder, cond, seed

def one(args):
    exp, path = args
    name = os.path.basename(path)
    if '__' not in name: return None
    binder, cond, seed = parse(exp, name)
    row = dict(experiment=exp, binder_id=binder, condition=cond, seed=seed, fold_dir=os.path.relpath(path, ROOT))
    summ = glob.glob(os.path.join(path, '*_summary_confidences.json'))
    top = [p for p in glob.glob(os.path.join(path, '*_confidences.json')) if 'summary' not in os.path.basename(p)]
    if not summ or not top: return row
    try:
        s = json.load(open(summ[0]))
        for k in ('iptm', 'ptm', 'ranking_score', 'fraction_disordered', 'has_clash'): row[k] = s.get(k)
        cp = s.get('chain_pair_iptm')
        if cp and len(cp) >= 2: row['chain_pair_iptm_01'] = cp[0][1]; row['chain_pair_iptm_10'] = cp[1][0]
        lo, hi, _ = ipsae_from_confidences(top[0])
        row['ipsae_min'], row['ipsae_max'] = lo, hi
    except Exception as e:
        row['error'] = str(e)[:80]
    return row

if __name__ == '__main__':
    jobs = [(e, p) for e in EXPS for p in sorted(glob.glob(f'{ROOT}/{e}/*')) if os.path.isdir(p)]
    print(len(jobs), 'folds', flush=True)
    with Pool(40) as pool: rows = [r for r in pool.imap(one, jobs, chunksize=8) if r]
    meta = {}
    for r in csv.DictReader(open(f'{ROOT}/e2_scores.tsv'), delimiter='\t'): meta[r['binder_id']] = (r['target_id'], r['label'])
    for r in csv.DictReader(open(f'{ROOT}/final_dataset.csv')): meta.setdefault(r['binder_id'], (r['target_id'], ''))
    for r in rows: r['target_id'], r['label'] = meta.get(r['binder_id'], ('', ''))
    cols = ['experiment','binder_id','target_id','label','condition','seed','ipsae_min','ipsae_max','iptm','ptm','ranking_score','chain_pair_iptm_01','chain_pair_iptm_10','fraction_disordered','has_clash','fold_dir','error']
    with open(f'{ROOT}/all_af3_folds.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, cols, extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    print('wrote', len(rows))
