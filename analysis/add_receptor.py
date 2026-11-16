import csv, glob, json, os
from multiprocessing import Pool
ROOT = os.environ.get('MSA_DECISION_ROOT', os.getcwd())
csv.field_size_limit(10**9)
def work(item):
    k, path = item
    d = json.load(open(glob.glob(f'{ROOT}/{path}/*_data.json')[0]))
    return k, ':'.join(s['protein']['sequence'] for s in d['sequences'] if 'protein' in s and s['protein']['id'] != 'A')
if __name__ == '__main__':
    rows = list(csv.DictReader(open(f'{ROOT}/all_af3_folds.csv')))
    key = lambda r: (r['experiment'], r['binder_id'], r['condition'])
    # tiny no-MSA sibling files give the same sequences; use them when present
    jobs = {}
    for r in rows:
        if r['experiment'] == 'e2_seeds_out': continue
        p = r['fold_dir']
        sib = p.rsplit('__', 1)[0] + '__' + {'full':'none','exfull':'exnone','tmfull':'tmnone'}.get(r['condition'], r['condition'])
        jobs[key(r)] = sib if os.path.isdir(f'{ROOT}/{sib}') and glob.glob(f'{ROOT}/{sib}/*_data.json') else p
    print(len(jobs), flush=True)
    with Pool(40) as p: res = dict(p.imap_unordered(work, jobs.items(), chunksize=8))
    for r in rows:
        k = ('e2_out', r['binder_id'], r['condition']) if r['experiment'] == 'e2_seeds_out' else key(r)
        r['receptor_seq'] = res[k]
    cols = list(rows[0]); 
    if 'receptor_seq' not in cols: cols.insert(cols.index('crop_len')+1, 'receptor_seq')
    bad = sum(len(r['receptor_seq'].replace(':','')) != int(r['crop_len']) for r in rows)
    with open(f'{ROOT}/all_af3_folds.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(rows)
    print('len mismatches', bad)
