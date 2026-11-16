"""Add binder_len, crop_len (target residues actually folded), MSA depth to all_af3_folds.csv.
Seed-2/3 folds reuse the MSA of the matching e2_out fold (same inputs apart from seed)."""
import csv, glob, json, os
from multiprocessing import Pool
ROOT = os.environ.get('MSA_DECISION_ROOT', os.getcwd())

def feat(path):
    d = json.load(open(glob.glob(f'{path}/*_data.json')[0]))
    prots = [s['protein'] for s in d['sequences'] if 'protein' in s]
    binder = [p for p in prots if p['id'] == 'A'] or prots[:1]
    target = [p for p in prots if p not in binder]
    cnt = lambda m: m.count('>') if m else 0
    return dict(binder_len=sum(len(p['sequence']) for p in binder),
                crop_len=sum(len(p['sequence']) for p in target),
                n_target_chains=len(target),
                msa_unpaired_depth=sum(cnt(p.get('unpairedMsa')) for p in target),
                msa_paired_depth=sum(cnt(p.get('pairedMsa')) for p in target),
                n_templates=sum(len(p.get('templates') or []) for p in target))

def work(item):
    k, path = item
    try: return k, feat(os.path.join(ROOT, path))
    except Exception as e: return k, dict(feat_error=str(e)[:60])

if __name__ == '__main__':
    rows = list(csv.DictReader(open(f'{ROOT}/all_af3_folds.csv')))
    key = lambda r: (r['experiment'], r['binder_id'], r['condition'])
    jobs = {}
    for r in rows:
        if r['experiment'] != 'e2_seeds_out': jobs[key(r)] = r['fold_dir']
    print(len(jobs), 'data.json to parse', flush=True)
    with Pool(40) as p: res = dict(p.imap_unordered(work, jobs.items(), chunksize=4))
    new = ['binder_len','crop_len','n_target_chains','msa_unpaired_depth','msa_paired_depth','n_templates']
    for r in rows:
        k = ('e2_out', r['binder_id'], r['condition']) if r['experiment'] == 'e2_seeds_out' else key(r)
        r.update(res.get(k, {}))
    cols = [c for c in rows[0] if c not in ('fold_dir','error')] + new + ['fold_dir','error']
    cols = list(dict.fromkeys(cols))
    with open(f'{ROOT}/all_af3_folds.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, cols, extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    print('done; feat errors:', sum(1 for r in rows if r.get('feat_error')))
