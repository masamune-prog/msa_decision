"""Independent ipSAE implementation (E0 gate).

Recomputes ipSAE_min / ipSAE_max directly from an AlphaFold 3 confidences.json
and compares against the values tabulated in the Zenodo deposit.

Formula per Dunbrack, bioRxiv 2025.02.10.637595 (reference implementation
DunbrackLab/IPSAE @ 6174cf9e71cb1bd660cc805856a18c4871a6dec3):

    ptm(x, d0)  = 1 / (1 + (x/d0)^2)
    d0(L)       = max(1.0, 1.24 * (max(L,26) - 15)^(1/3) - 1.8)

For an ordered chain pair (c1 -> c2), with PAE cutoff tau = 10:
    valid[i,j]  = (chain[i]==c1) & (chain[j]==c2) & (pae[i,j] < tau)
    n0res[i]    = number of valid j for row i
    ipsae[i]    = mean_j{valid} ptm(pae[i,j], d0(n0res[i]))
    asym(c1,c2) = max_i ipsae[i]

ipSAE_min = min(asym(A,B), asym(B,A)); ipSAE_max = max(...).

Only the PAE matrix and chain assignment are needed -- no coordinates, so the
model.cif is not required for this score.
"""
import json
import numpy as np

PAE_CUTOFF = 10.0


def calc_d0_array(L):
    L = np.maximum(np.asarray(L, dtype=float), 26.0)
    return np.maximum(1.0, 1.24 * (L - 15.0) ** (1.0 / 3.0) - 1.8)


def ipsae_from_confidences(path, pae_cutoff=PAE_CUTOFF):
    """Return (ipsae_min, ipsae_max, per_direction dict)."""
    with open(path) as fh:
        d = json.load(fh)
    pae = np.asarray(d['pae'], dtype=float)
    chains = np.asarray(d['token_chain_ids'])

    uniq = sorted(set(chains.tolist()))
    asym = {}
    for c1 in uniq:
        for c2 in uniq:
            if c1 == c2:
                continue
            valid = np.outer(chains == c1, chains == c2) & (pae < pae_cutoff)
            n0res = valid.sum(axis=1)
            d0res = calc_d0_array(n0res)
            # ptm transform of each row under that row's own d0
            ptm = 1.0 / (1.0 + (pae / d0res[:, None]) ** 2.0)
            rows = np.where(chains == c1)[0]
            best = 0.0
            for i in rows:
                v = valid[i]
                if v.any():
                    best = max(best, float(ptm[i, v].mean()))
            asym[(c1, c2)] = best
    vals = list(asym.values())
    return min(vals), max(vals), asym


if __name__ == '__main__':
    import sys
    for p in sys.argv[1:]:
        lo, hi, asym = ipsae_from_confidences(p)
        print(f'{p}\n  ipSAE_min={lo:.4f}  ipSAE_max={hi:.4f}  '
              f'directions={ {f"{a}->{b}": round(v,4) for (a,b),v in asym.items()} }')
