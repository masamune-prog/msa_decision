# Does the target MSA help AlphaFold 3 rank de novo binders?

An ablation study. For a benchmark of designed mini-binders with experimentally
measured binding, we re-fold every design with AlphaFold 3 under controlled
alignment conditions and ask whether the target's multiple sequence alignment
improves the *ranking* of true binders — not the accuracy of any single
structure.

The headline comparison is **target MSA on vs. off**, replicated across three
model seeds, and crossed with two other factors the literature conflates with
it: whether the target is **cropped to the interface**, and whether a
**self-template** supplies the structural prior that an MSA would otherwise
carry.

## What is in this repository

Code and derived results. The folds themselves are not here: the full work
directory is ~345 GB of AlphaFold 3 outputs.

```
scripts/     building AF3 inputs and running the folds
analysis/    scoring (ipSAE), compiling folds into one table, statistics
figures/     the report's figures and the scripts that make them
results/     the derived tables -- enough to redo the analysis without folding
ids/         the design subsets each experiment was run over
targets/     one AF3 input JSON per unique target sequence (for the MSA step)
msa_paths.py path resolution shared by the scripts
```

`results/all_af3_folds.csv` (15 MB, 32,477 rows) is the central artefact: one
row per fold, with ipSAE, ipTM, pTM, ranking score, alignment depth and the
binding label. **Every figure and every statistic in the report can be
regenerated from this file alone**, with no GPU and no fold data — see
"Reproducing the analysis" below.

## Prerequisites

### The input dataset

`final_dataset.csv` (82 MB; 3,676 rows, 312 columns, 16 targets, 394 measured
binders) is **not redistributed here** — it is too large to ship and it is not
ours. Download it from the authors' Zenodo deposit:

> Overath, M. D., Rygaard, A., & Jenkins, T. P. (2025). *Dataset for:
> Predicting Experimental Success in De Novo Binder Design: A Meta-Analysis of
> 3,766 Experimentally Characterised Binders* (v1.0.0) [Data set]. Technical
> University of Denmark. Zenodo. https://doi.org/10.5281/zenodo.15722219

```bash
cd "$MSA_DECISION_ROOT"
curl -L -O https://zenodo.org/records/15722219/files/final_dataset.csv
```

The deposit is CC BY 4.0, so **cite it in anything you publish from this
pipeline**. It also carries the original AF2 / AF3 / Boltz-1 / ColabFold
prediction outputs (9.3 GB in total); none of those are needed here, since this
study re-folds everything from scratch under controlled alignment conditions.

The columns this pipeline reads are: `binder_id`, `target_id`, `binder` (the
label), `A_seq` (binder), `B_seq` / `target_subchain_<chain>_seq` (target
chains), and `msa_info` (which target chains get an alignment).

### Installing AlphaFold 3

Folding uses the official implementation, unmodified:

```bash
git clone https://github.com/google-deepmind/alphafold3.git
```

Follow its own installation instructions, and obtain the model parameters from
DeepMind under their terms (the weights are **not** redistributable, and are
not in this repository). You also need the genetic databases for the MSA step
(`fetch_databases.sh` in that repo).

The scripts expect the clone at `$MSA_DECISION_ROOT/alphafold3`, its virtualenv
at `alphafold3/.venv`, its databases at `alphafold3/msa`, and the model
parameters in `$MSA_DECISION_ROOT` (AF3 is invoked with `--model_dir=$ROOT`).

### Hardware

These results were produced on 3× NVIDIA GPUs over roughly two weeks of
wall-clock time. The MSA step is CPU- and I/O-bound (jackhmmer over ~50 GB of
sequence databases); the folding is GPU-bound.

### The work directory

Everything the pipeline produces goes in one large directory, named by an
environment variable that every script reads:

```bash
export MSA_DECISION_ROOT=/path/to/a/big/disk/msa_decision
```

Budget ~350 GB. Put `final_dataset.csv` and the AF3 model parameters there.

## Running the experiments

### 1. Parent alignments, once per target

A target's homologues do not depend on which binder is attached, so the
expensive genetic search runs once per distinct target sequence (25 of them)
rather than once per design.

```bash
python scripts/prepare_targets.py        # -> $ROOT/targets/*.json + manifest.tsv
bash   scripts/run_msa_queue.sh          # -> $ROOT/msa_out/<key>/<key>_data.json
python scripts/compute_neff.py           # -> $ROOT/neff80.tsv
```

`compute_neff.py` measures effective alignment depth (Neff80 — the sum of
inverse cluster sizes at 80% identity) for every target. This is the quantity
downstream claims about "deep" and "shallow" alignments are made against, so
the convention is fixed in that file's docstring and must not drift.

`targets/` in this repo already holds the generated input JSONs and manifest,
so you can skip `prepare_targets.py` if you only want the MSA step.

### 2. E2 — the main arm: target MSA on vs. off

Both conditions over all 3,532 single-chain designs. The binder chain is
*always* single-sequence (it is novel by construction and has no homologues);
only the target's alignment is ablated. Templates are disabled throughout.

```bash
python scripts/build_inputs.py --ids_file ids/e2_ids.txt --condition full \
       --out_dir $MSA_DECISION_ROOT/e2_in_full
python scripts/build_inputs.py --ids_file ids/e2_ids.txt --condition none \
       --out_dir $MSA_DECISION_ROOT/e2_in_none
bash   scripts/run_e2.sh                 # 7,064 folds over 3 GPUs
python analysis/score_e2.py              # -> $ROOT/e2_scores.tsv
```

`run_e2.sh` shards work round-robin over GPUs and is **resumable**: a fold whose
output already contains a `*_confidences.json` is skipped, so re-running after
an interruption only does what is left. Shards hold symlinks, not copies — the
`full` inputs are 24 GB because each one embeds its target's alignment.

### 3. Seed replication

The same two arms at seeds 2 and 3, so that any effect can be separated from
seed noise. Fold names carry the seed, which is what makes this resumable.

```bash
for s in 2 3; do for c in full none; do
  python scripts/build_inputs.py --ids_file ids/e2_ids.txt --condition $c \
         --seed $s --out_dir $MSA_DECISION_ROOT/e2s${s}_in_${c}
done; done
bash scripts/run_e2_seeds.sh             # 14,128 folds
```

### 4. Excision — cropping the target to the interface

For each design, the target residues within an 11 Å halo of any binder atom are
found **in our own full-target model**, the single contiguous window spanning
them is taken, and both the target sequence and its alignment columns are
cropped to that window. Deriving the crop from our own folds keeps it
self-consistent rather than importing someone else's interface definition.

```bash
for c in full none; do
  python scripts/build_excised.py --ids_file ids/ex_ids.txt --condition $c \
         --out_dir $MSA_DECISION_ROOT/ex_in_$c      # 414 designs, 4 receptors
  python scripts/build_excised.py --ids_file ids/ex2_ids.txt --condition $c \
         --out_dir $MSA_DECISION_ROOT/ex2_in_$c     # the remaining 9 receptors
done
bash scripts/run_ex2.sh                  # GPUS="0 1 2" to use more than one
```

### 5. Self-template

Full target, ± MSA, with the target's own structure (extracted from our
full+MSA model) supplied as a template. A template and an alignment are partly
redundant sources of the same information, so this arm tests whether the MSA
effect survives when a structural prior is present — E2 had templates disabled
entirely.

```bash
for c in full none; do
  python scripts/build_template.py --ids_file ids/ex_ids.txt --condition $c \
         --out_dir $MSA_DECISION_ROOT/tm_in_$c
done
```

### 6. Depth × policy ladder

On excised targets, alignment subsets at controlled effective depth
(Neff80 = 2, 5, 10, 30) under two selection policies: `rand` (shuffle, then the
shortest prefix reaching the target depth) and `div` (greedy most-dissimilar
ordering, then the same). The depth-matched random arm is not optional —
without it, any claim about *which* homologues matter is uninterpretable.

```bash
python scripts/build_ladder.py --src_dir $MSA_DECISION_ROOT/ex_in_full \
       --out_dir $MSA_DECISION_ROOT/e3_in
```

The endpoints (depth 1 and full) already exist as the excised arms, so only the
intermediate rungs are generated.

### 7. Masked alignment

Intact full-length target, alignment restricted to the interface window
(columns outside it gapped out, rows empty in the window dropped). Comparing
this against `full` isolates cropping the *alignment*; comparing it against
`exfull` isolates cropping the *target*.

```bash
python scripts/build_masked.py $MSA_DECISION_ROOT/mk_in
```

### 8. Opportunistic scheduling

For long tails on a shared machine, `sched_gpu.py` runs pending folds on one
GPU *only while that GPU is otherwise idle*, claiming work under a file lock so
several can cooperate, and dropping corrupt outputs so they are retried.

```bash
bash scripts/launch_after_build.sh       # one scheduler per GPU, groups in priority order
```

## Reproducing the analysis

This needs no GPU and no fold data — the shipped tables in `results/` are
enough. `msa_paths.rpath()` prefers your own copy under `$MSA_DECISION_ROOT`
and falls back to `results/`, so with no work directory at all:

```bash
pip install -r requirements.txt
python figures/fig_excision.py            # intact vs excised, ± MSA
python figures/fig_excision_all13.py      # the same across all 13 receptors
python figures/fig_seed_effect.py         # per-seed effect, paired bootstrap
```

`figures/make_figures.py` regenerates figures 1–4, but figures 3 and 4 re-score
fold directories, so those two need the full work directory.

The statistics live in `analysis/analyze_e2.py`: per-target average precision
and ROC-AUC for each arm, the paired per-target delta with a bootstrap CI
(**target is the unit of replication**, not design), the sign distribution of
per-target effects, and a decomposition of whether a change comes from moving
the positive or the negative class. It has a self-test that validates the
machinery on synthetic data with no folding required:

```bash
python analysis/analyze_e2.py --selftest
```

### Rebuilding the fold table from scratch

If you have run the folds yourself:

```bash
python analysis/compile_all_folds.py     # all experiments -> all_af3_folds.csv
python analysis/add_features.py          # + binder/crop length, MSA depth
python analysis/add_receptor.py          # + receptor sequence
python analysis/append_ex2.py            # + the ex2 excision batch
```

## Scoring

Interface quality is scored with **ipSAE** (Dunbrack, bioRxiv 2025.02.10.637595),
reimplemented independently in `analysis/ipsae_recompute.py` and validated
against the reference implementation before any of this was run. It needs only
the PAE matrix and chain assignment, so no coordinates are required.

`ipsae_min` is the conservative direction (the smaller of the two asymmetric
chain-pair values) and is the primary metric throughout; `ipsae_max` and `iptm`
are carried alongside for comparison.

## What is in `results/`

| file | rows | what it is |
|---|---|---|
| `all_af3_folds.csv` | 32,477 | every fold, all experiments, long format |
| `e2_scores.tsv` | 3,532 | the main arm, one row per design, both conditions |
| `e2_seeds_scores.tsv` | — | the same at seeds 2 and 3 |
| `neff80.tsv` | 25 | effective alignment depth per target sequence |

Fold counts by arm, as compiled in `all_af3_folds.csv`:

| experiment | conditions | seeds | folds |
|---|---|---|---|
| `e2_out` | full, none | 1 | 7,064 |
| `e2_seeds_out` | full, none | 2, 3 | 14,128 |
| `ex_out` / `ex2_out` | exfull, exnone | 1 | 828 / 6,236 |
| `tm_out` | tmfull, tmnone | 1 | 828 |
| `e3_out` | rand/div × depth 2,5,10,30 | 1 | 3,312 |
| `e1_out` / `e1_out_s2` | full | 1, 2 | 40 / 40 |

Two further arms — the masked-alignment arm (`mk_out`) and the ladder extended
to all receptors (`e3b_out`) — were still folding when this table was compiled
and are **not** included in it. Re-run `compile_all_folds.py` after they finish
to pick them up.

## Caveats

- The crop in the excision arm is derived from our own models, so it is
  self-consistent but not identical to any published interface definition.
- `e1_out` is a 40-design pilot used to validate the pipeline against the
  deposit before scaling up; it is not an experimental arm.
- AlphaFold 3 is not bit-wise deterministic across hardware. Expect small
  numerical differences; this is exactly why the main arm is replicated across
  three seeds and why the unit of replication is the target, not the design.

## License

The code here is ours. Two dependencies carry their own terms:

- **AlphaFold 3** and its model parameters — see the AlphaFold 3 repository;
  the weights are obtained from DeepMind under their terms and are not
  redistributable.
- **The benchmark dataset** — CC BY 4.0, Overath et al. 2025,
  [10.5281/zenodo.15722219](https://doi.org/10.5281/zenodo.15722219).
  Attribution is required.
