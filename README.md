# GB-Persona paper reproducibility package

This repository contains only the code and data needed to reproduce the experimental results reported in **“GB-Persona: Budget-Constrained Granular-Ball Selection of LLM-Based Agents for Social Simulation.”** It covers the 128-persona WorldValuesBench experiment and its nested first-64-persona pool-size sensitivity experiment. Development sweeps, unreported methods, Nemotron experiments, research plans, approval records, and API credentials are intentionally excluded.

## Paper protocol

The frozen 128-persona pool has uniform population weights. The nested experiment uses the first 64 personas in the same frozen order and renormalizes their weights to `1/64`. Calibration and test sets contain 20 disjoint ordered-response questions each, with matched family counts. Every persona-question pair has eight independently sampled responses from `google/gemma-4-26b-a4b-it` through an OpenAI-compatible OpenRouter endpoint, using temperature 0.7, top-p 1.0, and 32 output tokens.

Calibration distributions use Jeffreys smoothing with `alpha=0.5`. GB-Persona uses mean normalized ordinal Wasserstein-1 distance, `lambda=0.25`, tail quantile `0.8`, and minimum child size `2`. The requested budgets are `{4, 8, 16, 32}` for the 128-persona pool and `{2, 4, 8, 16}` for the 64-persona pool. At every point, each baseline is evaluated with the actual number of representatives returned by GB-Persona. Randomized methods use seeds 7–26 and are averaged over 20 runs; deterministic methods run once.

The seven reported baselines are K-means, Weighted Random, Demographic Stratified, Density Peaks, DBSCAN, Spectral Clustering, and Behavior K-medoids. K-means uses the positive-eigenvalue classical-MDS embedding of the calibration distance, weighted Lloyd updates, and ten initializations. Demographic Stratified uses country/region strata. Every method returns existing representatives with aggregation weights summing to one; methods without native weights transfer each unselected persona's mass to its nearest representative in calibration distance.

## Reported results

Lower mean test normalized ordinal-W1 error and lower normalized AUEC are better.

### 128-persona pool

| Method | B=4 | B=8 | B=16 | B=32 | AUEC |
|---|---:|---:|---:|---:|---:|
| GB-Persona | 0.034424 | 0.027909 | 0.016002 | 0.012739 | 0.018937 |
| K-means | 0.034575 | 0.027495 | 0.021835 | 0.017397 | 0.022690 |
| Behavior K-medoids | 0.046303 | 0.029827 | 0.020978 | 0.019204 | 0.024177 |
| Spectral Clustering | 0.046140 | 0.034707 | 0.020320 | 0.021879 | 0.025693 |
| Density Peaks | 0.057507 | 0.050821 | 0.056513 | 0.054225 | 0.054711 |
| DBSCAN | 0.060226 | 0.055727 | 0.049556 | 0.049994 | 0.051766 |
| Demographic Stratified | 0.044127 | 0.032314 | 0.018086 | 0.015664 | 0.022303 |
| Weighted Random | 0.056628 | 0.036250 | 0.021302 | 0.018245 | 0.026155 |

GB-Persona's actual representative counts are `4, 8, 16, 21`. Requested budget 32 is therefore a 21-representative comparison, because no legal positive-gain split remains. Its AUEC is 15.1% below Demographic Stratified, the strongest baseline by AUEC.

### Nested 64-persona pool

| Method | B=2 | B=4 | B=8 | B=16 | AUEC |
|---|---:|---:|---:|---:|---:|
| GB-Persona | 0.073480 | 0.034232 | 0.028139 | 0.012255 | 0.028145 |
| K-means | 0.070619 | 0.046416 | 0.025773 | 0.024490 | 0.033033 |
| Behavior K-medoids | 0.052301 | 0.044983 | 0.035676 | 0.021346 | 0.034764 |
| Spectral Clustering | 0.053999 | 0.048835 | 0.037230 | 0.019486 | 0.035845 |
| Density Peaks | 0.048406 | 0.061194 | 0.058263 | 0.045670 | 0.054589 |
| DBSCAN | 0.060604 | 0.049596 | 0.044458 | 0.043288 | 0.046378 |
| Demographic Stratified | 0.060603 | 0.039869 | 0.031816 | 0.020003 | 0.032223 |
| Weighted Random | 0.076813 | 0.043559 | 0.032066 | 0.019295 | 0.034076 |

GB-Persona reaches all four requested budgets and has the lowest AUEC, 12.7% below Demographic Stratified. The advantage is curve-level rather than pointwise: Density Peaks is better at `B=2`, and K-means is better at `B=8`. AUECs should not be compared across the two pool sizes because their target distributions and integration intervals differ.

Full-precision values are in `results/paper_results.csv`.

## Reproduce the results

Python 3.10–3.12 is supported. The dependency versions are pinned to the environment used for the final paper figures and tables.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
gb-persona-reproduce
```

The command recomputes both experiments from the released response labels, verifies all 64 method-budget points and AUECs against the full-precision paper table, and writes selections and computed metrics to the ignored `reproduced_results/` directory. It makes no API calls.

Run the independent checks with:

```bash
python -m unittest discover -s tests -v
```

Regenerate the four paper figures with:

```bash
python scripts/plot_paper_figures.py
```

## Included files

- `config.json`: the model, sampling, GB-Persona, budget, baseline, and seed settings reported in the paper.
- `data/personas.csv`: the ordered 128-persona pool with the six prompt fields and uniform weights. Original WVS interview identifiers are replaced by `P001`–`P128`.
- `data/questions.json`, `data/question_splits.json`, and `data/question_families.json`: exactly the 20 calibration and 20 test questions used in the reported experiments, including the matched five-family allocation; pilot questions are excluded.
- `data/responses.csv.gz`: exactly 40,960 valid parsed response labels, one row per persona-question-repetition slot. It contains no prompts, survey answers, API keys, or unused responses.
- `data/collection_audit.json`: the paper-aligned collection counts, including 40,987 attempts for 40,960 slots and 27 recovered test-side transport errors.
- `src/gb_persona/`: only the probability construction, ordinal geometry, eight reported selectors, weighting, evaluation, and exact-result verification code.
- `scripts/`: the result and figure reproduction entry points.
- `results/paper_results.csv`: the full-precision values behind the paper's curves, AUECs, and rounded text.
- `tests/test_reproduction.py`: scope, integrity, and full numerical reproduction checks.

## Data provenance and scope

The persona specifications and ordered questions originate from WorldValuesBench, which is derived from World Values Survey Wave 7. This release contains no human survey-answer columns and no original `D_INTERVIEW` identifiers. The response labels are model-generated outputs, not human answers and not claims about a representative sample of real individuals.

Please cite the paper together with WorldValuesBench:

> Wenlong Zhao, Debanjan Mondal, Niket Tandon, Danica Dillion, Kurt Gray, and Yuling Gu. “WorldValuesBench: A Large-Scale Benchmark Dataset for Multi-Cultural Value Awareness of Language Models.” LREC-COLING 2024, pp. 17696–17706.
