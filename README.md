# Who should we contact? Customer analysis, experiment analysis and uplift targeting

**Status: work in progress. No results yet.** The main result and the main figure will be added after the decision notebook (Notebook 6) is finished.

## Summary for readers without a technical background

A retailer can send a promotion to only part of its customers, and contacting people costs money. This project uses a randomized pilot run by a large grocery retailer to estimate which customers actually change their behavior because of the contact, and compares that selection rule with simpler ones such as contacting everyone or contacting the customers most likely to buy anyway. Results will be reported as extra purchases and extra profit for each share of the budget, with the uncertainty shown and with cost and margin left as adjustable assumptions.

## Business question

Given a limited contact budget, which customers should be contacted so that the campaign creates the most additional purchases, and how much better is that than the simple alternatives?

## Main result

Pending. It will appear here with a confidence interval, the assumptions behind it and one main figure (or a short demo recording).

## Data

The X5 RetailHero uplift dataset was shared by X5 Retail Group (a large Russian food retailer) for the RetailHero hackathon in winter 2019. The figures below come from the scikit-uplift documentation and are re-checked against the files by `scripts/03_inspect_data.py`.

* `clients.csv`: general information about 400,162 customers.
* `purchases.csv`: 45,786,568 rows of purchase history before the communication.
* `uplift_train.csv` (called `train.csv` in parts of the documentation): 200,039 customers with a treatment flag (`treatment_flg`) and a binary outcome (`target`). The documented treatment share is 0.5.

The documentation describes `target` as whether a purchase happened after the communication. It does not state the time window or what counts as a purchase. This is checked and documented in Notebook 1.

Raw data is not stored in this repository. The terms of use and redistribution rules have not been verified yet, so only derived and aggregated outputs are published.

## Method (planned)

1. Data audit and randomization checks (group sizes, sample ratio mismatch, balance of pre contact features).
2. Customer base analysis from pre contact history only: recency, frequency and monetary features, segments, concentration of value.
3. Prediction of purchase probability and spend, with calibration.
4. Experiment analysis: average effect with confidence intervals, variance reduction, A/A tests, power analysis, semi synthetic effect injection, subgroup analysis with multiple testing correction.
5. Uplift modeling (the estimated change in purchase probability caused by contact), evaluated with Qini and AUUC curves, bootstrap intervals and several seeds.
6. Decision layer: offline policy evaluation on the randomized data, comparing contacting everyone, contacting at random, contacting likely buyers and contacting customers with high estimated uplift. Contact cost and margin per purchase are parameters, with a sensitivity analysis.
7. Ablation and error analysis by feature group and customer segment.
8. A small Streamlit demo that reads precomputed results from step 6.

Causal statements are made only inside the randomized experiment. Findings from purchase history alone are descriptive.

## Known limitations

* One company, one country, one season (2019), one campaign. The results are not claimed to generalize to other sectors or countries.
* The exact meaning of `target` is to be confirmed (see above).
* Only one communication moment exists, so effects over time and repeated experiments cannot be studied. The timestamp structure of the purchase history decides whether lifetime value and churn can be defined at all.
* The data has no currency and no margin. Cost and margin are parameters, so no absolute profit claim is made.
* Individual customers cannot be labeled as persuadable, sure thing, lost cause or sleeping dog. Such labels would only be estimated tendencies.

## How to run

```
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python -m pytest -q

python scripts/01_download_data.py      # about 650 MB download into data/raw
python scripts/02_convert_to_parquet.py # CSV to Parquet with DuckDB
python scripts/03_inspect_data.py       # audit, writes reports/data_audit.json
```

On systems with `make`, `make all` runs the three data steps. The purchases table is never loaded into pandas as a whole. DuckDB and Parquet are used so that it is processed out of core.

## Reproducibility

The random seed lives in `src/x5uplift/config.py`. Dependency versions are frozen into `requirements.lock` with `make freeze` after the first successful install.

## Repository layout

```
src/x5uplift/   shared code (config, helpers)
scripts/        data download, conversion and audit
notebooks/      analysis notebooks (added step by step)
reports/        aggregated audit outputs
figures/        exported figures
tests/          unit tests for helper functions
data/           local only, not tracked
```
