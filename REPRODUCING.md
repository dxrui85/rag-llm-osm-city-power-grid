# Offline reproduction guide

## Scope

The supplied entry points reproduce the descriptive object-level results,
deterministic graph assembly outputs, spatial-scale comparison, and all figures
that do not depend on the excluded administrative boundary. They never call a
model API, read an API key, or access the network.

## Recommended folder layout

Use a short path on Windows:

```text
D:/rag_osm_repro/
  code/
  data/
```

Install the recorded dependencies in a clean Python environment:

```text
python -m pip install -r requirements.txt
```

## Complete workflow

From the code folder:

```text
python reproduce_all.py --data-root ../data --output-dir ../output
```

The output folder must not exist before the command starts. The workflow:

1. runs 40 synthetic regression tests;
2. reproduces the 0.125 degree primary-scale descriptive analysis;
3. reproduces the 0.10 degree sensitivity analysis and scale comparison;
4. compares four key files per scale with the frozen result record;
5. redraws Main Figures 2, 3 and 5 and Supplementary Figures S1, S2, S3 and S5-S8;
6. writes machine-readable JSON audit reports.

## Separate commands

Results only:

```text
python reproduce_results.py --data-root ../data --output-dir ../results
```

Figures only:

```text
python reproduce_figures.py --data-root ../data --output-dir ../figures
```

Tests only:

```text
python -m unittest discover -s tests -p "test_*.py" -v
```

## Expected checks

- All 40 synthetic tests pass.
- Both analysis completion audits report zero API and network calls.
- The comparison report marks the four selected result files at each scale as
  both byte-identical and content-identical to the frozen records.
- Ten figure scripts return exit code 0.
- Figure 1, Figure 4 and Figure S4 are listed as not automatically regenerated.

Figure 1 is supplied as an editable PowerPoint source. Figure 4 and Figure S4
require an administrative boundary that is intentionally excluded until its
redistribution rights are confirmed.
