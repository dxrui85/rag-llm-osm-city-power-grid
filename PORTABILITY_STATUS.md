# Portability status

## Ready for repository review

- The analysis and figure source files are present.
- Frozen synthetic tests are included.
- The Python dependency versions are recorded.
- No API key or credential file is intentionally included.

## Completed in this candidate

1. Main Figures 2, 3 and 5 use repository-relative data paths and passed local regeneration tests.
2. Supplementary Figures S1, S2, S3 and S5-S8 passed local regeneration tests.
3. The primary and sensitivity analysis modules passed 40 synthetic regression tests.
4. Machine-specific paths were removed from executable figure entry points and the sensitivity-analysis dependency binding.
5. `reproduce_all.py`, `reproduce_results.py` and `reproduce_figures.py` provide explicit offline entry points.
6. A clean-directory run regenerated both descriptive analyses and 10 supported figures without network or API access.
7. Four key result files at each spatial scale were byte-identical to their frozen counterparts.

## Deliberate exclusions

1. Figure 4 and Figure S4 redraws remain disabled because the administrative boundary is not distributed pending rights confirmation.
2. Figure 1 is distributed as an editable PowerPoint source; automated rendering depends on desktop presentation software and is not part of the cross-platform Python workflow.
3. The public repository DOI and final repository names remain to be assigned at publication-deposit time.

## Platform note

Windows users should extract both records to a short path. A first audit run
under the deeply nested project path reached the legacy Windows path-length
limit while writing graph files; the same frozen workflow passed from
the short audit root `D:\rag_osm_repro`.

This folder remains a **release candidate** until repository metadata and the
final public deposit are approved by the authors.
