# Code for RAG-LLM OpenStreetMap topology approximation

This repository candidate contains the custom Python and PowerShell code used for descriptive evaluation, deterministic topology assembly, sensitivity analysis, and figure generation.

## Layout

- `src/analysis/`: role assembly, post hoc reference joining, descriptive scoring, and scale comparison.
- `src/figures/main/`: main-figure scripts and editable Figure 1 generator.
- `src/figures/supplementary/`: supplementary-figure scripts.
- `tests/`: frozen synthetic regression tests for the main and sensitivity analyses.
- `tools/`: release verification utilities.

## Data dependency

Download the companion data record and place its extracted folder next to this code folder:

```text
parent/
  public_code_release_v1.0/
  public_data_release_v1.0/
```

The folder names may differ. Pass the extracted data folder explicitly with
`--data-root`, as shown below.

## Offline reproduction

Run the complete offline workflow from the code folder:

```text
python reproduce_all.py --data-root ../public_data_release_v1.0 --output-dir ../reproduced_outputs
```

This command runs 40 synthetic regression tests, regenerates the descriptive
results at the 0.125 degree primary scale and 0.10 degree sensitivity scale,
compares key tables with the frozen results, and redraws the figures supported
by the public data record. It does not access the network, an API, or an API
key. The output directory must not already exist.

On Windows, use a short working path (for example `D:\\rag_osm_repro`) to avoid
the legacy path-length limit while graph files are being written. See
`REPRODUCING.md` for separate result and figure commands and expected outputs.

Figure 1 is supplied as an editable PowerPoint source. Figure 4 and Figure S4
are not regenerated because the third-party administrative boundary remains
excluded pending redistribution-rights confirmation. Their underlying public
topology and spatial-block source tables remain included.

## Environment

The frozen environment used Python 3.13.9 with pandas 2.3.3, NumPy 2.3.5, Matplotlib 3.10.6 and Pillow 12.0.0. Create a clean environment and install `requirements.txt`.

## Licence

The original custom code in this release candidate is licensed under the MIT License. Third-party libraries remain under their respective licences. See `LICENSE` and `THIRD_PARTY_NOTICES.md`.
