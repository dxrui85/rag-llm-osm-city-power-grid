import os
from pathlib import Path

from PIL import Image

# Portable release overrides. The historical definitions above are retained as provenance.
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("RAG_OSM_DATA_ROOT", REPOSITORY_ROOT.parent / "public_data_release_v1.0_candidate"))
SOURCE = DATA_ROOT / "data" / "figure_source_data" / "supplementary_figures" / "Figure_S3" / "Figure_S3_reference_render.png"
OUTPUT = Path(os.environ.get("RAG_OSM_FIGURE_OUTPUT", REPOSITORY_ROOT / "outputs" / "Figure_S3")) / "Figure_S3.png"


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(SOURCE) as image:
        # The original graphic contains a prose note in the bottom margin.
        # Retain all panels, axes and colour bars, and move the prose to the
        # Word caption in the integrated Supplementary Information.
        cropped = image.crop((0, 0, image.width, 1540))
        cropped.save(OUTPUT, dpi=image.info.get("dpi", (300, 300)))


if __name__ == "__main__":
    main()
