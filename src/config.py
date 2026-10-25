"""Shared paths, constants and plotting style for the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "figures"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MASTER_CSV = PROCESSED_DIR / "funding_master.csv"
MODEL_PATH = MODEL_DIR / "random_forest.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

STAGE_ORDER = ["Pre-Seed", "Seed", "Series A", "Series B", "Series C", "Series D"]

# Colours (validated reference palette: blue is the primary data colour,
# orange is the single highlight colour; greys are used for everything else).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e3e2de"
SURFACE = "#fcfcfb"


def apply_style():
    """Apply one consistent Matplotlib/Seaborn look to every chart."""
    import matplotlib as mpl
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "text.color": INK,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })
