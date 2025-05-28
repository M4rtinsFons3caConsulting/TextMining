from pathlib import Path

# ----------- PROJECT TREE -------------- #
# Stores path logic for the project, anchoring at ROOT.

ROOT = Path().resolve().parent

DATA_DIR = ROOT / "data"
TRAIN_DATA = DATA_DIR / "train.csv"
TEST_DATA = DATA_DIR / "test.csv"

LABELS = {
    'Bearish': 0
    ,'Bullish': 1
    ,'Neutral': 2
}