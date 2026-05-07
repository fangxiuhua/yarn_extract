"""Project-wide configuration constants."""

from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PICTURE_DIR = PROJECT_ROOT / "data" / "picture"
OUTPUT_DIR = PROJECT_ROOT / "new_pictures"

# --- Image preprocessing ---
MAX_DIMENSION = 640
GAUSSIAN_KERNEL = (5, 5)
BINARY_THRESHOLD = 127

# --- Default image list ---
DEFAULT_IMAGES = [
    "data/picture/black (171).jpg",
    "data/picture/black-cha (72).jpg",
    "data/picture/black-dian (127).jpg",
    "data/picture/black-tiao (192).jpg",
    "data/picture/Pic_20241016123223968.jpg",
    "data/picture/black-xing (154).jpg",
    "data/picture/purple-ge (167).jpg",
    "data/picture/red (42).jpg",
]
