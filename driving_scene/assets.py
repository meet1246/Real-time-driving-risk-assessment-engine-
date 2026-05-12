"""
assets.py — optional transparent-PNG icon loading for the scene renderer.

This module is OPTIONAL. The scene renderer first asks `get_vehicle_icon()`
for an asset; if it returns None, the renderer falls back to its procedural
drawing path. That means:

  * The project runs and looks polished with **zero** assets shipped.
  * If a user drops free / self-made transparent PNGs into
    `assets/vehicles/`, `assets/people/`, `assets/bikes/`, the renderer
    will pick them up automatically.

No copyrighted assets ship with the project.

PUBLIC API
==========
    load_icon(name) -> np.ndarray | None
        Generic loader. Tries assets/{name} relative to project root.

    get_vehicle_icon(class_name, color_idx) -> np.ndarray | None
        Tries assets/vehicles/{class}_{color_label}.png in this order:
            car_white.png, car_silver.png, car_blue.png, ...
        Falls back to None if nothing matches.

    get_person_icon() -> np.ndarray | None
    get_bike_icon(class_name) -> np.ndarray | None

The loaded PNGs are cached on first use so subsequent frames are free.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from .config import ASSET_DIR, USE_ICON_ASSETS, VEHICLE_COLOR_PALETTE


_CACHE: Dict[str, Optional[np.ndarray]] = {}


# Maps a config palette index to the filename suffix users would natural
# create (assets/vehicles/car_white.png, etc).
_COLOR_NAMES = ("white", "silver", "blue", "green", "cream", "lavender")


def safe_load_png(path: Path) -> Optional[np.ndarray]:
    """Read a 4-channel PNG with alpha. Returns None if anything goes wrong."""
    try:
        if not path.exists():
            return None
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        # Force 4-channel BGRA so the renderer can blend with alpha.
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        return img
    except Exception:
        return None


def load_icon(rel_path: str) -> Optional[np.ndarray]:
    """Load an icon by path relative to ASSET_DIR; cached."""
    if not USE_ICON_ASSETS:
        return None
    if rel_path in _CACHE:
        return _CACHE[rel_path]
    img = safe_load_png(ASSET_DIR / rel_path)
    _CACHE[rel_path] = img
    return img


def get_vehicle_icon(class_name: str, color_idx: int) -> Optional[np.ndarray]:
    """Return assets/vehicles/{class}_{color}.png if it exists, else None."""
    cls = (class_name or "car").lower()
    if cls in ("motorcycle", "motorbike"):
        return load_icon(f"bikes/{cls}.png")
    if cls in ("bicycle", "bike"):
        return load_icon("bikes/bicycle.png")
    color = _COLOR_NAMES[color_idx % len(_COLOR_NAMES)]
    return load_icon(f"vehicles/{cls}_{color}.png")


def get_person_icon() -> Optional[np.ndarray]:
    return load_icon("people/person.png")


def stable_color_index(track_id: Optional[int]) -> int:
    if track_id is None:
        return 0
    return int(track_id) % len(VEHICLE_COLOR_PALETTE)


def stable_vehicle_color(track_id: Optional[int]):
    """Pick a deterministic body color for a track from VEHICLE_COLOR_PALETTE.

    Same track_id -> same color for the whole lifetime of the track. Risk is
    NOT encoded here; that's the renderer's glow + primary-ring + HUD job.
    """
    return VEHICLE_COLOR_PALETTE[stable_color_index(track_id)]


def reset_cache() -> None:
    _CACHE.clear()
