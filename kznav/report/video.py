"""Frame composition and MP4 writing for trajectory videos (demo now, episode videos in stage 6).

A frame is a horizontal strip of panels (camera RGB, depth, top-down map) with a text band on
top. Everything is plain numpy / OpenCV so it runs headless.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

WHITE = (255, 255, 255)


def _cv2():
    import cv2  # opencv ships with habitat_env; imported lazily so kznav stays importable without it

    return cv2


def depth_to_rgb(depth: np.ndarray, max_m: float) -> np.ndarray:
    """Metric depth -> colour image (near = warm). Invalid / zero depth is black."""
    cv2 = _cv2()
    d = np.clip(np.nan_to_num(depth, nan=0.0) / max_m, 0.0, 1.0)
    img = cv2.applyColorMap((255 * (1.0 - d)).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1]
    img[depth <= 0] = 0
    return np.ascontiguousarray(img)


def resize_to_height(img: np.ndarray, height: int) -> np.ndarray:
    cv2 = _cv2()
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, round(w * height / h)), height), interpolation=cv2.INTER_AREA)


@dataclass
class TopDownMap:
    """Navigable-area image with helpers to draw paths and the robot in map coordinates.

    ``grid`` comes from ``PathFinder.get_topdown_view(mpp, height)``: rows follow Habitat +Z,
    columns +X, starting at ``lower`` (the navmesh bounds' lower corner).
    """

    grid: np.ndarray
    lower_x: float
    lower_z: float
    mpp: float

    def to_px(self, x: float, y_map: float) -> Tuple[int, int]:
        return int(round((x - self.lower_x) / self.mpp)), int(round((-y_map - self.lower_z) / self.mpp))

    def render(
        self,
        *,
        path: Optional[Sequence[Tuple[float, float]]] = None,
        trajectory: Optional[Sequence[Tuple[float, float]]] = None,
        pose: Optional[Tuple[float, float, float]] = None,
        robot_radius_m: float = 0.17,
        goal: Optional[Tuple[float, float]] = None,
        collisions: Sequence[Tuple[float, float]] = (),
    ) -> np.ndarray:
        cv2 = _cv2()
        img = np.where(self.grid[..., None], np.uint8(235), np.uint8(90)).repeat(3, axis=2)
        if path is not None and len(path) > 1:
            pts = np.array([self.to_px(*p) for p in path], dtype=np.int32)
            cv2.polylines(img, [pts], False, (60, 110, 230), 1, cv2.LINE_AA)
        if trajectory is not None and len(trajectory) > 1:
            pts = np.array([self.to_px(*p) for p in trajectory], dtype=np.int32)
            cv2.polylines(img, [pts], False, (220, 40, 40), 2, cv2.LINE_AA)
        for c in collisions:
            cv2.drawMarker(img, self.to_px(*c), (255, 140, 0), cv2.MARKER_TILTED_CROSS, 8, 2)
        if goal is not None:
            cv2.drawMarker(img, self.to_px(*goal), (20, 160, 20), cv2.MARKER_STAR, 14, 2)
        if pose is not None:
            x, y, yaw = pose
            c = self.to_px(x, y)
            r = max(2, int(round(robot_radius_m / self.mpp)))
            cv2.circle(img, c, r, (30, 30, 30), 2, cv2.LINE_AA)
            tip = self.to_px(x + 2 * robot_radius_m * np.cos(yaw), y + 2 * robot_radius_m * np.sin(yaw))
            cv2.line(img, c, tip, (30, 30, 30), 2, cv2.LINE_AA)
        return img


def compose(panels: Sequence[np.ndarray], height: int, text_lines: Sequence[str] = (), band_px: int = 26) -> np.ndarray:
    """Panels side by side at a common height, with a text band above. Output dims are even (H.264)."""
    cv2 = _cv2()
    strip = np.hstack([resize_to_height(p, height) for p in panels])
    band = np.full((band_px * max(1, len(text_lines)) + 6, strip.shape[1], 3), 20, dtype=np.uint8)
    for i, line in enumerate(text_lines):
        cv2.putText(band, line, (8, band_px * (i + 1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
    frame = np.vstack([band, strip])
    h, w = frame.shape[:2]
    return np.ascontiguousarray(frame[: h - h % 2, : w - w % 2])


class Mp4Writer:
    """H.264 MP4 via imageio-ffmpeg (plays in the default Windows player)."""

    def __init__(self, path: Path, fps: float):
        import imageio

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = Path(path)
        self._w = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=7,
                                     macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"])
        self.n_frames = 0
        self._shape: Optional[Tuple[int, ...]] = None

    def write(self, frame: np.ndarray) -> None:
        if self._shape is None:
            self._shape = frame.shape
        elif frame.shape != self._shape:
            frame = _cv2().resize(frame, (self._shape[1], self._shape[0]))
        self._w.append_data(frame)
        self.n_frames += 1

    def close(self) -> None:
        self._w.close()
