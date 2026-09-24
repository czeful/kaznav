"""Frame composition and MP4 writing for trajectory videos (demo now, episode videos in stage 6).

A frame is a horizontal strip of panels (camera RGB, depth, top-down map) with a text band on
top. Everything is plain numpy / OpenCV so it runs headless.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

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


_FONT_CANDIDATES = ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf")
_fonts: dict = {}


def _font(size: int):
    """A TrueType font with Cyrillic incl. the Kazakh letters (OpenCV's Hershey fonts are ASCII only)."""
    if size not in _fonts:
        from PIL import ImageFont

        _fonts[size] = None
        for f in _FONT_CANDIDATES:
            if Path(f).exists():
                _fonts[size] = ImageFont.truetype(f, size)
                break
    return _fonts[size]


def draw_text(img: np.ndarray, text: str, xy: Tuple[int, int], color=WHITE, size: int = 16) -> np.ndarray:
    """Draw one line with its top-left at ``xy``; Unicode-safe (falls back to ASCII-only OpenCV)."""
    font = _font(size)
    if font is None:
        cv2 = _cv2()
        cv2.putText(img, text, (xy[0], xy[1] + size), cv2.FONT_HERSHEY_SIMPLEX, size / 26, color, 1, cv2.LINE_AA)
        return img
    from PIL import Image, ImageDraw

    pil = Image.fromarray(img)
    ImageDraw.Draw(pil).text(xy, text, font=font, fill=tuple(int(c) for c in color))
    img[...] = np.asarray(pil)
    return img


def compose(panels: Sequence[np.ndarray], height: int, text_lines: Sequence[str] = (), band_px: int = 26) -> np.ndarray:
    """Panels side by side at a common height, with a text band above. Output dims are even (H.264).

    A text line may be a ``(text, (r, g, b))`` pair to colour it."""
    strip = np.hstack([resize_to_height(p, height) for p in panels])
    band = np.full((band_px * max(1, len(text_lines)) + 6, strip.shape[1], 3), 20, dtype=np.uint8)
    for i, line in enumerate(text_lines):
        text, color = line if isinstance(line, tuple) else (line, WHITE)
        draw_text(band, text, (8, band_px * i + 5), color, size=band_px - 9)
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


def render_policy_map(omap: Any, *, pose: Tuple[float, float, float], radius_m: float,
                      trajectory: Sequence[Tuple[float, float]] = (), path: Sequence[Tuple[float, float]] = (),
                      frontier_cells: Sequence[Tuple[int, int]] = (), candidates: Sequence[Any] = (),
                      gt_boxes: Sequence[Tuple[Sequence[float], Sequence[float]]] = (), margin_m: float = 1.0,
                      px_per_cell: int = 2) -> np.ndarray:
    """The robot's own occupancy map (north up), cropped to the observed area.

    unknown = dark grey, free = light, occupied = black, frontier = cyan, planned path = blue,
    trajectory = red, candidates = magenta, ground-truth goal boxes (debug only) = green outline.
    """
    cv2 = _cv2()
    obs = omap.observed
    rr, cc = np.nonzero(obs)
    r_pose, c_pose = (int(v[0]) for v in omap.to_cell(np.array([pose[0]]), np.array([pose[1]])))
    m = int(np.ceil(margin_m / omap.res))
    r0 = max(0, min(rr.min() if len(rr) else r_pose, r_pose) - m)
    r1 = min(omap.n, max(rr.max() if len(rr) else r_pose, r_pose) + m + 1)
    c0 = max(0, min(cc.min() if len(cc) else c_pose, c_pose) - m)
    c1 = min(omap.n, max(cc.max() if len(cc) else c_pose, c_pose) + m + 1)
    img = np.full((r1 - r0, c1 - c0, 3), 70, dtype=np.uint8)
    img[omap.free[r0:r1, c0:c1]] = (225, 225, 225)
    img[omap.occupied[r0:r1, c0:c1]] = (20, 20, 20)
    for r, c in frontier_cells:
        if r0 <= r < r1 and c0 <= c < c1:
            img[r - r0, c - c0] = (0, 200, 220)
    img = img[::-1].copy()                                  # north up
    img = cv2.resize(img, (img.shape[1] * px_per_cell, img.shape[0] * px_per_cell), interpolation=cv2.INTER_NEAREST)
    H = img.shape[0]

    def px(x: float, y: float) -> Tuple[int, int]:
        r, c = omap.to_cell(np.array([x]), np.array([y]))
        return int((c[0] - c0 + 0.5) * px_per_cell), int(H - (r[0] - r0 + 0.5) * px_per_cell)

    for lo, hi in gt_boxes:
        a, b = px(lo[0], -lo[2]), px(hi[0], -hi[2])
        cv2.rectangle(img, (min(a[0], b[0]), min(a[1], b[1])), (max(a[0], b[0]), max(a[1], b[1])), (40, 180, 40), 1)
    if len(path) > 1:
        cv2.polylines(img, [np.array([px(*p) for p in path], dtype=np.int32)], False, (60, 110, 230), 1, cv2.LINE_AA)
    if len(trajectory) > 1:
        cv2.polylines(img, [np.array([px(*p) for p in trajectory], dtype=np.int32)], False, (220, 40, 40), 1, cv2.LINE_AA)
    for c in candidates:
        cv2.circle(img, px(float(c.xy[0]), float(c.xy[1])), 5, (220, 0, 220), 2, cv2.LINE_AA)
    cp = px(pose[0], pose[1])
    rad = max(2, int(round(radius_m / omap.res * px_per_cell)))
    cv2.circle(img, cp, rad, (230, 120, 0), 2, cv2.LINE_AA)
    tip = px(pose[0] + 2 * radius_m * np.cos(pose[2]), pose[1] + 2 * radius_m * np.sin(pose[2]))
    cv2.line(img, cp, tip, (230, 120, 0), 2, cv2.LINE_AA)
    return img


def render_value_map(omap: Any, value01: np.ndarray, pose: Tuple[float, float, float], radius_m: float,
                     margin_m: float = 1.0, px_per_cell: int = 2) -> np.ndarray:
    """Value map (normalised 0..1, turbo colours) over the observed area, north up, with the robot."""
    cv2 = _cv2()
    rr, cc = np.nonzero(omap.observed)
    r_pose, c_pose = (int(v[0]) for v in omap.to_cell(np.array([pose[0]]), np.array([pose[1]])))
    m = int(np.ceil(margin_m / omap.res))
    r0 = max(0, min(rr.min() if len(rr) else r_pose, r_pose) - m)
    r1 = min(omap.n, max(rr.max() if len(rr) else r_pose, r_pose) + m + 1)
    c0 = max(0, min(cc.min() if len(cc) else c_pose, c_pose) - m)
    c1 = min(omap.n, max(cc.max() if len(cc) else c_pose, c_pose) + m + 1)
    v = value01[r0:r1, c0:c1]
    img = cv2.applyColorMap((255 * v).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1].copy()
    img[~omap.observed[r0:r1, c0:c1]] = 40
    img[omap.occupied[r0:r1, c0:c1]] = 0
    img = img[::-1].copy()
    img = cv2.resize(img, (img.shape[1] * px_per_cell, img.shape[0] * px_per_cell), interpolation=cv2.INTER_NEAREST)
    H = img.shape[0]
    r, c = omap.to_cell(np.array([pose[0]]), np.array([pose[1]]))
    cp = (int((c[0] - c0 + 0.5) * px_per_cell), int(H - (r[0] - r0 + 0.5) * px_per_cell))
    cv2.circle(img, cp, max(2, int(round(radius_m / omap.res * px_per_cell))), (255, 255, 255), 2, cv2.LINE_AA)
    return img
