"""Concatenate episode videos into one reel with a title card before each clip.

    bash tools/wsl.sh python tools/make_demo_reel.py --out runs/demo/reel.mp4 \
        --title "kznav: Kazakh commands -> Qolda -> robot" \
        runs/debug/<a>/video_x.mp4="Банан қайда? Тауып бер. | success" runs/debug/<b>/video_y.mp4="..."

Each clip argument is ``path=caption``; ``|`` in a caption starts a new line. Frames are padded to a
common size. A clip longer than ``--max-clip-seconds`` is played faster (every k-th frame; the last
frames are always kept, and the card says "xk"). The reel only re-encodes existing episode videos: it
adds no data of its own.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kznav.report.video import Mp4Writer, draw_text  # noqa: E402

FPS = 4.0


def read_frames(path: Path):
    import imageio

    r = imageio.get_reader(str(path))
    try:
        for f in r:
            yield np.asarray(f)[..., :3]
    finally:
        r.close()


def card(h: int, w: int, lines, size: int = 30) -> np.ndarray:
    img = np.full((h, w, 3), 18, dtype=np.uint8)
    y0 = h // 2 - len(lines) * (size + 14) // 2
    for i, line in enumerate(lines):
        draw_text(img, line, (40, y0 + i * (size + 14)), (240, 240, 240) if i else (255, 200, 80), size=size)
    return img


def pad(f: np.ndarray, h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[: f.shape[0], : f.shape[1]] = f[:h, :w]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+", help="path=caption")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--card-seconds", type=float, default=3.0)
    ap.add_argument("--hold-last-seconds", type=float, default=2.0)
    ap.add_argument("--max-clip-seconds", type=float, default=40.0)
    args = ap.parse_args()
    clips = [(Path(c.split("=", 1)[0]), c.split("=", 1)[1] if "=" in c else "") for c in args.clips]
    sizes = []
    for p, _ in clips:
        f = next(read_frames(p))
        sizes.append(f.shape[:2])
    h = max(s[0] for s in sizes) + max(s[0] for s in sizes) % 2
    w = max(s[1] for s in sizes) + max(s[1] for s in sizes) % 2
    n_card, n_hold = int(round(args.card_seconds * FPS)), int(round(args.hold_last_seconds * FPS))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wr = Mp4Writer(out, fps=FPS)
    try:
        if args.title:
            for _ in range(n_card):
                wr.write(card(h, w, args.title.split("|")))
        for i, (p, caption) in enumerate(clips, 1):
            frames = [pad(f, h, w) for f in read_frames(p)]
            k = max(1, int(np.ceil(len(frames) / (args.max_clip_seconds * FPS))))
            tail = int(3 * FPS)             # the last 3 s (the decision that ends the episode) at real speed
            keep = frames[: max(0, len(frames) - tail)][::k] + frames[max(0, len(frames) - tail):]
            speed = f"   (видео ускорено x{k}, конец — в реальном темпе)" if k > 1 else ""
            for _ in range(n_card):
                wr.write(card(h, w, [f"{i}/{len(clips)}{speed}"] + caption.split("|")))
            last = None
            for f in keep:
                last = f
                wr.write(last)
            for _ in range(n_hold if last is not None else 0):
                wr.write(last)
    finally:
        wr.close()
    print(f"reel: {out}")


if __name__ == "__main__":
    main()
