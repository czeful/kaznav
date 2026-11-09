"""Demo videos for the 2026-10-31 call: a spoken command reaches the robot (Qolda-AVL and the Kazakh NeMo ASR).

    bash tools/wsl.sh python reports/make_demo_2026-10-31.py

Each video: a title card with the speaker's own recording as sound (what was said, what the ASR heard, what Qolda
understood), then the robot's episode video already recorded in the dev run (deterministic runs: no re-run needed),
holding its last frame. The recording is made louder for listening only (peak to -1 dBFS): the models heard it
as in their runs. Writes reports/demo_2026-10-31/videos/*.mp4 (git-ignored).
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "reports/demo_2026-10-31/videos"
FF = imageio_ffmpeg.get_ffmpeg_exe()
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
M = REPO / "runs/matrix"
S01 = REPO / "data/speech/S01"

CLIPS = [
    dict(name="1_avl_vacuum_chair",
         video=M / "dev_speech_d085_vacuum/avl_kk_cascade_lvl/seed0/video_6imZUJGRUq4_7404.mp4",
         wav=S01 / "chair-a01-01__kk.wav", speed=2.0,
         title="Голосовая команда → робот-пылесос (Qolda-AVL)",
         said="«Орындықты тауып берші»", gloss="(найди мне стул)",
         steps=["1. Распознавание речи — Qolda-AVL:  «Орындық тауып берші.»",
                "2. Разбор — Qolda 4B:  цель «орындық» → chair",
                "    гипотезы для поиска: black chair · folding chair · chair in a living room",
                "3. Робот-пылесос, камера на высоте 0.15 м: ищет и проверяет кандидатов (Qolda 4B)"],
         foot="dev-сцена HM3D (train), эпизод 6imZUJGRUq4:7404 · итог: успех · навигация ускорена ×2"),
    dict(name="2_avl_e2e_agent_bed",
         video=M / "dev_agent_speech_s01/avl_kk_e2e/seed0/video_vLpv2VX547B_19520.mp4",
         wav=S01 / "bed-a02-01__kk.wav", speed=1.0,
         title="Голосовая команда → робот (Qolda-AVL слышит и сразу понимает)",
         said="«Төсек қайда? Шаршап тұрмын»", gloss="(где кровать? я устал)",
         steps=["1. Qolda-AVL слышит запись и сама разбирает команду, без отдельной расшифровки:",
                "    услышано «Төсек қайда шаршап отырмын.» → цель «төсек» → bed",
                "    гипотезы для поиска: bed · bed in a room · bed with a person",
                "2. Стандартный агент Habitat, камера на высоте 0.88 м (проверка — Qolda 4B)"],
         foot="dev-сцена HM3D (train), эпизод vLpv2VX547B:19520 · итог: успех"),
    dict(name="3_nemo_vacuum_plant",
         video=M / "dev_speech_d086_vacuum/kkasr_kk_cascade/seed0/video_FRQ75PjD278_31695.mp4",
         wav=S01 / "plant-a02-01__kk.wav", speed=1.0,
         title="Голосовая команда → робот-пылесос (казахская модель NeMo)",
         said="«Осы үйде өсімдік бар ма? Болса, соған апаршы»", gloss="(в доме есть растение? если есть, отвези к нему)",
         steps=["1. Распознавание речи — казахская модель NeMo (FastConformer-RNNT):",
                "    «Осы үйде өсімдік бар ма болса, соған апаршы.»",
                "2. Разбор — Qolda 4B:  цель «өсімдік» → plant;  гипотезы: houseplant · indoor plant · plant in a room",
                "3. Робот-пылесос, камера на высоте 0.15 м: ищет и проверяет кандидатов (Qolda 4B)"],
         foot="dev-сцена HM3D (train), эпизод FRQ75PjD278:31695 · итог: успех"),
    dict(name="4_nemo_agent_chair",
         video=M / "dev_speech_d086_agent/kkasr_kk_cascade/seed0/video_vLpv2VX547B_3094.mp4",
         wav=S01 / "chair-a02-01__kk.wav", speed=1.0,
         title="Голосовая команда → робот (казахская модель NeMo)",
         said="«Орындық жетіспей тұр, тауып берші»", gloss="(не хватает стула, найди)",
         steps=["1. Распознавание речи — казахская модель NeMo:  «Орындық жетіспей тұр, тауып берші.»",
                "2. Разбор — Qolda 4B:  цель «орындық» → chair",
                "    гипотезы: missing chair · empty chair space · chair not present in room",
                "3. Стандартный агент Habitat, камера на высоте 0.88 м (проверка — Qolda 4B)"],
         foot="dev-сцена HM3D (train), эпизод vLpv2VX547B:3094 · итог: успех"),
]


def duration(path: Path) -> float:
    """Length of a media file in seconds, from ffmpeg's banner."""
    err = subprocess.run([FF, "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    h, m, s = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err).groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def size(path: Path) -> tuple:
    """Width and height of a video."""
    err = subprocess.run([FF, "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    w, h = re.search(r"Video: .*?, (\d{3,5})x(\d{3,5})", err).groups()
    return int(w), int(h)


def peak_db(path: Path) -> float:
    """Peak level of a recording, dBFS."""
    err = subprocess.run([FF, "-hide_banner", "-i", str(path), "-af", "volumedetect", "-vn", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    return float(re.search(r"max_volume: (-?[\d.]+) dB", err).group(1))


def card(c: dict, w: int, h: int, out: Path) -> None:
    """The title card: what was said, what was heard and understood, which robot."""
    im = Image.new("RGB", (w, h), (24, 24, 28))
    d = ImageDraw.Draw(im)
    title, big, txt, small = (ImageFont.truetype(BOLD, 28), ImageFont.truetype(BOLD, 30),
                              ImageFont.truetype(FONT, 19), ImageFont.truetype(FONT, 15))
    y = 22
    d.text((28, y), c["title"], font=title, fill=(255, 200, 90))
    y += 52
    d.text((28, y), "Диктор S01, живой голос, по-казахски (звучит запись):", font=txt, fill=(200, 200, 200))
    y += 32
    d.text((28, y), c["said"], font=big, fill=(255, 255, 255))
    y += 42
    d.text((28, y), c["gloss"], font=txt, fill=(170, 170, 170))
    y += 44
    for s in c["steps"]:
        d.text((28, y), s, font=txt, fill=(225, 235, 255))
        y += 30
    d.text((28, h - 30), c["foot"], font=small, fill=(150, 150, 150))
    im.save(out)


def build(c: dict) -> Path:
    """Title card with the recording, then the episode video (sped up if asked), last frame held for 2 s."""
    w, h = size(c["video"])
    png = OUT / f"{c['name']}_card.png"
    card(c, w, h, png)
    t_card = max(5.0, duration(c["wav"]) + 2.5)
    t_nav = duration(c["video"]) / c["speed"] + 2.0
    out = OUT / f"{c['name']}.mp4"
    fc = (f"[0:v]fps=25,format=yuv420p,setsar=1[v0];"
          f"[1:a]volume={-1.0 - peak_db(c['wav']):.1f}dB,aresample=44100,aformat=channel_layouts=mono,adelay=700,apad,"
          f"atrim=0:{t_card:.2f}[a0];"
          f"[2:v]setpts=PTS/{c['speed']},fps=25,tpad=stop_mode=clone:stop_duration=2,scale={w}:{h},format=yuv420p,"
          f"setsar=1[v1];"
          f"[3:a]atrim=0:{t_nav:.2f}[a1];"
          f"[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]")
    cmd = [FF, "-hide_banner", "-loglevel", "error", "-y",
           "-loop", "1", "-framerate", "25", "-t", f"{t_card:.2f}", "-i", str(png),
           "-i", str(c["wav"]), "-i", str(c["video"]),
           "-f", "lavfi", "-t", f"{t_nav:.2f}", "-i", "anullsrc=r=44100:cl=mono",
           "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
           "-c:v", "libx264", "-crf", "20", "-preset", "medium", "-c:a", "aac", "-b:a", "128k",
           "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)
    print(f"{out.name}: card {t_card:.1f} s + navigation {t_nav:.1f} s", flush=True)
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for c in CLIPS:
        build(c)
    print("DEMO_DONE")
