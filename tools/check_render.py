"""Habitat render + physics smoke check. Prints the GL renderer so the manifest can record it.

    bash tools/wsl.sh python tools/check_render.py [--scene PATH] [--out render.png]

Uses gpu_device_id = -1 (no CUDA/EGL device matching), which is required under WSL2
(docs/decisions.md D-008). With the variables set by tools/wsl.sh, Mesa renders on the NVIDIA GPU
through D3D12; the renderer line in the log shows which adapter was used.
"""
from __future__ import annotations

import argparse
import glob
import os
import time


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=None, help="GLB scene; default: habitat-sim test asset simple_room.glb")
    ap.add_argument("--out", default="render_check.png")
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--gpu-device-id", type=int, default=-1)
    args = ap.parse_args()

    import habitat_sim
    import numpy as np
    from PIL import Image

    scene = args.scene
    if scene is None:
        cands = glob.glob(os.path.expanduser("~/habitat-sim/data/test_assets/scenes/simple_room.glb"))
        if not cands:
            raise SystemExit("no --scene given and ~/habitat-sim/data/test_assets/scenes/simple_room.glb not found")
        scene = cands[0]

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene
    sim_cfg.enable_physics = True
    sim_cfg.gpu_device_id = args.gpu_device_id
    agent = habitat_sim.agent.AgentConfiguration()
    specs = []
    for uuid, st in (("rgb", habitat_sim.SensorType.COLOR), ("depth", habitat_sim.SensorType.DEPTH)):
        s = habitat_sim.CameraSensorSpec()
        s.uuid, s.sensor_type, s.resolution, s.position, s.hfov = uuid, st, [480, 640], [0.0, 0.15, 0.0], 69.0
        specs.append(s)
    agent.sensor_specifications = specs
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent]))
    try:
        t0 = time.perf_counter()
        for _ in range(args.frames):
            obs = sim.get_sensor_observations()
        ms = (time.perf_counter() - t0) / args.frames * 1000
        t0 = time.perf_counter()
        sim.step_physics(1.0)
        phys_ms = (time.perf_counter() - t0) * 1000
        rgb, depth = obs["rgb"][..., :3], obs["depth"]
        Image.fromarray(rgb).save(args.out)
        print(f"habitat_sim {habitat_sim.__version__} bullet={habitat_sim.built_with_bullet}")
        print(f"scene {scene}")
        print(f"render {ms:.1f} ms/frame (RGB+D 640x480); physics 1 s sim = {phys_ms:.1f} ms wall")
        print(f"rgb mean {float(rgb.mean()):.3f}; depth range [{float(depth.min()):.4f}, {float(depth.max()):.4f}]")
        print(f"saved {args.out}  (the 'Renderer:' line above names the GPU adapter)")
        if float(np.asarray(rgb).std()) == 0.0:
            raise SystemExit("rendered image is constant — rendering is broken")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
