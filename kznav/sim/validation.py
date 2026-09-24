"""Physics plausibility suite for the wheeled robot on a primitive arena.

Every scenario starts the robot at rest with its reference point at the origin, heading
forward (map yaw = pi/2, i.e. Habitat -Z), on a flat static floor whose top is y = 0.
Obstacles are placed ahead along -Z. Results are measured, compared with what the
configuration predicts, and reported — including the failures.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..eval.robot import StuckMonitor, StuckParams, count_collision_events
from .arena import Box, floor, populate
from .env import HabitatEnv, SceneSpec, Tick
from .urdf import RobotGeometry

FORWARD_YAW = math.pi / 2


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class ScenarioResult:
    name: str
    metrics: Dict[str, Any]
    checks: List[Check] = field(default_factory=list)
    wall_s: float = 0.0
    sim_s: float = 0.0

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


def _stuck_params(cfg: Mapping[str, Any]) -> StuckParams:
    s = cfg["eval"]["stuck"]
    return StuckParams(float(s["window_s"]), float(s["min_displacement_m"]), float(s["min_yaw_change_rad"]),
                       float(s["min_cmd_lin"]), float(s["min_cmd_ang"]))


class Suite:
    def __init__(self, cfg: Mapping[str, Any], vcfg: Mapping[str, Any], workdir: Path):
        if cfg["embodiment"]["kind"] != "wheeled":
            raise ValueError("the physics suite validates the wheeled embodiment")
        self.cfg = cfg
        self.v = vcfg
        self.tol = vcfg["tolerances"]
        self.workdir = Path(workdir)
        self.geom = RobotGeometry.from_config(cfg["embodiment"])
        self.limits = cfg["embodiment"]["drive"]["limits"]
        self.control_dt = float(cfg["embodiment"]["physics"]["control_dt_s"])

    # ------------------------------------------------------------------ helpers
    def _env(self, boxes: Sequence[Box], tag: str) -> HabitatEnv:
        spec = SceneSpec(scene_id="NONE", populate=populate([floor(), *boxes]))
        return HabitatEnv(self.cfg, spec, self.workdir / tag, seed=int(self.cfg["run"]["seed"]))

    def _drive(self, env: HabitatEnv, v: float, w: float, seconds: float) -> Tuple[List[Tick], float]:
        n = int(round(seconds / self.control_dt))
        t0 = time.perf_counter()
        ticks = env.drive(v, w, n)
        return ticks, time.perf_counter() - t0

    def _motion_stats(self, env: HabitatEnv, ticks: List[Tick]) -> Dict[str, Any]:
        stuck = StuckMonitor(_stuck_params(self.cfg))
        detected = False
        for tk in ticks:
            detected |= stuck.update(tk.t, tk.pose[0], tk.pose[1], tk.pose[2], tk.v_cmd, tk.w_cmd)
        gap = float(self.cfg["eval"]["collisions"]["merge_gap_s"])
        return {
            "collision_events": count_collision_events([t.collision for t in ticks], [t.t for t in ticks], gap),
            "collision_links": sorted({l for t in ticks for l in t.collision_links}),
            "stuck_detected": bool(detected),
            "stuck_time_s": round(stuck.stuck_time_s, 3),
            "max_tilt_deg": round(max((t.tilt_deg for t in ticks), default=0.0), 2),
        }

    @staticmethod
    def _speeds(ticks: List[Tick], dt: float) -> np.ndarray:
        xy = np.array([[0.0, 0.0]] + [t.pose[:2] for t in ticks])
        return np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt

    # ------------------------------------------------------------------ scenarios
    def straight(self) -> ScenarioResult:
        c = self.v["straight"]
        env = self._env([], "straight")
        try:
            env.reset((0.0, 0.0, 0.0), FORWARD_YAW)
            ticks, wall = self._drive(env, c["v"], 0.0, c["seconds"])
            sp = self._speeds(ticks, self.control_dt)
            steady = float(sp[-20:].mean())
            acc = np.diff(np.concatenate([[0.0], sp])) / self.control_dt
            reach = next((t.t for t, s in zip(ticks, sp) if s >= 0.95 * c["v"]), math.nan)
            x_end, y_end, yaw_end = ticks[-1].pose
            m = {"steady_speed_mps": round(steady, 4), "cmd_speed_mps": c["v"],
                 "time_to_95pct_s": reach, "ideal_time_to_95pct_s": round(0.95 * c["v"] / self.limits["max_lin_acc_mps2"], 3),
                 "peak_accel_mps2": round(float(acc.max()), 3), "accel_limit_mps2": self.limits["max_lin_acc_mps2"],
                 "distance_m": round(y_end, 3), "lateral_drift_m": round(x_end, 4),
                 "heading_drift_deg": round(math.degrees(yaw_end - FORWARD_YAW), 3),
                 "rtf": round(c["seconds"] / wall, 1), **self._motion_stats(env, ticks)}
        finally:
            env.close()
        t = self.tol
        return ScenarioResult("straight", m, [
            Check("steady speed", abs(steady - c["v"]) / c["v"] <= t["speed_rel"], f"{steady:.4f} vs {c['v']}"),
            Check("acceleration limit", m["peak_accel_mps2"] <= self.limits["max_lin_acc_mps2"] * t["accel_factor"],
                  f"peak {m['peak_accel_mps2']} vs limit {self.limits['max_lin_acc_mps2']}"),
            Check("lateral drift", abs(x_end) <= t["lateral_drift_m"], f"{x_end:.4f} m"),
            Check("no collisions", m["collision_events"] == 0, str(m["collision_links"])),
            Check("not stuck", not m["stuck_detected"], ""),
        ], wall, c["seconds"])

    def spin(self) -> ScenarioResult:
        c = self.v["spin"]
        env = self._env([], "spin")
        try:
            env.reset((0.0, 0.0, 0.0), FORWARD_YAW)
            ticks, wall = self._drive(env, 0.0, c["w"], c["seconds"])
            yaw = np.unwrap([FORWARD_YAW] + [t.pose[2] for t in ticks])
            rate = float(np.mean(np.diff(yaw)[-20:]) / self.control_dt)
            drift = float(np.hypot(*ticks[-1].pose[:2]))
            m = {"yaw_rate_radps": round(rate, 4), "cmd_yaw_rate_radps": c["w"],
                 "center_drift_m": round(drift, 4), **self._motion_stats(env, ticks)}
        finally:
            env.close()
        t = self.tol
        return ScenarioResult("spin", m, [
            Check("yaw rate", abs(rate - c["w"]) / c["w"] <= t["speed_rel"], f"{rate:.4f} vs {c['w']}"),
            Check("turns in place", drift <= t["spin_center_drift_m"], f"centre moved {drift:.3f} m"),
        ], wall, c["seconds"])

    def arc(self) -> ScenarioResult:
        c = self.v["arc"]
        seconds = 2 * math.pi / c["w"]
        env = self._env([], "arc")
        try:
            env.reset((0.0, 0.0, 0.0), FORWARD_YAW)
            # let the drive reach the commanded twist before measuring the circle
            env.drive(c["v"], c["w"], int(round(1.0 / self.control_dt)))
            ticks, wall = self._drive(env, c["v"], c["w"], seconds)
            xy = np.array([t.pose[:2] for t in ticks])
            center, radius = _fit_circle(xy)
            m = {"radius_m": round(radius, 4), "expected_radius_m": round(c["v"] / c["w"], 4),
                 "center": [round(v, 3) for v in center]}
        finally:
            env.close()
        exp = c["v"] / c["w"]
        return ScenarioResult("arc", m, [
            Check("radius", abs(radius - exp) / exp <= self.tol["arc_radius_rel"], f"{radius:.4f} vs {exp:.4f}"),
        ], wall, seconds)

    def wall(self) -> ScenarioResult:
        c = self.v["wall"]
        d = c["distance_m"]
        wall_box = Box((0.0, 0.3, -(d + 0.05)), (2.0, 0.3, 0.05), name="wall")
        env = self._env([wall_box], "wall")
        try:
            obs0 = env.reset((0.0, 0.0, 0.0), FORWARD_YAW)
            h, w = obs0.depth.shape
            depth_center = float(obs0.depth[h // 2, w // 2])
            ticks, wall_t = self._drive(env, c["v"], 0.0, c["seconds"])
            front = ticks[-1].pose[1] + self.geom.body_radius  # map y of the chassis front edge
            first = next((t.t for t in ticks if t.collision), math.nan)
            m = {"camera_depth_to_wall_m": round(depth_center, 4),
                 "expected_depth_m": round(d - self.geom.cam_forward, 4),
                 "front_edge_y_m": round(front, 4), "wall_face_y_m": d,
                 "first_contact_s": first, **self._motion_stats(env, ticks)}
        finally:
            env.close()
        return ScenarioResult("wall", m, [
            Check("camera mount (depth to wall)", abs(depth_center - (d - self.geom.cam_forward)) <= 0.01,
                  f"{depth_center:.4f} vs {d - self.geom.cam_forward:.4f}"),
            Check("collision detected", m["collision_events"] >= 1, f"events={m['collision_events']}"),
            Check("no penetration", front <= d + 0.01, f"front edge {front:.4f} vs wall {d}"),
            Check("stuck detected against the wall", m["stuck_detected"], f"stuck {m['stuck_time_s']} s"),
        ], wall_t, c["seconds"])

    def camera_height(self) -> ScenarioResult:
        env = self._env([], "camera")
        try:
            obs = env.reset((0.0, 0.0, 0.0), FORWARD_YAW)
            rgbd = self.cfg["embodiment"]["sensors"]["rgbd"]
            H, W = obs.depth.shape
            hfov = math.radians(float(rgbd["hfov_deg"]))
            f = (W / 2) / math.tan(hfov / 2)
            pitch = math.radians(float(rgbd["mount"]["pitch_deg"]))
            # Floor pixels below the optical axis: with ray angle phi below the axis and pitch p,
            # height h = z * sin(phi - p) / cos(phi), where z is Habitat's planar (z-) depth.
            est = []
            for r in range(int(H * 0.8), H):
                phi = math.atan(((r + 0.5) - H / 2) / f)
                z = float(obs.depth[r, W // 2])
                if phi - pitch > 0 and z > 0:
                    est.append(z * math.sin(phi - pitch) / math.cos(phi))
            h_est = float(np.median(est))
            m = {"estimated_camera_height_m": round(h_est, 4), "configured_m": self.geom.cam_height,
                 "reference_height_m": round(float(obs.position[1]), 4)}
        finally:
            env.close()
        return ScenarioResult("camera_height", m, [
            Check("camera height from floor depth", abs(h_est - self.geom.cam_height) <= 0.005,
                  f"{h_est:.4f} vs {self.geom.cam_height}"),
            Check("robot rests on the floor", abs(m["reference_height_m"]) <= 0.003, f"{m['reference_height_m']}"),
        ])

    def steps(self) -> ScenarioResult:
        c = self.v["steps"]
        rows = []
        for hgt in c["heights_m"]:
            depth = 2.0
            step = Box((0.0, hgt / 2, -(c["distance_m"] + depth / 2)), (1.0, hgt / 2, depth / 2), name="step")
            env = self._env([step], f"step_{hgt}")
            try:
                env.reset((0.0, 0.0, 0.0), FORWARD_YAW)
                ticks, _ = self._drive(env, c["v"], 0.0, c["seconds"])
                pos = ticks[-1].position
                climbed = pos[1] >= hgt - 0.004 and -pos[2] > c["distance_m"] + 0.3
                rows.append({"height_m": hgt, "climbed": bool(climbed), "final_forward_m": round(-pos[2], 3),
                             "final_height_m": round(pos[1], 4), **self._motion_stats(env, ticks)})
            finally:
                env.close()
        climbed = [r["height_m"] for r in rows if r["climbed"]]
        max_climb = max(climbed) if climbed else 0.0
        nav_climb = float(self.cfg["embodiment"]["navmesh"]["agent_max_climb_m"])
        # monotone: nothing above the first failure should be climbable
        first_fail = min((r["height_m"] for r in rows if not r["climbed"]), default=math.inf)
        monotone = all(not r["climbed"] for r in rows if r["height_m"] > first_fail)
        return ScenarioResult("steps", {"rows": rows, "max_climbed_m": max_climb, "navmesh_max_climb_m": nav_climb}, [
            Check("navmesh climb <= physical climb", nav_climb <= max_climb + 1e-9,
                  f"navmesh {nav_climb} vs measured {max_climb}"),
            Check("climbing is monotone in height", monotone, ""),
        ])

    def clearance(self) -> ScenarioResult:
        c = self.v["clearance"]
        rows = []
        depth = 0.6
        for bottom in c["bottom_heights_m"]:
            t = c["thickness_m"]
            slab = Box((0.0, bottom + t / 2, -(c["distance_m"] + depth / 2)), (1.0, t / 2, depth / 2), name="slab")
            env = self._env([slab], f"slab_{bottom}")
            try:
                env.reset((0.0, 0.0, 0.0), FORWARD_YAW)
                ticks, _ = self._drive(env, c["v"], 0.0, c["seconds"])
                passed = -ticks[-1].position[2] > c["distance_m"] + depth + 0.2
                rows.append({"bottom_m": bottom, "passed": bool(passed), **self._motion_stats(env, ticks)})
            finally:
                env.close()
        top = self.geom.top_height
        nav_h = float(self.cfg["embodiment"]["navmesh"]["agent_height_m"])
        consistent = all(r["passed"] == (r["bottom_m"] > top) for r in rows)
        min_pass = min((r["bottom_m"] for r in rows if r["passed"]), default=math.inf)
        return ScenarioResult("clearance", {"rows": rows, "robot_top_m": top, "navmesh_agent_height_m": nav_h,
                                            "min_passed_bottom_m": min_pass}, [
            Check("passes iff slab is above the robot top", consistent, f"top {top:.3f} m"),
            Check("navmesh height >= robot top", nav_h >= top, f"{nav_h} vs {top:.3f}"),
        ])

    def push(self) -> ScenarioResult:
        c = self.v["push"]
        rows = []
        hb = c["box_half_m"]
        for mass in c["masses_kg"]:
            box = Box((0.0, hb, -(c["distance_m"] + hb)), (hb, hb, hb), static=False, mass_kg=mass,
                      friction=c["box_friction"], name="load")
            env = self._env([box], f"push_{mass}")
            try:
                rom = env.sim.get_rigid_object_manager()
                obj = [rom.get_object_by_handle(hd) for hd in rom.get_object_handles() if "load" in hd][0]
                p0 = np.array(obj.translation)
                env.reset((0.0, 0.0, 0.0), FORWARD_YAW)
                ticks, _ = self._drive(env, c["v"], 0.0, c["seconds"])
                moved = float(np.linalg.norm(np.array(obj.translation) - p0))
                friction_n = float(c["box_friction"]) * float(self.cfg["embodiment"]["physics"]["scene_friction"]) * mass * 9.8
                rows.append({"mass_kg": mass, "box_moved_m": round(moved, 3), "box_friction_force_n": round(friction_n, 2),
                             **self._motion_stats(env, ticks)})
            finally:
                env.close()
        pushed = [r for r in rows if r["box_moved_m"] > 0.3]
        blocked = [r for r in rows if r["box_moved_m"] < 0.05]
        max_traction = 2 * float(self.cfg["embodiment"]["drive"]["max_wheel_torque_nm"]) / self.geom.wheel_radius
        return ScenarioResult("push", {"rows": rows, "motor_force_limit_n": round(max_traction, 2)}, [
            Check("pushes a load needing less force than the motors give",
                  all(r["box_moved_m"] > 0.3 for r in rows if r["box_friction_force_n"] < 0.8 * max_traction), ""),
            Check("blocked by a load needing more force", all(r["box_moved_m"] < 0.05 and r["stuck_detected"]
                                                               for r in rows if r["box_friction_force_n"] > 1.2 * max_traction), ""),
        ])

    def determinism(self) -> ScenarioResult:
        finals = []
        for rep in range(int(self.v["determinism"]["repeats"])):
            env = self._env([Box((0.0, 0.3, -1.05), (2.0, 0.3, 0.05), name="wall")], f"det_{rep}")
            try:
                env.reset((0.0, 0.0, 0.0), FORWARD_YAW)
                env.drive(0.3, 0.2, 30)
                ticks = env.drive(0.3, 0.0, 40)
                finals.append(ticks[-1].pose + ticks[-1].position)
            finally:
                env.close()
        diff = max(abs(a - b) for a, b in zip(finals[0], finals[-1]))
        return ScenarioResult("determinism", {"max_abs_diff": diff, "final": finals[0]}, [
            Check("bit-identical repeat", diff == 0.0, f"max diff {diff:.3e}"),
        ])

    def run(self, only: Optional[Sequence[str]] = None) -> List[ScenarioResult]:
        order = ["camera_height", "straight", "spin", "arc", "wall", "steps", "clearance", "push", "determinism"]
        out = []
        for name in order:
            if only and name not in only:
                continue
            t0 = time.perf_counter()
            res = getattr(self, name)()
            res.wall_s = time.perf_counter() - t0
            out.append(res)
        return out


def _fit_circle(xy: np.ndarray) -> Tuple[Tuple[float, float], float]:
    """Algebraic least-squares circle fit (Kasa)."""
    x, y = xy[:, 0], xy[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = c[0] / 2, c[1] / 2
    r = math.sqrt(c[2] + cx * cx + cy * cy)
    return (float(cx), float(cy)), float(r)


def to_markdown(results: Sequence[ScenarioResult]) -> str:
    lines = ["# Robot physics validation", "", "| scenario | result | checks |", "|---|---|---|"]
    for r in results:
        checks = "; ".join(f"{'PASS' if c.passed else 'FAIL'} {c.name} ({c.detail})" for c in r.checks)
        lines.append(f"| {r.name} | {'PASS' if r.passed else 'FAIL'} | {checks} |")
    for r in results:
        lines += ["", f"## {r.name}", ""]
        rows = r.metrics.get("rows")
        if rows:
            keys = list(rows[0].keys())
            lines += ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
            for row in rows:
                lines.append("| " + " | ".join(str(row[k]) for k in keys) + " |")
            rest = {k: v for k, v in r.metrics.items() if k != "rows"}
            if rest:
                lines += ["", "```", *(f"{k}: {v}" for k, v in rest.items()), "```"]
        else:
            lines += ["```", *(f"{k}: {v}" for k, v in r.metrics.items()), "```"]
    return "\n".join(lines) + "\n"


def results_json(results: Sequence[ScenarioResult]) -> List[Dict[str, Any]]:
    return [{**asdict(r), "passed": r.passed} for r in results]
