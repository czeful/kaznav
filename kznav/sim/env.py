"""Habitat wrapper shared by both embodiments.

    env = HabitatEnv(cfg, SceneSpec(scene_id=...), workdir)
    obs = env.reset(position, yaw)
    ticks = env.drive(v, w, n_ticks)      # wheeled robot: continuous (v, w) control
    tick = env.act("move_forward")        # point agent: discrete habitat actions
    obs = env.observe()

Everything the rest of the pipeline needs from the simulator goes through this class:
observations, the robot pose, contact/collision flags per control tick, the navmesh of the
evaluated embodiment and geodesic distances on it.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .frames import habitat_to_map, quat_from_yaw, yaw_from_quat
from .noise import build_noise_models


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str                               # GLB path, a dataset scene handle, or "NONE"
    scene_dataset_config: Optional[str] = None
    #: Called after the simulator is created and before the navmesh is (re)computed;
    #: used to build primitive arenas and add objects.
    populate: Optional[Callable[[Any], None]] = None


@dataclass
class Observation:
    rgb: np.ndarray
    depth: np.ndarray
    semantic: Optional[np.ndarray]
    pose: Tuple[float, float, float]            # map frame (x, y, yaw) of the reference point
    position: np.ndarray                        # habitat world position of the reference point
    camera_transform: np.ndarray                # 4x4 world transform of the optical centre
    sim_time: float


@dataclass(frozen=True)
class Tick:
    """One control tick (wheeled) or one discrete action (point agent)."""

    t: float
    pose: Tuple[float, float, float]
    position: Tuple[float, float, float]
    v_cmd: float
    w_cmd: float
    collision: bool
    collision_links: Tuple[str, ...] = ()
    max_collision_force_n: float = 0.0
    tilt_deg: float = 0.0
    odom_twist: Optional[Tuple[float, float]] = None
    cliff: bool = False                     # front cliff sensors see no floor (wheeled robot)


def physics_config_json(emb: Mapping[str, Any]) -> Dict[str, Any]:
    phys = emb.get("physics") or {}
    return {
        "physics_simulator": "bullet",
        "timestep": float(phys.get("physics_dt_s", 0.008)),
        "gravity": [0.0, -9.8, 0.0],
        "friction_coefficient": float(phys.get("scene_friction", 0.4)),
        "restitution_coefficient": float(phys.get("scene_restitution", 0.1)),
    }


class HabitatEnv:
    def __init__(self, cfg: Mapping[str, Any], scene: SceneSpec, workdir: Path, seed: int = 0):
        import habitat_sim

        self._hs = habitat_sim
        self.cfg = cfg
        self.emb = cfg["embodiment"]
        self.kind = str(self.emb["kind"])
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.depth_noise, self.actuation_noise, self.odometry_drift = build_noise_models(self.emb["noise"], seed)
        wheeled = self.kind == "wheeled"

        physics_file = None
        if wheeled:
            # Habitat only recognises physics config files by this suffix.
            physics_file = self.workdir / "kznav.physics_config.json"
            physics_file.write_text(json.dumps(physics_config_json(self.emb), indent=2), encoding="utf-8")

        def sim_cfg(scene_id: str):
            c = habitat_sim.SimulatorConfiguration()
            c.scene_id = scene_id
            if scene.scene_dataset_config:
                c.scene_dataset_config_file = scene.scene_dataset_config
            c.enable_physics = wheeled
            c.gpu_device_id = int(cfg["sim"]["gpu_device_id"])
            c.random_seed = int(seed)
            if physics_file is not None:
                c.physics_config_file = str(physics_file)
            return c

        agents = [self._agent_config()]
        if wheeled and scene.scene_id != "NONE":
            # Stage physics attributes (collision margin, friction) must be registered BEFORE the
            # stage is loaded: start empty, override the stage template, then load the scene (D-015).
            # reconfigure() needs a NEW SimulatorConfiguration: it detects a scene change by comparing
            # against the object it already holds.
            self.sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg("NONE"), agents))
            self.stage_attributes = self._override_stage_template(scene.scene_id)
            self.sim.reconfigure(habitat_sim.Configuration(sim_cfg(scene.scene_id), agents))
            applied = self.sim.get_stage_initialization_template()
            if applied is None or abs(applied.margin - self.stage_attributes["margin"]) > 1e-9:
                raise RuntimeError("stage physics override was not applied")
        else:
            self.sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg(scene.scene_id), agents))
            self.stage_attributes = {}
        self.agent = self.sim.get_agent(0)
        if scene.populate is not None:
            scene.populate(self.sim)
        self._setup_navmesh()
        self.robot = None
        if wheeled:
            from .robot import WheeledRobot

            self.robot = WheeledRobot(self.sim, self.emb, self.workdir, self.actuation_noise)
        self.sim_time = 0.0

    # ------------------------------------------------------------------ construction
    def _agent_config(self):
        hs = self._hs
        rgbd = self.emb["sensors"]["rgbd"]
        h, w = (int(v) for v in rgbd["resolution_hw"])
        # Wheeled: the agent node is moved to the optical centre each frame, sensors sit at its origin.
        # Point agent: sensors are offset from the agent node like habitat-lab's ObjectNav agent.
        offset = [0.0, 0.0, 0.0] if self.kind == "wheeled" else [0.0, float(rgbd["mount"]["height_m"]), 0.0]
        pitch = 0.0 if self.kind == "wheeled" else math.radians(float(rgbd["mount"]["pitch_deg"]))
        specs = []
        kinds = [("rgb", hs.SensorType.COLOR), ("depth", hs.SensorType.DEPTH)]
        if self.emb["sensors"].get("semantic"):
            kinds.append(("semantic", hs.SensorType.SEMANTIC))
        for uuid, st in kinds:
            s = hs.CameraSensorSpec()
            s.uuid, s.sensor_type, s.resolution = uuid, st, [h, w]
            s.position, s.orientation = offset, [pitch, 0.0, 0.0]
            s.hfov = float(rgbd["hfov_deg"])
            specs.append(s)
        ag = hs.agent.AgentConfiguration()
        ag.sensor_specifications = specs
        body = self.emb["body"]
        ag.height, ag.radius = float(body.get("height_m", 0.88)), float(body["radius_m"])
        if self.kind == "point_agent":
            d = self.emb["drive"]
            ag.action_space = {
                "move_forward": hs.agent.ActionSpec("move_forward", hs.agent.ActuationSpec(amount=float(d["forward_step_m"]))),
                "turn_left": hs.agent.ActionSpec("turn_left", hs.agent.ActuationSpec(amount=float(d["turn_angle_deg"]))),
                "turn_right": hs.agent.ActionSpec("turn_right", hs.agent.ActuationSpec(amount=float(d["turn_angle_deg"]))),
                "look_up": hs.agent.ActionSpec("look_up", hs.agent.ActuationSpec(amount=float(d["tilt_angle_deg"]))),
                "look_down": hs.agent.ActionSpec("look_down", hs.agent.ActuationSpec(amount=float(d["tilt_angle_deg"]))),
            }
        return ag

    def _override_stage_template(self, scene_id: str) -> Dict[str, Any]:
        phys = self.emb["physics"]
        stm = self.sim.get_stage_template_manager()
        handles = [h for h in stm.get_template_handles() if h == scene_id] or             stm.get_template_handles(Path(scene_id).name.split(".")[0])
        tpl = stm.get_template_by_handle(handles[0]) if handles else stm.create_new_template(scene_id, False)
        tpl.margin = float(phys["scene_collision_margin_m"])
        tpl.friction_coefficient = float(phys["scene_friction"])
        tpl.restitution_coefficient = float(phys["scene_restitution"])
        stm.register_template(tpl, tpl.handle)
        return {"handle": tpl.handle, "margin": tpl.margin, "friction": tpl.friction_coefficient,
                "restitution": tpl.restitution_coefficient}

    def navmesh_settings(self):
        n = self.emb["navmesh"]
        s = self._hs.NavMeshSettings()
        s.set_defaults()
        s.agent_radius = float(n["agent_radius_m"])
        s.agent_height = float(n["agent_height_m"])
        s.agent_max_climb = float(n["agent_max_climb_m"])
        s.agent_max_slope = float(n["agent_max_slope_deg"])
        s.cell_size = float(n["cell_size_m"])
        s.cell_height = float(n["cell_height_m"])
        return s

    def _setup_navmesh(self) -> None:
        n = self.emb["navmesh"]
        if n["source"] == "dataset" and self.sim.pathfinder.is_loaded:
            self.navmesh_origin = "dataset"
            return
        ok = self.sim.recompute_navmesh(self.sim.pathfinder, self.navmesh_settings(), include_static_objects=True)
        if not ok:
            raise RuntimeError("navmesh recomputation failed")
        self.navmesh_origin = "recomputed"

    # ------------------------------------------------------------------ episode control
    def reset(self, position: Sequence[float], yaw: float) -> Observation:
        """Place the embodiment with its reference point at ``position`` (on the floor) and map heading ``yaw``."""
        self.sim_time = 0.0
        if self.robot is not None:
            self.robot.place(position, yaw)
            self._settle()
        else:
            st = self._hs.AgentState()
            st.position = np.asarray(position, dtype=np.float32)
            w, x, y, z = quat_from_yaw(yaw)
            import quaternion  # numpy-quaternion ships with habitat-sim

            st.rotation = quaternion.quaternion(w, x, y, z)
            self.agent.set_state(st, reset_sensors=True)
        return self.observe()

    def _settle(self, seconds: float = 0.25) -> None:
        """Let the robot come to rest on its wheels before the episode clock starts."""
        self.robot.command(0.0, 0.0)
        self.sim.step_physics(seconds)
        self.robot.limiter.reset()

    def drive(self, v_cmd: float, w_cmd: float, n_ticks: int = 1) -> List[Tick]:
        if self.robot is None:
            raise RuntimeError("drive() needs the wheeled embodiment; use act() for the point agent")
        out = []
        dt = self.robot.control_dt
        for _ in range(n_ticks):
            v, w = self.robot.command(v_cmd, w_cmd)
            self.sim.step_physics(dt)
            self.sim_time += dt
            c = self.robot.contacts()
            pos = self.robot.reference_position()
            out.append(Tick(
                t=self.sim_time, pose=self.robot.pose2d(), position=(float(pos[0]), float(pos[1]), float(pos[2])),
                v_cmd=v, w_cmd=w, collision=c.collision, collision_links=c.collision_links,
                max_collision_force_n=c.max_collision_force_n, tilt_deg=self.robot.tilt_deg(),
                odom_twist=self.robot.odometry_twist(),
                cliff=self.robot.cliff_detected(self.emb["sensors"].get("cliff", {})),
            ))
        return out

    def act(self, action: str) -> Tick:
        if self.robot is not None:
            raise RuntimeError("act() is for the point agent; the wheeled robot uses drive()")
        collided = bool(self.agent.act(action))  # navmesh sliding; observations are rendered in observe()
        self.sim_time += 1.0
        st = self.agent.get_state()
        pos = np.asarray(st.position, dtype=np.float64)
        r = st.rotation
        yaw = yaw_from_quat((r.w, r.x, r.y, r.z))
        x, y = habitat_to_map(pos)
        return Tick(t=self.sim_time, pose=(x, y, yaw), position=tuple(float(v) for v in pos),  # type: ignore[arg-type]
                    v_cmd=0.0, w_cmd=0.0, collision=collided)

    def observe(self) -> Observation:
        if self.robot is not None:
            T = self.robot.camera_transform(self.emb["sensors"]["rgbd"]["mount"])
            self.agent.scene_node.transformation = T
            pos = self.robot.reference_position()
            pose = self.robot.pose2d()
            cam = magnum_to_numpy(T)
        else:
            st = self.agent.get_state()
            pos = np.asarray(st.position, dtype=np.float64)
            r = st.rotation
            pose = (*habitat_to_map(pos), yaw_from_quat((r.w, r.x, r.y, r.z)))
            ss = st.sensor_states["rgb"]
            q = ss.rotation
            cam = pose_matrix(ss.position, (q.w, q.x, q.y, q.z))
        obs = self.sim.get_sensor_observations()
        depth = self.depth_noise.apply(np.asarray(obs["depth"], dtype=np.float32))
        return Observation(
            rgb=np.asarray(obs["rgb"])[..., :3], depth=depth, semantic=obs.get("semantic"),
            pose=pose, position=pos, camera_transform=cam, sim_time=self.sim_time,
        )

    def camera_transform_at(self, position: Sequence[float], yaw: float) -> np.ndarray:
        """4x4 world transform of the embodiment's camera if its reference point stood at ``position``
        with map heading ``yaw`` (level floor assumed). Used for view points without moving the robot."""
        mount = self.emb["sensors"]["rgbd"]["mount"]
        theta = yaw - math.pi / 2                       # rotation about +Y from the default heading
        c, s_ = math.cos(theta), math.sin(theta)
        R_yaw = np.array([[c, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c]])
        p = math.radians(float(mount["pitch_deg"]))
        R_pitch = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(p), -math.sin(p)], [0.0, math.sin(p), math.cos(p)]])
        T = np.eye(4)
        T[:3, :3] = R_yaw @ R_pitch
        T[:3, 3] = np.asarray(position, dtype=np.float64)[:3] + R_yaw @ np.array(
            [0.0, float(mount["height_m"]), -float(mount["forward_m"])])
        return T

    def render_at(self, position: Sequence[float], yaw: float) -> Dict[str, np.ndarray]:
        """Sensor observations from a hypothetical pose; the embodiment's actual state is restored."""
        import magnum as mn

        node = self.agent.scene_node
        saved = node.transformation
        T = self.camera_transform_at(position, yaw)
        if self.robot is None:
            # point-agent sensors are offset by the mount height from the agent node
            T = T.copy()
            T[1, 3] -= float(self.emb["sensors"]["rgbd"]["mount"]["height_m"])
        node.transformation = mn.Matrix4([[T[r][c] for r in range(4)] for c in range(4)])
        try:
            return dict(self.sim.get_sensor_observations())
        finally:
            node.transformation = saved

    # ------------------------------------------------------------------ navmesh queries
    # ``pathfinder=None`` means the evaluation navmesh of this embodiment (used for SPL / reachability).
    def make_pathfinder(self, extra_radius_m: float):
        """A separate navmesh with the robot radius inflated by ``extra_radius_m`` (planning with a safety margin)."""
        pf = self._hs.nav.PathFinder()
        s = self.navmesh_settings()
        s.agent_radius = s.agent_radius + float(extra_radius_m)
        if not self.sim.recompute_navmesh(pf, s, include_static_objects=True):
            raise RuntimeError("navmesh recomputation failed")
        return pf

    def _pf(self, pathfinder):
        return self.sim.pathfinder if pathfinder is None else pathfinder

    def geodesic_distance(self, start: Sequence[float], goals: Sequence[Sequence[float]], pathfinder=None) -> float:
        path = self._hs.MultiGoalShortestPath()
        path.requested_start = np.asarray(start, dtype=np.float32)
        path.requested_ends = [np.asarray(g, dtype=np.float32) for g in goals]
        if not self._pf(pathfinder).find_path(path):
            return math.inf
        return float(path.geodesic_distance)

    def shortest_path(self, start: Sequence[float], goal: Sequence[float], pathfinder=None) -> List[np.ndarray]:
        path = self._hs.ShortestPath()
        path.requested_start = np.asarray(start, dtype=np.float32)
        path.requested_end = np.asarray(goal, dtype=np.float32)
        if not self._pf(pathfinder).find_path(path):
            return []
        return [np.asarray(p, dtype=np.float64) for p in path.points]

    def snap(self, point: Sequence[float], pathfinder=None) -> np.ndarray:
        pf = self._pf(pathfinder)
        return np.asarray(pf.snap_point(np.asarray(point, dtype=np.float32)), dtype=np.float64)

    def random_navigable_point(self, pathfinder=None) -> np.ndarray:
        return np.asarray(self._pf(pathfinder).get_random_navigable_point(), dtype=np.float64)

    def navigable_area(self) -> float:
        return float(self.sim.pathfinder.navigable_area)

    def close(self) -> None:
        self.sim.close()


def magnum_to_numpy(m: Any) -> np.ndarray:
    """Magnum Matrix4 (column-major; m[c] is column c) -> numpy 4x4 with M[r, c]."""
    return np.array([[m[c][r] for c in range(4)] for r in range(4)], dtype=np.float64)


def pose_matrix(position: Sequence[float], q_wxyz: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in q_wxyz)
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = np.asarray(position, dtype=np.float64)[:3]
    return M
