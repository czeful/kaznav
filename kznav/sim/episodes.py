"""ObjectNav episodes independent of the source dataset, and per-embodiment goal view points.

Standard HM3D episodes define success through *view points*: navigable positions within 1 m of a
goal instance from which the object is visible, computed for the 0.88 m agent. For another
embodiment they must be recomputed with its own navmesh and camera (D-001):
``compute_view_points`` samples a grid around the object's footprint, keeps candidates on the
embodiment's navmesh, turns the camera towards the object and keeps the pose if at least
``min_pixels`` of the instance are visible in the semantic image.

The debug scene helpers place objects with known semantic ids into any scene; such scenes have
``role: debug`` and never enter the main tables.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .frames import habitat_to_map


@dataclass
class GoalInstance:
    object_id: str
    semantic_id: int
    category: str
    position: List[float]            # habitat world, AABB centre
    aabb_min: List[float]
    aabb_max: List[float]
    view_points: List[List[float]] = field(default_factory=list)


@dataclass
class ObjectNavEpisode:
    episode_id: str
    scene_id: str
    dataset: str
    split: str
    start_position: List[float]
    start_yaw: float
    goal_category: str
    goals: List[GoalInstance]
    start_geodesic_m: float          # on the evaluated embodiment's navmesh, to the nearest view point
    embodiment: str
    info: Dict[str, Any] = field(default_factory=dict)

    def view_points(self) -> List[List[float]]:
        return [vp for g in self.goals for vp in g.view_points]

    def goal_semantic_ids(self) -> List[int]:
        return [g.semantic_id for g in self.goals]


def save_episodes(path: Path, episodes: Sequence[ObjectNavEpisode]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([asdict(e) for e in episodes], indent=1), encoding="utf-8")


def load_episodes(path: Path) -> List[ObjectNavEpisode]:
    out = []
    for d in json.loads(Path(path).read_text(encoding="utf-8")):
        d["goals"] = [GoalInstance(**g) for g in d["goals"]]
        out.append(ObjectNavEpisode(**d))
    return out


# ---------------------------------------------------------------------------- view points
def footprint_distance(p: Sequence[float], aabb_min: Sequence[float], aabb_max: Sequence[float]) -> float:
    """Horizontal distance from a point to the object's axis-aligned footprint (0 inside)."""
    dx = max(aabb_min[0] - p[0], 0.0, p[0] - aabb_max[0])
    dz = max(aabb_min[2] - p[2], 0.0, p[2] - aabb_max[2])
    return math.hypot(dx, dz)


def compute_view_points(env: Any, goal: GoalInstance, *, success_radius_m: float, grid_step_m: float,
                        min_pixels: int, max_snap_m: float, success_distance_m: float = 0.1) -> List[List[float]]:
    """View points on a grid. The grid step must not exceed success_distance * sqrt(2): otherwise a valid
    stopping pose can be farther than the success distance from every grid node, and success would
    depend on the grid instead of on the robot."""
    if grid_step_m > success_distance_m * math.sqrt(2) + 1e-12:
        raise ValueError(f"view-point grid step {grid_step_m} m exceeds success_distance*sqrt(2) = "
                         f"{success_distance_m * math.sqrt(2):.3f} m")
    lo, hi = goal.aabb_min, goal.aabb_max
    xs = np.arange(lo[0] - success_radius_m, hi[0] + success_radius_m + 1e-9, grid_step_m)
    zs = np.arange(lo[2] - success_radius_m, hi[2] + success_radius_m + 1e-9, grid_step_m)
    centre_xy = np.asarray(habitat_to_map(goal.position))
    floor_guess = lo[1]
    out: List[List[float]] = []
    for x in xs:
        for z in zs:
            cand = np.array([x, floor_guess, z])
            d = footprint_distance(cand, lo, hi)
            if d <= 0.0 or d > success_radius_m:
                continue
            snapped = env.snap(cand)
            if not np.all(np.isfinite(snapped)) or math.hypot(*(snapped[[0, 2]] - cand[[0, 2]])) > max_snap_m:
                continue
            if abs(snapped[1] - floor_guess) > 0.5:      # another floor
                continue
            here = np.asarray(habitat_to_map(snapped))
            yaw = math.atan2(*(centre_xy - here)[::-1])
            sem = env.render_at(snapped, yaw).get("semantic")
            if sem is not None and int((np.asarray(sem) == goal.semantic_id).sum()) >= min_pixels:
                out.append([float(v) for v in snapped])
    return out


# ---------------------------------------------------------------------------- debug scenes
@dataclass(frozen=True)
class ObjectSpec:
    category: str
    config_path: str                 # habitat object_config.json
    count: int


def placement_populate(specs: Sequence[ObjectSpec], rng: np.random.Generator, placed: List[GoalInstance],
                       min_separation_m: float = 1.5, collision_margin_m: float = 0.0,
                       first_semantic_id: int = 1) -> Callable[[Any], None]:
    """``SceneSpec.populate`` callback: put static objects on random navigable spots; fills ``placed``."""

    def _populate(sim: Any) -> None:
        import magnum as mn
        from habitat_sim.physics import MotionType

        otm = sim.get_object_template_manager()
        rom = sim.get_rigid_object_manager()
        sim.pathfinder.seed(int(rng.integers(0, 2**31 - 1)))
        sem_id = first_semantic_id
        for spec in specs:
            tid = otm.load_configs(spec.config_path)[0]
            tpl = otm.get_template_by_id(tid)
            tpl.margin = float(collision_margin_m)     # same reason as the stage margin (D-015)
            otm.register_template(tpl, tpl.handle)
            for _ in range(spec.count):
                for _attempt in range(200):
                    p = np.asarray(sim.pathfinder.get_random_navigable_point(), dtype=np.float64)
                    if all(math.hypot(p[0] - g.position[0], p[2] - g.position[2]) >= min_separation_m for g in placed):
                        break
                obj = rom.add_object_by_template_id(tid)
                yaw = float(rng.uniform(-math.pi, math.pi))
                obj.rotation = mn.Quaternion.rotation(mn.Rad(yaw), mn.Vector3(0.0, 1.0, 0.0))
                bb = obj.root_scene_node.cumulative_bb
                obj.translation = mn.Vector3(p[0], p[1] - bb.min[1], p[2])
                obj.motion_type = MotionType.STATIC
                obj.semantic_id = sem_id
                corners = np.array([[x, y, z] for x in (bb.min[0], bb.max[0]) for y in (bb.min[1], bb.max[1])
                                    for z in (bb.min[2], bb.max[2])])
                c, s = math.cos(yaw), math.sin(yaw)
                R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
                world = corners @ R.T + np.array([p[0], p[1] - bb.min[1], p[2]])
                lo, hi = world.min(axis=0), world.max(axis=0)
                placed.append(GoalInstance(
                    object_id=f"{spec.category}_{sem_id}", semantic_id=sem_id, category=spec.category,
                    position=[float(v) for v in (lo + hi) / 2], aabb_min=[float(v) for v in lo],
                    aabb_max=[float(v) for v in hi]))
                sem_id += 1

    return _populate


def sample_episodes(env: Any, goals: Sequence[GoalInstance], category: str, n: int, rng: np.random.Generator,
                    *, min_geodesic_m: float, max_geodesic_m: float, scene_id: str, dataset: str, split: str,
                    embodiment: str, max_tries: int = 5000) -> List[ObjectNavEpisode]:
    cat_goals = [g for g in goals if g.category == category and g.view_points]
    vps = [vp for g in cat_goals for vp in g.view_points]
    if not vps:
        return []
    env.sim.pathfinder.seed(int(rng.integers(0, 2**31 - 1)))
    out: List[ObjectNavEpisode] = []
    for _ in range(max_tries):
        if len(out) >= n:
            break
        start = env.random_navigable_point()
        d = env.geodesic_distance(start, vps)
        if not (min_geodesic_m <= d <= max_geodesic_m):
            continue
        out.append(ObjectNavEpisode(
            episode_id=f"{Path(scene_id).stem}_{category.replace(' ', '-')}_{len(out):03d}", scene_id=scene_id, dataset=dataset,
            split=split, start_position=[float(v) for v in start], start_yaw=float(rng.uniform(-math.pi, math.pi)),
            goal_category=category, goals=list(cat_goals), start_geodesic_m=float(d), embodiment=embodiment))
    return out
