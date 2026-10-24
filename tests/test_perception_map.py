"""Occupancy mapping, frontiers and grid planning on analytically generated depth."""
import math

import numpy as np
import pytest

from kznav.perception.camera import Intrinsics, backproject, pixel_grid, to_world
from kznav.perception.frontiers import find_frontiers
from kznav.perception.occupancy import MapParams, OccupancyMap
from kznav.policy.planner import GridPlanner, PlannerParams, simplify_path

K = Intrinsics.from_hfov(69.0, 160, 120)
CAM_H = 0.15


def params(**kw):
    base = dict(size_m=10.0, resolution_m=0.05, pixel_stride=1, max_range_m=4.5, l_hit=0.85, l_miss=-0.5,
                l_min=-3.5, l_max=3.5, p_occupied=0.6, p_free=0.4, obstacle_min_height_m=0.03,
                obstacle_max_height_m=0.195)
    base.update(kw)
    return MapParams(**base)


def cam_T(x=0.0, z=0.0, h=CAM_H):
    """Camera at (x, h, z), looking along -Z (map +y), no pitch."""
    T = np.eye(4)
    T[:3, 3] = [x, h, z]
    return T


def ray_dirs():
    v, u = np.mgrid[0:K.height, 0:K.width]
    dx = (u + 0.5 - K.cx) / K.fx
    dy = -(v + 0.5 - K.cy) / K.fy
    return dx, dy          # direction (dx, dy, -1) per unit planar depth


def synth_depth(wall_at=None, table=None):
    """Planar depth for a flat floor (y=0), optional wall at map y = wall_at, optional table top.

    table = (y_near, y_far, height): a horizontal slab over the floor between those map-y distances.
    """
    dx, dy = ray_dirs()
    depth = np.full(dx.shape, np.inf)
    down = dy < 0
    depth[down] = CAM_H / -dy[down]                     # floor: CAM_H + dy * z = 0
    if wall_at is not None:
        depth = np.minimum(depth, wall_at)
    if table is not None:
        y0, y1, th = table
        up = dy > 0
        z_top = np.where(up, (th - CAM_H) / np.where(up, dy, 1), np.inf)   # hits the slab underside
        hit = up & (z_top >= y0) & (z_top <= y1)
        depth[hit] = np.minimum(depth[hit], z_top[hit])
    depth[~np.isfinite(depth)] = 0.0
    return depth.astype(np.float32)


class TestCamera:
    def test_backproject_project_round_trip(self):
        v, u = pixel_grid(K, 7)
        depth = np.full((K.height, K.width), 2.5, dtype=np.float32)
        p = backproject(depth, K, v, u)
        uu, vv = K.project(p)
        assert np.allclose(uu, u) and np.allclose(vv, v)
        assert np.allclose(p[:, 2], -2.5)

    def test_floor_points_land_on_floor(self):
        depth = synth_depth()
        v, u = pixel_grid(K, 2)
        ok = depth[v, u] > 0
        pts = to_world(backproject(depth, K, v[ok], u[ok]), cam_T())
        assert np.abs(pts[:, 1]).max() < 1e-6          # float32 depth


class TestOccupancy:
    def test_open_floor_is_free_not_occupied(self):
        m = OccupancyMap(params(), K, (0.0, 0.0))
        m.update(synth_depth(), cam_T(), floor_y=0.0)
        assert not m.occupied.any()
        r, c = m.to_cell(np.array([0.0]), np.array([1.0]))
        assert m.free[r[0], c[0]]
        r, c = m.to_cell(np.array([0.0]), np.array([-1.0]))       # behind the camera: never seen
        assert m.unknown[r[0], c[0]]

    def test_wall_is_occupied_space_before_it_free_behind_unknown(self):
        m = OccupancyMap(params(), K, (0.0, 0.0))
        for _ in range(3):
            m.update(synth_depth(wall_at=2.02), cam_T(), floor_y=0.0)
        ys = np.array([1.0, 2.02, 2.6])
        r, c = m.to_cell(np.zeros(3), ys)
        assert m.free[r[0], c[0]]
        assert m.occupied[r[1], c[1]]
        assert m.unknown[r[2], c[2]]

    def test_overhead_table_is_passable_for_a_low_robot(self):
        m = OccupancyMap(params(), K, (0.0, 0.0))
        m.update(synth_depth(table=(1.0, 3.0, 0.72)), cam_T(), floor_y=0.0)
        assert not m.occupied.any()

    def test_same_table_blocks_a_tall_agent(self):
        m = OccupancyMap(params(obstacle_max_height_m=0.91), K, (0.0, 0.0))
        m.update(synth_depth(table=(1.0, 3.0, 0.72)), cam_T(), floor_y=0.0)
        assert m.occupied.any()

    def test_log_odds_clamped(self):
        m = OccupancyMap(params(), K, (0.0, 0.0))
        for _ in range(20):
            m.update(synth_depth(wall_at=2.0), cam_T(), floor_y=0.0)
        assert m.log_odds.max() <= 3.5 + 1e-6 and m.log_odds.min() >= -3.5 - 1e-6

    def test_invalid_depth_ignored(self):
        m = OccupancyMap(params(), K, (0.0, 0.0))
        m.update(np.zeros((K.height, K.width), np.float32), cam_T(), floor_y=0.0)
        assert not m.observed.any()


def grid_map(free_box=None, walls=(), size=6.0):
    """Hand-built map: everything unknown except ``free_box`` (x0, y0, x1, y1) free; ``walls`` occupied boxes."""
    m = OccupancyMap(params(size_m=size), K, (0.0, 0.0))
    def box(b, value):
        r0, c0 = m.to_cell(np.array([b[0]]), np.array([b[1]]))
        r1, c1 = m.to_cell(np.array([b[2]]), np.array([b[3]]))
        m.log_odds[r0[0]:r1[0] + 1, c0[0]:c1[0] + 1] = value
        m.observed[r0[0]:r1[0] + 1, c0[0]:c1[0] + 1] = True
    if free_box:
        box(free_box, -3.0)
    for w in walls:
        box(w, 3.0)
    return m


PP = PlannerParams(inflation_margin_m=0.03, unknown_cost=1.5, proximity_weight=0.0, proximity_range_m=0.3,
                   crop_margin_m=1.0, start_escape_m=0.3)


class TestFrontiers:
    def test_frontier_at_edge_of_free_area(self):
        m = grid_map(free_box=(-1.0, -1.0, 1.0, 1.0))
        start = tuple(int(v[0]) for v in m.to_cell(np.array([0.0]), np.array([0.0])))
        fr = find_frontiers(m.free, m.unknown, m.inflated_obstacles(0.2), start, min_size=6)
        assert len(fr) == 1 and fr[0].size > 100      # the whole boundary ring is one region

    def test_walled_off_frontier_is_unreachable(self):
        # free room with a full wall ring; outside the ring unknown
        m = grid_map(free_box=(-2.0, -2.0, 2.0, 2.0), walls=[(-1.0, -1.0, 1.0, -0.9), (-1.0, 0.9, 1.0, 1.0),
                                                           (-1.0, -1.0, -0.9, 1.0), (0.9, -1.0, 1.0, 1.0)])
        start = tuple(int(v[0]) for v in m.to_cell(np.array([0.0]), np.array([0.0])))
        fr = find_frontiers(m.free, m.unknown, m.inflated_obstacles(0.2), start, min_size=6)
        assert fr == []

    def test_min_size_filter(self):
        m = grid_map(free_box=(-1.0, -1.0, 1.0, 1.0))
        start = tuple(int(v[0]) for v in m.to_cell(np.array([0.0]), np.array([0.0])))
        assert find_frontiers(m.free, m.unknown, m.inflated_obstacles(0.2), start, min_size=10_000) == []


class TestPlanner:
    def test_straight_line_cost_equals_distance(self):
        m = grid_map(free_box=(-2.5, -2.5, 2.5, 2.5))
        res = GridPlanner(PP, 0.17).plan(m, (-1.5, 0.0))
        goal = tuple(int(v[0]) for v in m.to_cell(np.array([1.5]), np.array([0.0])))
        assert res.ok and res.cost_to(goal) == pytest.approx(3.0, abs=0.06)

    def test_goes_through_gap(self):
        # wall across y = 0 with a 0.8 m gap at x in [0.6, 1.4]
        m = grid_map(free_box=(-2.5, -2.5, 2.5, 2.5), walls=[(-3.0, -0.05, 0.6, 0.05), (1.4, -0.05, 3.0, 0.05)])
        res = GridPlanner(PP, 0.17).plan(m, (-1.0, -1.0))
        goal = tuple(int(v[0]) for v in m.to_cell(np.array([-1.0]), np.array([1.0])))
        cells = res.path_cells(goal)
        xs, ys = m.to_xy(np.array([c[0] for c in cells]), np.array([c[1] for c in cells]))
        crossing = xs[np.argmin(np.abs(ys))]
        assert 0.6 + 0.2 <= crossing <= 1.4 - 0.2
        assert res.cost_to(goal) > 2.0 + 1e-6

    def test_gap_narrower_than_robot_is_closed(self):
        m = grid_map(free_box=(-2.5, -2.5, 2.5, 2.5), walls=[(-3.0, -0.05, 0.6, 0.05), (0.9, -0.05, 3.0, 0.05)])
        res = GridPlanner(PP, 0.17).plan(m, (-1.0, -1.0))
        goal = tuple(int(v[0]) for v in m.to_cell(np.array([-1.0]), np.array([1.0])))
        assert math.isinf(res.cost_to(goal))

    def test_unknown_is_optimistic_but_costlier(self):
        m = grid_map(free_box=(-2.5, -0.5, 2.5, 0.5))
        goal = tuple(int(v[0]) for v in m.to_cell(np.array([0.0]), np.array([2.0])))
        res = GridPlanner(PP, 0.17).plan(m, (0.0, 0.0), extra_cells=[goal])   # goal lies outside the observed crop
        # 0.5 m of free cells then 1.5 m of unknown at 1.5x
        assert res.cost_to(goal) == pytest.approx(0.5 + 1.5 * 1.5, abs=0.08)

    def test_start_escape_when_touching_wall(self):
        m = grid_map(free_box=(-2.5, -2.5, 2.5, 2.5), walls=[(-2.5, 0.0, 2.5, 0.1)])
        res = GridPlanner(PP, 0.17).plan(m, (0.0, -0.15))
        assert res.ok and res.start_escaped

    def test_simplified_path_is_short_and_collision_free(self):
        m = grid_map(free_box=(-2.5, -2.5, 2.5, 2.5), walls=[(-3.0, -0.05, 0.6, 0.05), (1.4, -0.05, 3.0, 0.05)])
        res = GridPlanner(PP, 0.17).plan(m, (-1.0, -1.0))
        goal = tuple(int(v[0]) for v in m.to_cell(np.array([-1.0]), np.array([1.0])))
        cells = res.path_cells(goal)
        simple = simplify_path(cells, res.blocked)
        assert simple[0] == res.start and simple[-1] == goal
        assert len(simple) < len(cells) / 5
        from kznav.policy.planner import _line_free
        assert all(_line_free(a, b, res.blocked) for a, b in zip(simple[:-1], simple[1:]))


def test_missing_floor_return_is_a_drop_off():
    """Downward pixels with no depth return mark where the floor should have been; others do nothing."""
    depth = synth_depth()
    dx, dy = ray_dirs()
    down = dy < 0
    z_floor = np.where(down, CAM_H / -np.where(down, dy, -1), np.inf)       # planar distance to the floor
    hole = down & (z_floor >= 1.0) & (z_floor <= 1.2) & (np.abs(dx * z_floor) <= 0.2)   # 20 cm x 40 cm hole
    depth[hole] = 0.0
    for rmax, expect in ((1.5, True), (0.0, False), (0.8, False)):
        m = OccupancyMap(params(p_occupied=0.8, missing_return_max_range_m=rmax), K, (0.0, 0.0))
        for _ in range(2):                                    # two consistent observations (D-019)
            m.update(depth, cam_T(), 0.0)
        r, c = m.to_cell(np.array([0.0]), np.array([1.1]))
        assert bool(m.occupied[r[0], c[0]]) is expect, rmax
        r, c = m.to_cell(np.array([0.0]), np.array([0.6]))
        assert not m.occupied[r[0], c[0]]                     # the floor before the hole stays free
    # no-return pixels above the horizon (sky, open door) never mark anything
    m = OccupancyMap(params(p_occupied=0.8, missing_return_max_range_m=1.5), K, (0.0, 0.0))
    sky = synth_depth()
    sky[~down] = 0.0
    m.update(sky, cam_T(), 0.0)
    m.update(sky, cam_T(), 0.0)
    assert not m.occupied.any()


def test_drops_are_obstacles_only_for_the_wheeled_robot():
    """D-052: the standard agent walks down stairs on its navmesh (as in VLFM); the wheeled robot treats drops as cliffs."""
    import math

    from kznav.perception.occupancy import MapParams
    from kznav.run.config import compose_config

    for emb, cliff in (("wheeled_vacuum", True), ("habitat_agent", False)):
        cfg = compose_config([f"embodiment={emb}"])
        p = MapParams.from_config(cfg["perception"]["map"], cfg["embodiment"])
        assert math.isfinite(p.cliff_depth_m) is cliff
        assert (p.missing_return_max_range_m > 0) is cliff


def test_box_area_fraction_and_candidate_point_filter():
    """D-054: frame-filling boxes can be dropped; candidate points lose the floor and the background behind the object."""
    from kznav.perception.grounding_dino import box_area_frac
    from kznav.perception.objects import DetectionParams, ObjectMemory

    assert box_area_frac((0, 0, 640, 480), 640, 480) == pytest.approx(1.0)
    assert box_area_frac((-10, 240, 320, 600), 640, 480) == pytest.approx(0.25)
    K = Intrinsics.from_hfov(90.0, 64, 48)
    depth = np.full((48, 64), 3.0, np.float32)          # wall 3 m away
    depth[10:30, 20:44] = 1.0                            # object 1 m away
    T = np.eye(4)
    T[1, 3] = 0.15                                       # camera 0.15 m above the floor, looking along -Z
    mask = np.zeros((48, 64), bool)
    mask[5:48, 15:50] = True                             # the box: object + wall + lower image rows
    plain = ObjectMemory(DetectionParams(0.3, 5, 0.5, 10_000, 5.0), K, 0)
    filt = ObjectMemory(DetectionParams(0.3, 5, 0.5, 10_000, 5.0, floor_margin_m=0.05, depth_band_m=0.5), K, 0)
    p_all = plain.localize(mask, depth, T, floor_y=0.0)
    p_obj = filt.localize(mask, depth, T, floor_y=0.0)
    assert np.isclose(-p_obj[:, 2], 1.0, atol=1e-5).all()          # only the object's front layer
    assert (-p_all[:, 2] > 2.0).any() and len(p_obj) < len(p_all)


class _FakeGdino:
    """Stand-in for the HF processor + model: a caption's score is looked up by its text; 'none' grounds no phrase."""

    def __init__(self, scores):
        self.scores, self.captions, self.resize = scores, [], []
        self.image_processor = self._pixels
        self.tokenizer = self._tokenize

    def _pixels(self, images, return_tensors, **kw):
        self.resize.append(kw.get("size"))
        return _ToDevice({"pixel_values": "px"})

    def _tokenize(self, caption, return_tensors):
        self.captions.append(caption)
        return _ToDevice({"input_ids": caption})

    def __call__(self, **inputs):           # the model
        return inputs["input_ids"]

    def post_process_grounded_object_detection(self, outputs, input_ids, threshold, text_threshold, target_sizes):
        hits = [(s, p) for s, p in self.scores.get(outputs, []) if s >= threshold]
        return [{"boxes": _L([[0, 0, 10, 10]] * len(hits)), "scores": _L([s for s, _ in hits]),
                 "text_labels": [p for _, p in hits]}]


class _ToDevice(dict):
    __getattr__ = dict.__getitem__

    def to(self, device):
        return self


class _L(list):
    def tolist(self):
        return list(self)


@pytest.mark.parametrize("mode,resize", [("separate", "native"), ("joint", "default")])
def test_grounding_dino_query_modes(mode, resize):
    """D-058: separate = one caption and forward pass per query, boxes belong to their caption's query (a phrase
    sharing no token with it is dropped, as in joint mode); joint = one caption, phrases matched back to queries."""
    import contextlib

    from kznav.perception.grounding_dino import GroundingDinoDetector

    fake = _FakeGdino({"red chair .": [(0.7, "red chair"), (0.9, "")],
                       "chair by a table .": [(0.8, "table")],
                       "red chair . chair by a table .": [(0.66, "red chair")]})
    det = GroundingDinoDetector.__new__(GroundingDinoDetector)
    det.spec = {}
    det.processor = det.model = fake
    det.torch = type("T", (), {"inference_mode": staticmethod(contextlib.nullcontext)})
    det.device, det.box_threshold, det.text_threshold = "cpu", 0.65, 0.25
    det.max_box_area_frac, det.query_mode, det.input_resize = 1.0, mode, resize
    det.dropped_large_boxes = det.unmatched_phrases = 0
    out = det.detect(np.zeros((48, 64, 3), np.uint8), ["Red chair", "chair by a table"])
    if mode == "separate":
        assert fake.captions == ["red chair .", "chair by a table ."]
        assert [(d.query_index, d.score) for d in out] == [(0, 0.7), (1, 0.8)]
        assert det.unmatched_phrases == 1
    else:
        assert fake.captions == ["red chair . chair by a table ."]
        assert [(d.query_index, d.score) for d in out] == [(0, 0.66)]
    # D-059: native = the frame's own size (one image preprocessing per frame); default = the processor's resize
    assert fake.resize == [{"shortest_edge": 48, "longest_edge": 64} if resize == "native" else None]


@pytest.mark.parametrize("key", ["query_mode", "input_resize"])
def test_grounding_dino_rejects_unknown_modes(key):
    from kznav.perception.grounding_dino import GroundingDinoDetector

    with pytest.raises(ValueError, match=key):
        GroundingDinoDetector({"model_id": "unused", "box_threshold": 0.5, "text_threshold": 0.25, key: "x"})


def _fusion_memory(**kw):
    from kznav.perception.objects import DetectionParams, ObjectMemory

    K = Intrinsics.from_hfov(90.0, 64, 48)
    p = dict(score_threshold=0.3, min_mask_pixels=5, merge_radius_m=0.5, max_points=10_000, max_depth_m=5.0,
             fusion=True, reliable_conf=0.4, min_visible_points=5, occlusion_tol_m=0.3)
    p.update(kw)
    return ObjectMemory(DetectionParams(**p), K, 0), K


def _scene():
    depth = np.full((48, 64), 3.0, np.float32)          # wall 3 m away
    depth[10:30, 20:44] = 1.0                            # object 1 m away
    mask = np.zeros((48, 64), bool)
    mask[10:30, 20:44] = True
    return depth, mask, np.eye(4)                        # camera at the origin, looking along -Z


def test_fusion_confirms_detected_and_penalises_unseen_candidates():
    """D-060 (ApexNav fusion): a candidate in view but not detected loses confidence and becomes suspected."""
    from kznav.perception.base import Detection

    mem, _ = _fusion_memory()
    depth, mask, T = _scene()
    det = Detection((20, 10, 44, 30), 0.6, 0, "chair")
    c = mem.add(det, mask, depth, T, step=0)
    mem.end_frame(depth, T)
    assert c.conf == pytest.approx(0.6) and mem.best() is c
    mem.end_frame(depth, T)                              # next frame: in view, no detection
    assert c.conf == pytest.approx(0.3) and c.misses == 1
    assert mem.best() is None and mem.best(include_suspected=True) is c
    mem.add(det, mask, depth, T, step=2)                 # seen again: back above the bar
    mem.end_frame(depth, T)
    assert c.conf == pytest.approx(0.4) and mem.best() is c


def test_fusion_ignores_occluded_or_out_of_view_candidates():
    from kznav.perception.base import Detection

    mem, _ = _fusion_memory()
    depth, mask, T = _scene()
    c = mem.add(Detection((20, 10, 44, 30), 0.6, 0, "chair"), mask, depth, T, step=0)
    mem.end_frame(depth, T)
    occluded = depth.copy()
    occluded[:, :] = 0.5                                 # something 0.5 m in front of the camera hides it
    mem.end_frame(occluded, T)
    turned = np.eye(4)
    turned[:3, :3] = [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]   # yaw 180 deg: the object is behind the camera
    mem.end_frame(depth, turned)
    assert c.conf == pytest.approx(0.6) and c.misses == 0


def test_fusion_counts_a_frame_once_and_off_keeps_the_old_ranking():
    """Several hypotheses firing on one object in one frame are one observation (the best score); without fusion
    the ranking is the best detector score, as before D-060."""
    from kznav.perception.base import Detection

    mem, _ = _fusion_memory()
    depth, mask, T = _scene()
    for qi, s in enumerate((0.5, 0.7, 0.6)):
        c = mem.add(Detection((20, 10, 44, 30), s, qi, "chair"), mask, depth, T, step=0)
    mem.end_frame(depth, T)
    assert len(mem.candidates) == 1 and c.conf == pytest.approx(0.7)
    off, _ = _fusion_memory(fusion=False)
    weak = off.add(Detection((20, 10, 44, 30), 0.5, 0, "chair"), mask, depth, T, step=0)
    off.end_frame(depth, T)
    off.end_frame(depth, T)
    assert weak.conf == 0.0 and off.best() is weak


def test_closed_set_detector_maps_categories_to_coco_and_ignores_language():
    """D-062: the VLFM reference detector answers only queries that name a COCO class (directly or by alias)."""
    import contextlib

    from kznav.perception.coco_detector import CocoClosedSetDetector

    class Proc:
        def __call__(self, images, return_tensors):
            return _ToDevice({"pixel_values": "px"})

        def post_process_object_detection(self, outputs, threshold, target_sizes):
            hits = [(b, s, l) for b, s, l in [([0, 0, 10, 10], 0.9, 57), ([5, 5, 20, 20], 0.85, 62),
                                              ([1, 1, 4, 4], 0.5, 57)] if s >= threshold]
            return [{"boxes": _L([b for b, _, _ in hits]), "scores": _L([s for _, s, _ in hits]),
                     "labels": _L([l for _, _, l in hits])}]

    det = CocoClosedSetDetector.__new__(CocoClosedSetDetector)
    det.spec, det.device, det.box_threshold, det.max_box_area_frac = {}, "cpu", 0.8, 1.0
    det.aliases = {"sofa": "couch", "tv_monitor": "tv"}
    det.class_ids = {"couch": 57, "tv": 62, "chair": 56}
    det.processor = Proc()
    det.model = type("M", (), {"__call__": lambda self, **kw: None,
                               "config": type("C", (), {"id2label": {57: "couch", 62: "tv", 56: "chair"}})})()
    det.torch = type("T", (), {"inference_mode": staticmethod(contextlib.nullcontext)})
    det.unmapped_queries = det.dropped_large_boxes = 0
    assert det.coco_class("Sofa") == 57 and det.coco_class("tv_monitor") == 62 and det.coco_class("диван") is None
    out = det.detect(np.zeros((48, 64, 3), np.uint8), ["sofa", "a red couch in the living room", "tv_monitor"])
    assert [(d.query_index, d.phrase) for d in out] == [(0, "couch"), (2, "tv")]
    assert det.unmapped_queries == 1                    # free text is not a COCO class: no detections for it
    assert det.detect(np.zeros((48, 64, 3), np.uint8), ["қазан"]) == []
