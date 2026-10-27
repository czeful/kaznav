"""Per-episode debug video: camera with detections, the robot's map, the value map, and a text band with the
command, the hypotheses, the decision and the last verification. Shared by the debug and benchmark runners."""
from __future__ import annotations

from pathlib import Path

from .video import Mp4Writer, compose, depth_to_rgb, draw_text, render_policy_map, render_value_map

GREEN, RED, AMBER, CYAN = (90, 220, 90), (255, 90, 90), (255, 190, 60), (120, 210, 255)


class EpisodeVideo:
    """Writes one episode's debug video, one frame per decision step.

    Args:
        path: Output MP4 path.
        cfg: The run config (depth range and robot radius are read).
        label: Text shown in the frame's header.
        language: The episode's language goal, shown in the text band, or None.
    """
    def __init__(self, path: Path, cfg, label: str, language=None):
        self.w = Mp4Writer(path, fps=4.0)
        self.max_depth = float(cfg["embodiment"]["sensors"]["rgbd"]["depth"]["max_m"])
        self.radius = float(cfg["embodiment"]["body"]["radius_m"])
        self.label = label
        self.language = language

    def __call__(self, *, obs, decision, policy, trace, step, episode):
        """Render and append the frame of one decision step (camera, maps, decision text)."""
        import cv2

        fr_cells = [f.cell for f in policy.last_frontiers]
        mp = render_policy_map(policy.map, pose=obs.pose, radius_m=self.radius,
                               trajectory=[p[:2] for p in trace.poses], path=decision.path_xy,
                               frontier_cells=fr_cells, candidates=policy.memory.candidates,
                               gt_boxes=[(g.aabb_min, g.aabb_max) for g in episode.goals])
        kind_color = {"stop": GREEN, "face": AMBER, "approach": CYAN}.get(decision.kind, (255, 255, 255))
        text = [f"{self.label} | {episode.episode_id} | step {step}  t={obs.sim_time:.1f}s"]
        lg = self.language
        if lg is not None:
            text.append((f"command ({lg.lang}): «{lg.text}»   ->  target: {lg.target or '-'}", AMBER))
            if lg.translation:
                text.append((f"translation ({lg.goal_mode}): «{lg.translation}»", AMBER))
            w = policy.query_weights or lg.weights
            text.append("hypotheses: " + "   ".join(f"{h} [{wt:.2f}]" for h, wt in zip(policy.queries, w)))
        else:
            text.append(f"goal (English category): {episode.goal_category}")
        text.append((f"decision: {decision.kind} - {decision.reason}", kind_color))
        v = policy.last_verify
        if v is not None:
            p = "-" if v.p_raw is None else f"{v.p_raw:.2f}"
            text.append((f"last verification (step {v.step}, cand {v.cid}): P(yes)={p} ({v.answer!r}) -> "
                         f"P(target)={v.p_candidate:.2f} -> {v.decision}",
                         GREEN if v.decision == "stop" else RED if v.decision == "reject" else AMBER))
        text.append(f"detections: {len(policy.last_detections)}  candidates: {len(policy.memory.candidates)} "
                    f"(rejected {sum(c.rejected for c in policy.memory.candidates)})  frontiers: {len(policy.last_frontiers)}"
                    "   map: cyan=frontier, green=GT goal (not seen by the robot)")
        rgb = obs.rgb.copy()
        for d in policy.last_detections:
            x1, y1, x2, y2 = (int(v) for v in d.box_xyxy)
            cv2.rectangle(rgb, (x1, y1), (x2, y2), RED, 2)
            draw_text(rgb, f"{d.phrase} {d.score:.2f}", (x1 + 2, max(0, y1 - 18)), RED, size=14)
        panels = [rgb, mp]
        vm = policy.value_map.normalized() if policy.value_map is not None else None
        if vm is not None:
            panels.append(render_value_map(policy.map, vm, obs.pose, self.radius))
        else:
            panels.insert(1, depth_to_rgb(obs.depth, self.max_depth))
        self.w.write(compose(panels, 360, text))

    def close(self):
        """Finish the MP4 file."""
        self.w.close()
