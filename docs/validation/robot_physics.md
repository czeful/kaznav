# Robot physics validation

| scenario | result | checks |
|---|---|---|
| camera_height | PASS | PASS camera height from floor depth (0.1500 vs 0.15); PASS robot rests on the floor (-0.0) |
| straight | PASS | PASS steady speed (0.2986 vs 0.3); PASS acceleration limit (peak 0.511 vs limit 0.5); PASS lateral drift (-0.0003 m); PASS no collisions ([]); PASS not stuck () |
| spin | PASS | PASS yaw rate (1.4966 vs 1.5); PASS turns in place (centre moved 0.001 m) |
| arc | PASS | PASS radius (0.3992 vs 0.4000) |
| wall | PASS | PASS camera mount (depth to wall) (0.8601 vs 0.8600); PASS collision detected (events=1); PASS no penetration (front edge 1.0000 vs wall 1.0); PASS stuck detected against the wall (stuck 7.1 s) |
| steps | PASS | PASS navmesh climb <= physical climb (navmesh 0.01 vs measured 0.01); PASS climbing is monotone in height () |
| clearance | PASS | PASS passes iff slab is above the robot top (top 0.165 m); PASS navmesh height >= robot top (0.17 vs 0.165) |
| push | PASS | PASS pushes a load needing less force than the motors give (); PASS blocked by a load needing more force () |
| determinism | PASS | PASS bit-identical repeat (max diff 0.000e+00) |

## camera_height

```
estimated_camera_height_m: 0.15
configured_m: 0.15
reference_height_m: -0.0
```

## straight

```
steady_speed_mps: 0.2986
cmd_speed_mps: 0.3
time_to_95pct_s: 0.6
ideal_time_to_95pct_s: 0.57
peak_accel_mps2: 0.511
accel_limit_mps2: 0.5
distance_m: 1.715
lateral_drift_m: -0.0003
heading_drift_deg: 0.011
rtf: 255.7
collision_events: 0
collision_links: []
stuck_detected: False
stuck_time_s: 0.0
max_tilt_deg: 0.04
```

## spin

```
yaw_rate_radps: 1.4966
cmd_yaw_rate_radps: 1.5
center_drift_m: 0.0006
collision_events: 0
collision_links: []
stuck_detected: False
stuck_time_s: 0.0
max_tilt_deg: 0.04
```

## arc

```
radius_m: 0.3992
expected_radius_m: 0.4
center: [-0.399, -0.02]
```

## wall

```
camera_depth_to_wall_m: 0.8601
expected_depth_m: 0.86
front_edge_y_m: 1.0
wall_face_y_m: 1.0
first_contact_s: 3.1000000000000014
collision_events: 1
collision_links: ['base_link']
stuck_detected: True
stuck_time_s: 7.1
max_tilt_deg: 0.06
```

## steps

| height_m | climbed | final_forward_m | final_height_m | collision_events | collision_links | stuck_detected | stuck_time_s | max_tilt_deg |
|---|---|---|---|---|---|---|---|---|
| 0.005 | True | 2.328 | 0.0047 | 0 | [] | False | 0.0 | 3.13 |
| 0.01 | True | 2.312 | 0.0097 | 0 | [] | False | 0.0 | 4.44 |
| 0.0125 | False | 0.392 | 0.0004 | 1 | ['front_caster'] | True | 6.5 | 5.41 |
| 0.015 | False | 0.362 | 0.0001 | 1 | ['front_caster'] | True | 6.6 | 5.82 |
| 0.0175 | False | 0.359 | 0.0001 | 1 | ['front_caster'] | True | 6.7 | 5.95 |
| 0.02 | False | 0.35 | -0.0 | 1 | ['base_link', 'front_caster'] | True | 6.7 | 0.59 |
| 0.025 | False | 0.33 | -0.0 | 1 | ['base_link'] | True | 6.8 | 0.04 |
| 0.03 | False | 0.33 | -0.0 | 1 | ['base_link'] | True | 6.8 | 0.04 |
| 0.04 | False | 0.33 | -0.0 | 1 | ['base_link'] | True | 6.8 | 0.03 |

```
max_climbed_m: 0.01
navmesh_max_climb_m: 0.01
```

## clearance

| bottom_m | passed | collision_events | collision_links | stuck_detected | stuck_time_s | max_tilt_deg |
|---|---|---|---|---|---|---|
| 0.12 | False | 1 | ['base_link'] | True | 6.4 | 0.03 |
| 0.14 | False | 1 | ['base_link'] | True | 6.3 | 5.33 |
| 0.155 | False | 1 | ['base_link'] | True | 6.3 | 5.33 |
| 0.16 | False | 1 | ['base_link'] | True | 6.3 | 5.33 |
| 0.17 | True | 0 | [] | False | 0.0 | 0.04 |
| 0.18 | True | 0 | [] | False | 0.0 | 0.04 |
| 0.2 | True | 0 | [] | False | 0.0 | 0.04 |

```
robot_top_m: 0.16499999999999998
navmesh_agent_height_m: 0.17
min_passed_bottom_m: 0.17
```

## push

| mass_kg | box_moved_m | box_friction_force_n | collision_events | collision_links | stuck_detected | stuck_time_s | max_tilt_deg |
|---|---|---|---|---|---|---|---|
| 0.5 | 1.96 | 2.45 | 1 | ['base_link'] | False | 0.0 | 0.03 |
| 2.0 | 1.878 | 9.8 | 1 | ['base_link'] | False | 0.0 | 1.22 |
| 5.0 | 0.085 | 24.5 | 1 | ['base_link'] | True | 6.3 | 0.46 |
| 50.0 | 0.0 | 245.0 | 1 | ['base_link'] | True | 6.8 | 1.11 |

```
motor_force_limit_n: 22.86
```

## determinism

```
max_abs_diff: 0.0
final: (-0.3453458845615387, 0.8300167322158813, 1.8677933467168202, -0.3453458845615387, -2.5756657123565674e-05, -0.8300167322158813)
```
