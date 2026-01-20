"""
Generate path-planning MILP instances (LP + scenario JSON) for imitation learning.

This unified script:
1. Generates random multi-vehicle path planning scenarios
2. Builds MILP models using the formulation from true_receding_horizon.py
3. Exports both scenario JSON and LP files for Ecole/branching dataset pipelines

Based on:
- Schouwenaars et al. (2001): MILP formulation for multi-vehicle path planning
- Gasse et al. (2019): Learn2Branch imitation learning pipeline

Map bounds: x in [-10, 22], y in [-13, 19]
Vehicles: 4-6 per instance
Obstacles: 5-10 axis-aligned rectangles (2.5-5.0m size)
Thor (planning horizon): sampled from {3.5, 4.0, 4.5, 5.0} seconds

IMPORTANT: Instance Generation Modes

1. Original mode (default, --use-receding-horizon=False):
   - Each scenario generates 1 instance (first iteration only)
   - Fast but may not capture receding horizon diversity
   - Good for initial experiments

2. Receding horizon mode (--use-receding-horizon=True):
   - Simulates receding horizon and saves instances from EACH iteration
   - State rollout can be fast (heuristic/random) or MILP-solved
   - One scenario → 50-200 instances (depending on iterations)
   - Captures real receding horizon MILP diversity
   - Recommended for training branching models that will be used in receding horizon

Example usage:
  # Original mode (1 instance per scenario)
  python generate_instances_path_planning.py --train 1000

  # Receding horizon mode (many instances per scenario)
  python generate_instances_path_planning.py --train 1000 --use-receding-horizon \
      --rollout-mode heuristic
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from pyscipopt import Model, quicksum

# Ensure imports work relative to repo root
ROOT = Path(__file__).resolve().parents[1]
MILP_PYTHON = ROOT / "MILP_Python"
CO_PATH = ROOT / "CO_to_Path_MILP"
if str(MILP_PYTHON) not in sys.path:
    sys.path.append(str(MILP_PYTHON))
if str(CO_PATH) not in sys.path:
    sys.path.append(str(CO_PATH))

import true_receding_horizon as trh  # type: ignore


# ==============================================================================
# CONSTANTS
# ==============================================================================

# Fixed map bounds (0.75x of 2x area, centered, integer values)
MAP_X = (-10.0, 22.0)  # Width: 32.0 (0.75 × 42.0), center: 6.0
MAP_Y = (-13.0, 19.0)  # Height: 32.0 (0.75 × 42.0), center: 3.0
DEFAULT_OBSTACLE_GAP = 0.50

# Default dynamics/bounds (drone-like, holonomic)
DEFAULT_DT = 0.2
DEFAULT_VEL_BOUNDS = np.array([[-22.5, 22.5], [-22.5, 22.5]])
DEFAULT_INPUT_BOUNDS = np.array([[-9.0, 9.0], [-9.0, 9.0]])
# Allow positions to drift a bit outside the main map to reduce infeasibility
DEFAULT_POS_BOUNDS = np.array([
    [MAP_X[0] - 5.0, MAP_X[1] + 5.0],
    [MAP_Y[0] - 5.0, MAP_Y[1] + 5.0]
])

# Horizon choices (seconds)
THOR_CHOICES = [4.0, 5.0, 6.0]


# ==============================================================================
# SCENARIO GENERATION UTILITIES
# ==============================================================================

def sample_rect(
    rng: np.random.RandomState,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    min_w: float,
    max_w: float,
    min_h: float,
    max_h: float,
) -> Tuple[float, float, float, float]:
    """Sample an integer-aligned rectangle [xmin, xmax, ymin, ymax] inside bounds."""
    w_lo = max(1, math.ceil(min_w))
    w_hi = max(w_lo, math.floor(max_w))
    h_lo = max(1, math.ceil(min_h))
    h_hi = max(h_lo, math.floor(max_h))

    w = rng.randint(w_lo, w_hi + 1)
    h = rng.randint(h_lo, h_hi + 1)
    x_low_int = int(math.floor(x_bounds[0]))
    x_high_int = int(math.floor(x_bounds[1] - w))
    y_low_int = int(math.floor(y_bounds[0]))
    y_high_int = int(math.floor(y_bounds[1] - h))
    xmin = rng.randint(x_low_int, x_high_int + 1)
    ymin = rng.randint(y_low_int, y_high_int + 1)
    xmax = xmin + w
    ymax = ymin + h
    return float(xmin), float(xmax), float(ymin), float(ymax)


def point_in_rect(pt: np.ndarray, rect: np.ndarray, buffer: float = 0.0) -> bool:
    """Return True if point lies inside rectangle (with optional buffer)."""
    xmin, xmax, ymin, ymax = rect
    return (xmin - buffer <= pt[0] <= xmax + buffer) and (ymin - buffer <= pt[1] <= ymax + buffer)


def rects_overlap(rect_a: Tuple[float, float, float, float],
                  rect_b: Tuple[float, float, float, float],
                  min_gap: float = 0.0) -> bool:
    """Return True if rectangles overlap or are closer than min_gap."""
    axmin, axmax, aymin, aymax = rect_a
    bxmin, bxmax, bymin, bymax = rect_b
    gap = max(float(min_gap), 0.0) / 2.0
    axmin -= gap
    axmax += gap
    aymin -= gap
    aymax += gap
    bxmin -= gap
    bxmax += gap
    bymin -= gap
    bymax += gap
    overlap_x = (axmin < bxmax) and (axmax > bxmin)
    overlap_y = (aymin < bymax) and (aymax > bymin)
    return overlap_x and overlap_y


def sample_points_avoiding_obstacles(
    rng: np.random.RandomState,
    n: int,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    obstacles: np.ndarray,
    safe_sep: float,
    max_trials: int = 400,
) -> np.ndarray:
    """Sample n integer grid points in bounds, avoiding obstacles and each other."""
    pts: List[np.ndarray] = []
    x_lo = int(math.floor(x_bounds[0]))
    x_hi = int(math.floor(x_bounds[1]))
    y_lo = int(math.floor(y_bounds[0]))
    y_hi = int(math.floor(y_bounds[1]))
    for _ in range(max_trials):
        x = rng.randint(x_lo, x_hi + 1)
        y = rng.randint(y_lo, y_hi + 1)
        candidate = np.array([float(x), float(y)])
        if obstacles.size > 0:
            inside_any = any(point_in_rect(candidate, rect) for rect in obstacles)
            if inside_any:
                continue
        too_close = any(np.linalg.norm(candidate - p) < safe_sep for p in pts)
        if too_close:
            continue
        pts.append(candidate)
        if len(pts) == n:
            return np.stack(pts, axis=0)
    raise RuntimeError("Could not sample enough points avoiding obstacles and separation constraints.")


def generate_obstacles(
    rng: np.random.RandomState,
    n_obs_range: Tuple[int, int],
    min_gap: float
) -> np.ndarray:
    """Generate axis-aligned rectangular obstacles."""
    # Fixed obstacle size parameters
    min_w, max_w = 2.5, 5.0
    min_h, max_h = 2.5, 5.0

    n_obs = rng.randint(n_obs_range[0], n_obs_range[1] + 1)
    rects: List[Tuple[float, float, float, float]] = []
    max_trials_per_obs = 200
    for _ in range(n_obs):
        placed = False
        for _ in range(max_trials_per_obs):
            rect = sample_rect(rng, MAP_X, MAP_Y, min_w, max_w, min_h, max_h)
            if all(not rects_overlap(rect, other, min_gap=min_gap) for other in rects):
                rects.append(rect)
                placed = True
                break
        if not placed:
            raise RuntimeError("Could not place non-overlapping obstacles within trial limit.")
    return np.array(rects, dtype=float)


def compute_max_reachable_distance(Thor: float, dt: float, max_accel: float) -> float:
    """Compute maximum distance reachable in Thor seconds from rest.
    
    For a double integrator starting from rest, the maximum distance 
    is achieved with constant maximum acceleration for half the time,
    then constant deceleration. This gives d = a*T^2/4.
    
    Args:
        Thor: Horizon time in seconds
        dt: Time step (not used but kept for consistency)
        max_accel: Maximum acceleration magnitude
    
    Returns:
        Maximum reachable distance
    """
    return max_accel * (Thor ** 2) / 4.0


def compute_separation_big_m(pos_bounds: np.ndarray, safe_sep: float) -> float:
    """Compute a safe Big-M for separation constraints from position bounds."""
    bounds = np.asarray(pos_bounds, dtype=float)
    if bounds.shape != (2, 2):
        raise ValueError("pos_bounds must be a (2, 2) array like [[xmin, xmax], [ymin, ymax]].")
    ranges = bounds[:, 1] - bounds[:, 0]
    max_range = float(np.max(ranges))
    return max_range + float(safe_sep)


def positions_are_valid(
    positions: np.ndarray,
    obstacles: np.ndarray,
    pos_bounds: np.ndarray,
    buffer: float = 0.0,
    safe_sep: float = 0.0,
) -> bool:
    """Return True if positions are within bounds and not inside obstacles."""
    pts = np.asarray(positions, dtype=float)
    bounds = np.asarray(pos_bounds, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return False
    if np.any(pts[:, 0] < bounds[0, 0]) or np.any(pts[:, 0] > bounds[0, 1]):
        return False
    if np.any(pts[:, 1] < bounds[1, 0]) or np.any(pts[:, 1] > bounds[1, 1]):
        return False
    if obstacles is not None and np.asarray(obstacles).size > 0:
        for pt in pts:
            if any(point_in_rect(pt, rect, buffer=buffer) for rect in obstacles):
                return False
    if safe_sep > 0 and pts.shape[0] > 1:
        for i in range(pts.shape[0]):
            for j in range(i + 1, pts.shape[0]):
                if np.linalg.norm(pts[i] - pts[j]) < safe_sep:
                    return False
    return True




def rollout_next_state(
    current_state: np.ndarray,
    goal: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    input_bounds: np.ndarray,
    pos_bounds: np.ndarray,
    obstacles: np.ndarray,
    obstacle_buffer: float,
    rng: np.random.RandomState,
    mode: str = "heuristic",
    noise_scale: float = 0.2,
    max_tries: int = 20,
    safe_sep: float = 0.0,
) -> Optional[np.ndarray]:
    """Update state without solving MILP (fast rollout)."""
    mode = mode.lower()
    input_bounds = np.asarray(input_bounds, dtype=float)
    input_min = input_bounds[:, 0]
    input_max = input_bounds[:, 1]
    current_state = np.asarray(current_state, dtype=float)
    goal = np.asarray(goal, dtype=float)

    for _ in range(max_tries):
        if mode == "random":
            accel = rng.uniform(low=input_min, high=input_max, size=(current_state.shape[0], 2))
        elif mode == "heuristic":
            pos = current_state[:, 0:2]
            vel = current_state[:, 2:4]
            direction = goal[:, 0:2] - pos
            dist = np.linalg.norm(direction, axis=1, keepdims=True)
            unit = direction / np.maximum(dist, 1e-6)
            accel = unit * input_max * 0.8 - 0.2 * vel
            if noise_scale > 0:
                accel += rng.normal(scale=noise_scale, size=accel.shape)
            accel = np.clip(accel, input_min, input_max)
        else:
            raise ValueError(f"Unknown rollout mode: {mode}")

        next_state = current_state @ A.T + accel @ B.T
        if positions_are_valid(
            next_state[:, 0:2],
            obstacles,
            pos_bounds,
            buffer=obstacle_buffer,
            safe_sep=safe_sep,
        ):
            return next_state

    return None


def is_feasible_instance(
    starts: np.ndarray,
    goals: np.ndarray,
    obstacles: np.ndarray,
    safe_sep: float,
    Thor: float = None,
    dt: float = 0.2,
    max_accel: float = 30.0,
    enforce_horizon_reachability: bool = True,
) -> bool:
    """Enhanced feasibility sanity checks.
    
    Checks:
    1. Start/goal not inside obstacles
    2. Sufficient separation between vehicles at start and goal
    3. (Optional) Goal reachable within horizon time
    """
    # Start/goal not inside obstacles (with small buffer for safety)
    buffer = 0.25
    for pt in np.vstack([starts, goals]):
        if any(point_in_rect(pt, rect, buffer=buffer) for rect in obstacles):
            return False
    
    # Separation at start and goal
    if starts.shape[0] > 1:
        for i in range(starts.shape[0]):
            for j in range(i + 1, starts.shape[0]):
                if np.linalg.norm(starts[i] - starts[j]) < safe_sep:
                    return False
                if np.linalg.norm(goals[i] - goals[j]) < safe_sep:
                    return False
    
    # Horizon-distance feasibility check
    if enforce_horizon_reachability and Thor is not None and Thor > 0:
        max_dist = compute_max_reachable_distance(Thor, dt, max_accel)
        for i in range(starts.shape[0]):
            dist_to_goal = np.linalg.norm(goals[i] - starts[i])
            # Allow some margin (0.8) since obstacles may force longer paths
            if dist_to_goal > 0.8 * max_dist:
                return False
    
    return True


def generate_single_scenario(
    rng: np.random.RandomState,
    n_obs_range: Tuple[int, int] = None,
    obstacle_gap: float = DEFAULT_OBSTACLE_GAP,
    use_terminal_cost: bool = True,
    terminal_weights: Optional[Dict[str, float]] = None,
) -> Dict:
    """Generate one scenario dict with all parameters for MILP construction."""
    if terminal_weights is None:
        terminal_weights = {"position": 10.0, "velocity": 5.0}
    
    # Fixed vehicle count: 4-6 vehicles
    n_vehicles = rng.randint(4, 7)

    # Fixed obstacle count: 5-10 obstacles
    if n_obs_range is None:
        n_obs_range = (5, 10)
    
    obstacles = generate_obstacles(rng, n_obs_range, obstacle_gap)

    # Safety parameters by difficulty
    # All difficulties use same separation distance and enforce separation
    safe_sep = 0.1
    enforce_sep = True

    # Start and goal sampling
    starts_xy = sample_points_avoiding_obstacles(
        rng, n_vehicles, MAP_X, MAP_Y, obstacles, safe_sep
    )
    goals_xy = sample_points_avoiding_obstacles(
        rng, n_vehicles, MAP_X, MAP_Y, obstacles, safe_sep
    )
    starts = np.hstack([starts_xy, np.zeros((n_vehicles, 2))])
    goals = np.hstack([goals_xy, np.zeros((n_vehicles, 2))])

    # Thor sampling (sample first for feasibility check)
    Thor = float(rng.choice(THOR_CHOICES))
    max_accel = float(DEFAULT_INPUT_BOUNDS[0, 1])
    separation_big_m = compute_separation_big_m(DEFAULT_POS_BOUNDS, safe_sep)

    # Enhanced feasibility check including horizon-distance reachability
    if not is_feasible_instance(
        starts,
        goals,
        obstacles,
        safe_sep,
        Thor=Thor,
        dt=DEFAULT_DT,
        max_accel=max_accel,
        enforce_horizon_reachability=not use_terminal_cost,
    ):
        raise RuntimeError("Instance failed feasibility checks.")

    scenario = {
        "dt": DEFAULT_DT,
        "start": starts,
        "goal": goals,
        "obstacles": obstacles,
        "posBounds": DEFAULT_POS_BOUNDS,
        "velBounds": DEFAULT_VEL_BOUNDS,
        "inputBounds": DEFAULT_INPUT_BOUNDS,
        "obstacleBuffer": 0.1,
        "obstaclePruneMargin": 2.0,
        "obstacleGap": float(obstacle_gap),
        "safeSeparation": safe_sep,
        "enforceSeparation": enforce_sep,
        "pairPruneMargin": 0.6,
        "separationBigM": separation_big_m,
        "nVehicles": int(n_vehicles),
        "Thor": Thor,
        # Terminal cost settings (Schouwenaars et al. 2001, Eq. 4)
        "useTerminalCost": bool(use_terminal_cost),
        "terminalWeights": {
            "position": float(terminal_weights.get("position", 10.0)),
            "velocity": float(terminal_weights.get("velocity", 5.0)),
        },
    }
    return scenario


# ==============================================================================
# MILP MODEL BUILDING
# ==============================================================================

def build_model_from_data(data: dict) -> Tuple[Model, dict]:
    """Create a SCIP model from `assemble_milp_data` output (no optimize).
    
    This function mirrors the constraint structure in true_receding_horizon.py's solve_milp.
    Key features aligned with Schouwenaars et al. (2001):
    - Terminal COST (soft constraint) instead of terminal equality (hard constraint)
    - Obstacle constraints cover k=0 to T (including initial state)
    """
    model = Model("MILP_instance")
    model.setParam("display/verblevel", 0)

    idx = data["idx"]
    nvar = idx["nvar"]
    A = data["A"]
    B = data["B"]
    n_vehicles = idx["nVehicles"]
    horizon = idx["T"]
    use_terminal_cost = data.get("useTerminalCost", True)

    vars_dict = {}
    intcon = set(idx["intcon"])
    for i in range(nvar):
        vtype = "B" if i in intcon else "C"
        lb_val = data["lb"].get(i, None)
        ub_val = data["ub"].get(i, None)
        if lb_val is None:
            lb_val = -model.infinity()
        if ub_val is None:
            ub_val = model.infinity()
        vars_dict[i] = model.addVar(name=f"x{i}", vtype=vtype, lb=lb_val, ub=ub_val)

    # Objective
    obj_expr = quicksum(data["f_cost"].get(i, 0.0) * vars_dict[i] for i in range(nvar))
    model.setObjective(obj_expr, "minimize")

    # Fuel cost (|u| <= s)
    for veh in range(n_vehicles):
        for k in range(horizon):
            ux_idx = idx["ux"][veh][k]
            uy_idx = idx["uy"][veh][k]
            sx_idx = idx["sx"][veh][k]
            sy_idx = idx["sy"][veh][k]
            model.addCons(vars_dict[ux_idx] <= vars_dict[sx_idx], name=f"fuel_ux1_v{veh}_{k}")
            model.addCons(-vars_dict[ux_idx] <= vars_dict[sx_idx], name=f"fuel_ux2_v{veh}_{k}")
            model.addCons(vars_dict[uy_idx] <= vars_dict[sy_idx], name=f"fuel_uy1_v{veh}_{k}")
            model.addCons(-vars_dict[uy_idx] <= vars_dict[sy_idx], name=f"fuel_uy2_v{veh}_{k}")

    # Dynamics
    for veh in range(n_vehicles):
        for k in range(horizon):
            model.addCons(
                vars_dict[idx["px"][veh][k + 1]]
                == A[0, 0] * vars_dict[idx["px"][veh][k]]
                + A[0, 2] * vars_dict[idx["vx"][veh][k]]
                + B[0, 0] * vars_dict[idx["ux"][veh][k]],
                name=f"dyn_px_v{veh}_{k}",
            )
            model.addCons(
                vars_dict[idx["py"][veh][k + 1]]
                == A[1, 1] * vars_dict[idx["py"][veh][k]]
                + A[1, 3] * vars_dict[idx["vy"][veh][k]]
                + B[1, 1] * vars_dict[idx["uy"][veh][k]],
                name=f"dyn_py_v{veh}_{k}",
            )
            model.addCons(
                vars_dict[idx["vx"][veh][k + 1]]
                == A[2, 2] * vars_dict[idx["vx"][veh][k]]
                + B[2, 0] * vars_dict[idx["ux"][veh][k]],
                name=f"dyn_vx_v{veh}_{k}",
            )
            model.addCons(
                vars_dict[idx["vy"][veh][k + 1]]
                == A[3, 3] * vars_dict[idx["vy"][veh][k]]
                + B[3, 1] * vars_dict[idx["uy"][veh][k]],
                name=f"dyn_vy_v{veh}_{k}",
            )

    # Initial state constraints
    start = data.get("start", None)
    if start is not None:
        for veh in range(n_vehicles):
            model.addCons(vars_dict[idx["px"][veh][0]] == start[veh, 0], name=f"init_px_v{veh}")
            model.addCons(vars_dict[idx["py"][veh][0]] == start[veh, 1], name=f"init_py_v{veh}")
            model.addCons(vars_dict[idx["vx"][veh][0]] == start[veh, 2], name=f"init_vx_v{veh}")
            model.addCons(vars_dict[idx["vy"][veh][0]] == start[veh, 3], name=f"init_vy_v{veh}")

    # Terminal state handling: cost (soft) vs constraint (hard)
    goal = data.get("goal", None)
    if goal is not None:
        if use_terminal_cost and idx.get("useTerminalCost", False):
            # Soft terminal constraint via cost function (prevents infeasibility)
            for veh in range(n_vehicles):
                model.addCons(
                    vars_dict[idx["px"][veh][horizon]] - goal[veh, 0] <= vars_dict[idx["tpx"][veh]],
                    name=f"term_px_pos_v{veh}",
                )
                model.addCons(
                    goal[veh, 0] - vars_dict[idx["px"][veh][horizon]] <= vars_dict[idx["tpx"][veh]],
                    name=f"term_px_neg_v{veh}",
                )
                model.addCons(
                    vars_dict[idx["py"][veh][horizon]] - goal[veh, 1] <= vars_dict[idx["tpy"][veh]],
                    name=f"term_py_pos_v{veh}",
                )
                model.addCons(
                    goal[veh, 1] - vars_dict[idx["py"][veh][horizon]] <= vars_dict[idx["tpy"][veh]],
                    name=f"term_py_neg_v{veh}",
                )
                model.addCons(
                    vars_dict[idx["vx"][veh][horizon]] - goal[veh, 2] <= vars_dict[idx["tvx"][veh]],
                    name=f"term_vx_pos_v{veh}",
                )
                model.addCons(
                    goal[veh, 2] - vars_dict[idx["vx"][veh][horizon]] <= vars_dict[idx["tvx"][veh]],
                    name=f"term_vx_neg_v{veh}",
                )
                model.addCons(
                    vars_dict[idx["vy"][veh][horizon]] - goal[veh, 3] <= vars_dict[idx["tvy"][veh]],
                    name=f"term_vy_pos_v{veh}",
                )
                model.addCons(
                    goal[veh, 3] - vars_dict[idx["vy"][veh][horizon]] <= vars_dict[idx["tvy"][veh]],
                    name=f"term_vy_neg_v{veh}",
                )
        else:
            # Hard terminal equality constraint (may cause infeasibility)
            for veh in range(n_vehicles):
                model.addCons(vars_dict[idx["px"][veh][horizon]] == goal[veh, 0], name=f"term_px_v{veh}")
                model.addCons(vars_dict[idx["py"][veh][horizon]] == goal[veh, 1], name=f"term_py_v{veh}")
                model.addCons(vars_dict[idx["vx"][veh][horizon]] == goal[veh, 2], name=f"term_vx_v{veh}")
                model.addCons(vars_dict[idx["vy"][veh][horizon]] == goal[veh, 3], name=f"term_vy_v{veh}")

    # Obstacle constraints - k=0 to T (all positions including initial state)
    obstacles = data.get("obstacles", None)
    if obstacles is not None and len(obstacles) > 0:
        buffer = data.get("obstacleBuffer", 0.25)
        Mx = 800
        My = 800
        n_obs = len(obstacles)
        for veh in range(n_vehicles):
            for k in range(horizon + 1):
                for obs in range(n_obs):
                    rect = obstacles[obs, :]
                    bins = idx["aCube"][veh, :, obs, k]
                    xmin = rect[0] - buffer
                    xmax = rect[1] + buffer
                    ymin = rect[2] - buffer
                    ymax = rect[3] + buffer
                    eps = 1e-4
                    model.addCons(
                        vars_dict[idx["px"][veh][k]] <= xmin - eps + Mx * vars_dict[bins[0]],
                        name=f"obs_px_left_v{veh}_{k}_{obs}",
                    )
                    model.addCons(
                        vars_dict[idx["px"][veh][k]] >= xmax + eps - Mx * vars_dict[bins[1]],
                        name=f"obs_px_right_v{veh}_{k}_{obs}",
                    )
                    model.addCons(
                        vars_dict[idx["py"][veh][k]] <= ymin - eps + My * vars_dict[bins[2]],
                        name=f"obs_py_bottom_v{veh}_{k}_{obs}",
                    )
                    model.addCons(
                        vars_dict[idx["py"][veh][k]] >= ymax + eps - My * vars_dict[bins[3]],
                        name=f"obs_py_top_v{veh}_{k}_{obs}",
                    )
                    model.addCons(
                        vars_dict[bins[0]] + vars_dict[bins[1]] + vars_dict[bins[2]] + vars_dict[bins[3]] <= 3,
                        name=f"obs_bin_v{veh}_{k}_{obs}",
                    )

    # Separation constraints
    safe_sep = float(data.get("safeSeparation", 0.0))
    vehicle_pairs = data.get("vehicle_pairs", [])
    pair_cube = idx.get("pairCube", np.zeros((0, 4, horizon), dtype=int))
    Msep = float(data.get("separationBigM", 800.0))
    if data.get("enforceSeparation", False) and safe_sep > 0 and len(vehicle_pairs) > 0 and pair_cube.size > 0:
        for p_idx, (veh_i, veh_j) in enumerate(vehicle_pairs):
            for k in range(horizon):
                bins = pair_cube[p_idx, :, k]
                px_i = idx["px"][veh_i][k + 1]
                px_j = idx["px"][veh_j][k + 1]
                py_i = idx["py"][veh_i][k + 1]
                py_j = idx["py"][veh_j][k + 1]
                model.addCons(
                    vars_dict[px_i] <= vars_dict[px_j] - safe_sep + Msep * vars_dict[bins[0]],
                    name=f"sep_px_left_p{p_idx}_{k}",
                )
                model.addCons(
                    vars_dict[px_i] >= vars_dict[px_j] + safe_sep - Msep * vars_dict[bins[1]],
                    name=f"sep_px_right_p{p_idx}_{k}",
                )
                model.addCons(
                    vars_dict[py_i] <= vars_dict[py_j] - safe_sep + Msep * vars_dict[bins[2]],
                    name=f"sep_py_bottom_p{p_idx}_{k}",
                )
                model.addCons(
                    vars_dict[py_i] >= vars_dict[py_j] + safe_sep - Msep * vars_dict[bins[3]],
                    name=f"sep_py_top_p{p_idx}_{k}",
                )
                model.addCons(
                    vars_dict[bins[0]] + vars_dict[bins[1]] + vars_dict[bins[2]] + vars_dict[bins[3]] <= 3,
                    name=f"sep_bin_p{p_idx}_{k}",
                )

    return model, vars_dict


# ==============================================================================
# INSTANCE SAVING AND DATASET GENERATION
# ==============================================================================

def save_instance_with_feasibility_check(
    out_dir: Path,
    idx: int,
    scenario: dict,
    data: dict,
    feas_time_limit: float = 0.0,
) -> bool:
    """Build model ONCE, check feasibility, and write to disk if valid.
    
    Args:
        out_dir: Output directory
        idx: Instance index
        scenario: Scenario dictionary
        data: MILP data from assemble_milp_data
        feas_time_limit: Time limit for feasibility check (0 = skip check)
    
    Returns:
        True if instance was saved (feasible), False otherwise.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"instance_{idx}.json"
    lp_path = out_dir / f"instance_{idx}.lp"

    # Build model ONCE
    model, _ = build_model_from_data(data)
    
    # Feasibility check (if enabled)
    if feas_time_limit > 0:
        model.setParam("display/verblevel", 0)
        model.setParam("limits/time", float(feas_time_limit))
        model.setParam("limits/solutions", 1)
        model.optimize()
        if model.getNSols() == 0:
            model.freeProb()
            return False  # Infeasible - don't save
    
    # Write scenario JSON
    scn_to_write = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in scenario.items()}
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(scn_to_write, fh, indent=2)

    # Write LP file (reusing same model)
    try:
        model.writeProblem(str(lp_path))
    except Exception as exc:
        model.freeProb()
        raise RuntimeError(
            f"Failed to write LP to {lp_path}. On Windows, SCIP can fail on non-ASCII paths; "
            f"try an ASCII-only output directory."
        ) from exc
    model.freeProb()
    return True


def count_existing_instances(directory: Path) -> int:
    """Count existing instance LP files in directory."""
    if not directory.exists():
        return 0
    return len(list(directory.glob("instance_*.lp")))


def extract_instances_from_receding_horizon(
    scenario: dict,
    Thor: float,
    max_iterations: int = 200,
    rollout_mode: str = "heuristic",
    rng: Optional[np.random.RandomState] = None,
    rollout_noise: float = 0.2,
    rollout_max_tries: int = 20,
) -> List[Tuple[dict, dict, int]]:
    """Extract MILP instances from receding horizon iterations.
    
    Simulates receding horizon and collects MILP data from each iteration.
    
    Args:
        scenario: Base scenario dictionary
        Thor: Horizon time in seconds
        max_iterations: Maximum number of iterations
        rollout_mode: How to update state between iterations (solve/heuristic/random)
        rng: Random number generator (for rollout noise and save offset)
        rollout_noise: Noise scale for heuristic rollout inputs
        rollout_max_tries: Max attempts to find a valid rollout step
    
    Returns:
        List of (local_scenario, milp_data, iteration) tuples, one per saved iteration
    """
    import time
    
    dt = scenario['dt']
    horizon_steps = int(round(Thor / dt))
    
    start = np.atleast_2d(np.asarray(scenario['start'], dtype=float))
    goal = np.atleast_2d(np.asarray(scenario['goal'], dtype=float))
    if start.shape[0] != goal.shape[0]:
        raise ValueError('Start and goal must contain the same number of vehicles.')
    n_vehicles = start.shape[0]
    current_state = start.copy()
    
    pos_tolerance = 0.1
    vel_tolerance = 0.1
    A, B = trh.double_integrator_matrices(dt)
    rollout_mode = rollout_mode.lower()
    rng = rng or np.random.RandomState(0)
    input_bounds = np.asarray(scenario.get("inputBounds", DEFAULT_INPUT_BOUNDS), dtype=float)
    pos_bounds = np.asarray(scenario.get("posBounds", DEFAULT_POS_BOUNDS), dtype=float)
    obstacles = np.asarray(scenario.get("obstacles", []), dtype=float)
    obstacle_buffer = float(scenario.get("obstacleBuffer", 0.1))
    safe_sep = float(scenario.get("safeSeparation", 0.0))
    
    instances = []
    iterations = 0
    scenario_start_time = time.time()
    
    while iterations < max_iterations:
        # Check if target is reached
        pos_error = np.linalg.norm(current_state[:, 0:2] - goal[:, 0:2], axis=1)
        vel_error = np.linalg.norm(current_state[:, 2:4] - goal[:, 2:4], axis=1)
        
        if np.all(pos_error < pos_tolerance) and np.all(vel_error < vel_tolerance):
            break
        
        try:
            local_scenario = None
            milp_data = None

            local_scenario = scenario.copy()
            local_scenario['start'] = current_state.copy()
            local_scenario['goal'] = goal.copy()
            local_scenario['horizonSteps'] = horizon_steps
            local_scenario['nVehicles'] = n_vehicles
            prune_margin = scenario.get('obstaclePruneMargin', 2.0)
            local_scenario['obstacles'] = trh.filter_active_obstacles(
                current_state, goal, scenario.get('obstacles', []), prune_margin)

            if rollout_mode == "solve":
                milp_data = trh.assemble_milp_data(local_scenario)

                # Save this iteration
                instances.append((local_scenario, milp_data, iterations))

                # Solve MILP to get optimal input for state update (with timeout)
                result = trh.solve_milp(milp_data, time_limit=None)

                if result is None or result.get('status') not in ['optimal']:
                    # MILP infeasible, timeout, or error - stop this scenario
                    break

                # Extract first input from solution and update state
                u_sol = result['input']

                # Apply first input to update state using dynamics
                next_state = np.zeros_like(current_state)
                for veh in range(n_vehicles):
                    u_first = u_sol[veh, :, 0]  # First input for this vehicle [ux, uy]
                    next_state[veh, :] = A @ current_state[veh, :] + B @ u_first

                current_state = next_state
            else:
                milp_data = trh.assemble_milp_data(local_scenario)
                instances.append((local_scenario, milp_data, iterations))

                next_state = rollout_next_state(
                    current_state,
                    goal,
                    A,
                    B,
                    input_bounds,
                    pos_bounds,
                    obstacles,
                    obstacle_buffer,
                    rng,
                    mode=rollout_mode,
                    noise_scale=rollout_noise,
                    max_tries=rollout_max_tries,
                    safe_sep=safe_sep,
                )
                if next_state is None:
                    break
                current_state = next_state

        except Exception:
            # If MILP assembly or rollout fails, stop this scenario
            break
        
        iterations += 1
    
    return instances


def generate_split(
    split_name: str,
    count: int,
    n_obs_range: tuple = None,
    seed: int = 0,
    out_dir: Path = None,
    feas_time_limit: float = 2.0,
    max_attempts_mult: int = 50,
    obstacle_gap: float = DEFAULT_OBSTACLE_GAP,
    use_terminal_cost: bool = True,
    terminal_weights: Optional[Dict[str, float]] = None,
    resume: bool = True,
    use_receding_horizon: bool = False,
    rollout_mode: str = "heuristic",
    rollout_noise: float = 0.2,
    rollout_max_tries: int = 20,
    max_iterations: int = 200,
):
    """Generate instances for a single split (train/valid/test).
    
    If resume=True, continues from existing instances instead of starting over.
    
    Args:
        use_receding_horizon: If True, simulate receding horizon and save instances from each iteration.
                             If False, save only the first iteration (original behavior).
        rollout_mode: How to update RH state (solve/heuristic/random)
        rollout_noise: Noise scale for heuristic rollout
        rollout_max_tries: Max attempts to find a valid rollout step
        max_iterations: Maximum RH iterations per scenario
    """
    base_dir = out_dir / split_name
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file for scenario tracking
    log_file = base_dir / "scenario_log.txt"
    log_fh = open(log_file, "w", encoding="utf-8")
    log_fh.write(f"=== Instance Generation Log for {split_name} ===\n")
    log_fh.write(f"Target: {count} instances\n")
    log_fh.write(f"Receding Horizon: {use_receding_horizon}\n")
    log_fh.write(f"Rollout mode: {rollout_mode}\n")
    if use_receding_horizon:
        log_fh.write(f"Max iterations: {max_iterations}\n")
    log_fh.write("-" * 60 + "\n\n")
    log_fh.flush()

    # Check for existing instances and resume if enabled
    existing_count = count_existing_instances(base_dir) if resume else 0
    if existing_count >= count:
        print(f"  [{split_name}] Already have {existing_count}/{count} instances, skipping.")
        return
    
    if existing_count > 0:
        print(f"  [{split_name}] Found {existing_count} existing instances, resuming from {existing_count + 1}...")
    
    # Initialize RNG - advance state to match where we left off
    # This ensures reproducibility even when resuming
    rng = np.random.RandomState(seed)
    # Skip RNG states for already generated instances (approximate)
    for _ in range(existing_count * 5):  # ~5 RNG calls per instance attempt
        rng.random()
    
    generated = existing_count
    attempts = 0
    remaining = count - existing_count
    max_attempts = remaining * max(1, int(max_attempts_mult))
    
    while generated < count and attempts < max_attempts:
        attempts += 1
        try:
            scenario = generate_single_scenario(
                rng,
                n_obs_range,
                obstacle_gap=obstacle_gap,
                use_terminal_cost=use_terminal_cost,
                terminal_weights=terminal_weights,
            )
            
            if use_receding_horizon:
                # Extract instances from receding horizon iterations
                instances = extract_instances_from_receding_horizon(
                    scenario,
                    scenario["Thor"],
                    max_iterations=max_iterations,
                    rollout_mode=rollout_mode,
                    rng=rng,
                    rollout_noise=rollout_noise,
                    rollout_max_tries=rollout_max_tries,
                )
                
                # Track instances saved from this scenario
                scenario_saved_count = 0
                
                # Save each iteration as a separate instance
                for local_scn, milp_data, iter_idx in instances:
                    # Create enriched scenario with iteration info
                    enriched_scenario = local_scn.copy()
                    enriched_scenario['original_scenario_iteration'] = int(iter_idx)
                    enriched_scenario['original_scenario_id'] = attempts  # Scenario attempt number
                    
                    if not save_instance_with_feasibility_check(
                        base_dir, generated + 1, enriched_scenario, milp_data, feas_time_limit
                    ):
                        continue  # Infeasible or too easy, skip this iteration
                    
                    scenario_saved_count += 1
                    generated += 1
                
                # Log scenario summary
                if scenario_saved_count > 0:
                    print(f"  [{split_name}] Scenario {attempts}: {scenario_saved_count} instance(s)")
                    log_fh.write(f"Scenario {attempts}: {scenario_saved_count} instance(s)\n")
                    log_fh.flush()
            else:
                # Original behavior: save only first iteration
                local_scn = scenario.copy()
                horizon_steps = int(round(scenario["Thor"] / scenario["dt"]))
                local_scn["horizonSteps"] = horizon_steps
                data = trh.assemble_milp_data(local_scn)
                # Build model once, check feasibility, and save if feasible
                if not save_instance_with_feasibility_check(
                    base_dir, generated + 1, scenario, data, feas_time_limit
                ):
                    continue  # Infeasible or too easy, try another scenario
                generated += 1
                
                # Log scenario summary (single instance per scenario)
                print(f"  [{split_name}] Scenario {attempts}: 1 instance")
                log_fh.write(f"Scenario {attempts}: 1 instance\n")
                log_fh.flush()
                    
        except RuntimeError:
            continue
        except Exception:
            continue
    
    if generated < count:
        print(f"[warn] split {split_name}: generated {generated}/{count} after {attempts} attempts.")
        log_fh.write(f"\n[WARNING] Only generated {generated}/{count} instances after {attempts} attempts.\n")
    else:
        print(f"[ok] split {split_name}: generated {generated} instances.")
        log_fh.write(f"\n[SUCCESS] Generated {generated} instances from {attempts} scenarios.\n")
    
    log_fh.write(f"Total scenarios attempted: {attempts}\n")
    log_fh.write(f"Total instances generated: {generated}\n")
    log_fh.close()
    print(f"  [{split_name}] Scenario log saved to: {log_file}")


def parse_range(s: str) -> Tuple[int, int]:
    """Parse a range string like '6-12' into a tuple (6, 12)."""
    parts = s.split("-")
    if len(parts) != 2:
        raise ValueError("Range must be in the form 'a-b'.")
    lo, hi = int(parts[0]), int(parts[1])
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate path-planning MILP instances (LP + JSON) for imitation learning."
    )
    parser.add_argument("--train", type=int, default=5000, help="Number of train instances.")
    parser.add_argument("--valid", type=int, default=1000, help="Number of validation instances.")
    parser.add_argument("--test", type=int, default=1000, help="Number of test instances.")
    parser.add_argument(
        "--n-obstacles", 
        type=str, 
        default=None, 
        help="Range for obstacle count, e.g., 6-12. If not specified, difficulty-based ranges are used."
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument(
        "--feas-time-limit",
        type=float,
        default=0.0,
        help="Feasibility check time limit per instance (seconds). Default 0 (disabled).",
    )
    parser.add_argument(
        "--max-attempts-mult",
        type=int,
        default=100,
        help="Maximum attempts per split as a multiple of target count.",
    )
    parser.add_argument(
        "--obstacle-gap",
        type=float,
        default=DEFAULT_OBSTACLE_GAP,
        help="Minimum gap between obstacles (meters).",
    )
    parser.add_argument(
        "--terminal-mode",
        choices=["soft", "hard"],
        default="soft",
        help="Terminal condition: soft cost or hard equality constraint.",
    )
    parser.add_argument(
        "--terminal-w-pos",
        type=float,
        default=10.0,
        help="Terminal position weight when using soft terminal cost.",
    )
    parser.add_argument(
        "--terminal-w-vel",
        type=float,
        default=5.0,
        help="Terminal velocity weight when using soft terminal cost.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="CO_to_Path_MILP/pp_instances_gasse",
        help="Output directory root.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start from scratch instead of resuming from existing instances.",
    )
    parser.add_argument(
        "--use-receding-horizon",
        dest="use_receding_horizon",
        action="store_true",
        default=True,
        help="Simulate receding horizon and save instances from each iteration (default: True). "
             "This is the recommended approach for training branching models.",
    )
    parser.add_argument(
        "--no-use-receding-horizon",
        dest="use_receding_horizon",
        action="store_false",
        help="Disable receding horizon mode - only save first iteration (legacy behavior).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=200,
        help="Maximum RH iterations per scenario (default: 200).",
    )
    parser.add_argument(
        "--rollout-mode",
        type=str,
        default="heuristic",
        choices=["solve", "heuristic", "random"],
        help="How to update RH state between iterations: solve (MILP), heuristic, or random.",
    )
    parser.add_argument(
        "--rollout-noise",
        type=float,
        default=0.2,
        help="Noise scale for heuristic rollout inputs (default: 0.2).",
    )
    parser.add_argument(
        "--rollout-max-tries",
        type=int,
        default=20,
        help="Max attempts to find a valid rollout step (default: 20).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    n_obs_range = parse_range(args.n_obstacles) if args.n_obstacles else None
    out_dir = Path(args.out_dir)
    use_terminal_cost = args.terminal_mode == "soft"
    terminal_weights = {"position": args.terminal_w_pos, "velocity": args.terminal_w_vel}

    resume = not args.no_resume
    
    print("Starting instance generation...")
    print(f"Generating path-planning MILP instances...")
    print(f"  Output directory: {out_dir}")
    print(f"  Vehicles: 4-6, Obstacles: 5-10")
    print(f"  Train/Valid/Test: {args.train}/{args.valid}/{args.test}")
    print(f"  Resume mode: {resume}")
    print()

    print(f"  Use receding horizon: {args.use_receding_horizon}")
    if args.use_receding_horizon:
        print(f"  Max iterations: {args.max_iterations}")
        print(f"  Rollout mode: {args.rollout_mode}")
    print()

    generate_split(
        "train",
        args.train,
        n_obs_range,
        args.seed,
        out_dir,
        feas_time_limit=args.feas_time_limit,
        max_attempts_mult=args.max_attempts_mult,
        obstacle_gap=args.obstacle_gap,
        use_terminal_cost=use_terminal_cost,
        terminal_weights=terminal_weights,
        resume=resume,
        use_receding_horizon=args.use_receding_horizon,
        rollout_mode=args.rollout_mode,
        rollout_noise=args.rollout_noise,
        rollout_max_tries=args.rollout_max_tries,
        max_iterations=args.max_iterations,
    )
    generate_split(
        "valid",
        args.valid,
        n_obs_range,
        args.seed + 1,
        out_dir,
        feas_time_limit=args.feas_time_limit,
        max_attempts_mult=args.max_attempts_mult,
        obstacle_gap=args.obstacle_gap,
        use_terminal_cost=use_terminal_cost,
        terminal_weights=terminal_weights,
        resume=resume,
        use_receding_horizon=args.use_receding_horizon,
        rollout_mode=args.rollout_mode,
        rollout_noise=args.rollout_noise,
        rollout_max_tries=args.rollout_max_tries,
        max_iterations=args.max_iterations,
    )
    generate_split(
        "test",
        args.test,
        n_obs_range,
        args.seed + 2,
        out_dir,
        feas_time_limit=args.feas_time_limit,
        max_attempts_mult=args.max_attempts_mult,
        obstacle_gap=args.obstacle_gap,
        use_terminal_cost=use_terminal_cost,
        terminal_weights=terminal_weights,
        resume=resume,
        use_receding_horizon=args.use_receding_horizon,
        rollout_mode=args.rollout_mode,
        rollout_noise=args.rollout_noise,
        rollout_max_tries=args.rollout_max_tries,
        max_iterations=args.max_iterations,
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
