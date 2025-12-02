"""
Generate path-planning MILP instances (LP + scenario JSON) for imitation learning.

- Uses the random scenario generator in `generate_milp_instance_path_planning.py`.
- Builds the MILP model with the same formulation as `true_receding_horizon.py`
  (assemble_milp_data), then writes it to `.lp` for later use with a branching
  dataset pipeline (Learn2Branch-style).
- Outputs both the scenario JSON and the LP file.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Tuple

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
from generate_milp_instance_path_planning import generate_single_scenario, parse_range  # type: ignore


def build_model_from_data(data: dict) -> Tuple[Model, dict]:
    """Create a SCIP model from `assemble_milp_data` output (no optimize)."""
    model = Model("MILP_instance")
    model.setParam("display/verblevel", 0)

    idx = data["idx"]
    nvar = idx["nvar"]
    A = data["A"]
    B = data["B"]
    n_vehicles = idx["nVehicles"]
    horizon = idx["T"]

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

    # Initial
    start = data.get("start", None)
    if start is not None:
        for veh in range(n_vehicles):
            model.addCons(vars_dict[idx["px"][veh][0]] == start[veh, 0], name=f"init_px_v{veh}")
            model.addCons(vars_dict[idx["py"][veh][0]] == start[veh, 1], name=f"init_py_v{veh}")
            model.addCons(vars_dict[idx["vx"][veh][0]] == start[veh, 2], name=f"init_vx_v{veh}")
            model.addCons(vars_dict[idx["vy"][veh][0]] == start[veh, 3], name=f"init_vy_v{veh}")

    # Terminal
    goal = data.get("goal", None)
    if goal is not None:
        for veh in range(n_vehicles):
            model.addCons(vars_dict[idx["px"][veh][horizon]] == goal[veh, 0], name=f"term_px_v{veh}")
            model.addCons(vars_dict[idx["py"][veh][horizon]] == goal[veh, 1], name=f"term_py_v{veh}")
            model.addCons(vars_dict[idx["vx"][veh][horizon]] == goal[veh, 2], name=f"term_vx_v{veh}")
            model.addCons(vars_dict[idx["vy"][veh][horizon]] == goal[veh, 3], name=f"term_vy_v{veh}")

    # Obstacles
    obstacles = data.get("obstacles", None)
    if obstacles is not None and len(obstacles) > 0:
        buffer = data.get("obstacleBuffer", 0.25)
        Mx = 800
        My = 800
        n_obs = len(obstacles)
        for veh in range(n_vehicles):
            for k in range(horizon):
                for obs in range(n_obs):
                    rect = obstacles[obs, :]
                    bins = idx["aCube"][veh, :, obs, k]
                    xmin = rect[0] - buffer
                    xmax = rect[1] + buffer
                    ymin = rect[2] - buffer
                    ymax = rect[3] + buffer
                    eps = 1e-4
                    model.addCons(
                        vars_dict[idx["px"][veh][k + 1]] <= xmin - eps + Mx * vars_dict[bins[0]],
                        name=f"obs_px_left_v{veh}_{k}_{obs}",
                    )
                    model.addCons(
                        vars_dict[idx["px"][veh][k + 1]] >= xmax + eps - Mx * vars_dict[bins[1]],
                        name=f"obs_px_right_v{veh}_{k}_{obs}",
                    )
                    model.addCons(
                        vars_dict[idx["py"][veh][k + 1]] <= ymin - eps + My * vars_dict[bins[2]],
                        name=f"obs_py_bottom_v{veh}_{k}_{obs}",
                    )
                    model.addCons(
                        vars_dict[idx["py"][veh][k + 1]] >= ymax + eps - My * vars_dict[bins[3]],
                        name=f"obs_py_top_v{veh}_{k}_{obs}",
                    )
                    model.addCons(
                        vars_dict[bins[0]]
                        + vars_dict[bins[1]]
                        + vars_dict[bins[2]]
                        + vars_dict[bins[3]]
                        <= 3,
                        name=f"obs_bin_v{veh}_{k}_{obs}",
                    )

    # Separation
    safe_sep = float(data.get("safeSeparation", 0.0))
    vehicle_pairs = data.get("vehicle_pairs", [])
    pair_cube = idx.get("pairCube", np.zeros((0, 4, horizon), dtype=int))
    Msep = float(data.get("separationBigM", 800.0))
    if (
        data.get("enforceSeparation", False)
        and safe_sep > 0
        and len(vehicle_pairs) > 0
        and pair_cube.size > 0
    ):
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


def save_instance(out_dir: Path, idx: int, scenario: dict, data: dict):
    """Write scenario JSON and LP model to disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"instance_{idx}.json"
    lp_path = out_dir / f"instance_{idx}.lp"

    # Scenario JSON (convert arrays)
    scn_to_write = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in scenario.items()}
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(scn_to_write, fh, indent=2)

    # LP
    model, _ = build_model_from_data(data)
    try:
        model.writeProblem(str(lp_path))
    except Exception as exc:  # pragma: no cover
        model.freeProb()
        raise RuntimeError(
            f"Failed to write LP to {lp_path}. On Windows, SCIP can fail on non-ASCII paths; "
            f"try an ASCII-only output directory (e.g., C:\\\\path_planning_instances)."
        ) from exc
    model.freeProb()


def generate_split(split_name: str,
                   count: int,
                   difficulties: list,
                   n_obs_range: tuple,
                   seed: int,
                   out_dir: Path):
    rng = np.random.RandomState(seed)
    base_dir = out_dir / split_name
    base_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    attempts = 0
    while generated < count and attempts < count * 50:
        attempts += 1
        difficulty = rng.choice(difficulties).lower()
        try:
            scenario = generate_single_scenario(rng, difficulty, n_obs_range)
            # Build MILP data
            local_scn = scenario.copy()
            # Thor -> horizonSteps needed by assemble_milp_data
            horizon_steps = int(round(scenario["Thor"] / scenario["dt"]))
            local_scn["horizonSteps"] = horizon_steps
            data = trh.assemble_milp_data(local_scn)
            save_instance(base_dir, generated + 1, scenario, data)
            generated += 1
        except RuntimeError as exc:
            print(f"[error] {exc}")
            return
        except Exception:
            # skip and retry
            continue
    if generated < count:
        print(f"[warn] split {split_name}: generated {generated}/{count} after {attempts} attempts.")
    else:
        print(f"[ok] split {split_name}: generated {generated} instances.")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate LP instances for path planning MILP.")
    parser.add_argument("--train", type=int, default=50, help="Number of train instances.")
    parser.add_argument("--valid", type=int, default=10, help="Number of validation instances.")
    parser.add_argument("--test", type=int, default=10, help="Number of test instances.")
    parser.add_argument(
        "--difficulties", type=str, default="easy,medium,hard", help="Comma-separated difficulties."
    )
    parser.add_argument("--n-obstacles", type=str, default="6-12", help="Range for obstacle count, e.g., 6-12.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/instances/path_planning",
        help="Output directory root.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    n_obs_range = parse_range(args.n_obstacles)
    difficulties = [d.strip().lower() for d in args.difficulties.split(",") if d.strip()]
    out_dir = Path(args.out_dir)

    generate_split("train", args.train, difficulties, n_obs_range, args.seed, out_dir)
    generate_split("valid", args.valid, difficulties, n_obs_range, args.seed + 1, out_dir)
    generate_split("test", args.test, difficulties, n_obs_range, args.seed + 2, out_dir)


if __name__ == "__main__":
    main()
