"""
true_receding_horizon.py
True Receding Horizon MILP implementation
Applies only the first control input at each iteration and updates the state
Follows the approach from Schouwenaars et al. (2001) paper
Converted from MATLAB to Python using SCIP solver.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyscipopt import Model, quicksum
import time
import os
from itertools import combinations


def double_integrator_matrices(dt):
    """Discretized double integrator with zero-order hold."""
    A = np.eye(4)
    A[0, 2] = dt
    A[1, 3] = dt
    B = np.zeros((4, 2))
    B[0, 0] = 0.5 * dt**2
    B[1, 1] = 0.5 * dt**2
    B[2, 0] = dt
    B[3, 1] = dt
    return A, B


def filter_active_obstacles(state, goal, obstacles, margin=2.0):
    """Keep only obstacles that lie near the current-to-goal corridor."""
    obs_arr = np.asarray(obstacles, dtype=float)
    if obs_arr.size == 0:
        return obs_arr.reshape((-1, 4))
    
    state = np.atleast_2d(np.asarray(state, dtype=float))
    goal = np.atleast_2d(np.asarray(goal, dtype=float))
    margin = max(float(margin), 0.0)
    
    positions = np.vstack([state[:, 0:2], goal[:, 0:2]])
    min_x = np.min(positions[:, 0]) - margin
    max_x = np.max(positions[:, 0]) + margin
    min_y = np.min(positions[:, 1]) - margin
    max_y = np.max(positions[:, 1]) + margin
    
    overlaps_x = (obs_arr[:, 1] >= min_x) & (obs_arr[:, 0] <= max_x)
    overlaps_y = (obs_arr[:, 3] >= min_y) & (obs_arr[:, 2] <= max_y)
    mask = overlaps_x & overlaps_y
    
    if not np.any(mask):
        return np.zeros((0, 4))
    return obs_arr[mask]


def identify_active_pairs(state, goal, margin=0.5):
    """Select vehicle pairs whose corridors overlap, so separation binaries stay sparse."""
    state = np.atleast_2d(np.asarray(state, dtype=float))
    goal = np.atleast_2d(np.asarray(goal, dtype=float))
    margin = max(float(margin), 0.0)
    if state.shape[0] != goal.shape[0]:
        raise ValueError('State and goal must share vehicle count.')
    n_vehicles = state.shape[0]
    boxes = []
    for veh in range(n_vehicles):
        pts = np.vstack([state[veh, 0:2], goal[veh, 0:2]])
        min_xy = np.min(pts, axis=0) - margin
        max_xy = np.max(pts, axis=0) + margin
        boxes.append((min_xy, max_xy))
    
    active_pairs = []
    for i, j in combinations(range(n_vehicles), 2):
        min_i, max_i = boxes[i]
        min_j, max_j = boxes[j]
        overlap_x = (max_i[0] >= min_j[0]) and (max_j[0] >= min_i[0])
        overlap_y = (max_i[1] >= min_j[1]) and (max_j[1] >= min_i[1])
        if overlap_x and overlap_y:
            active_pairs.append((i, j))
    return active_pairs


def compute_milp_stats(idx, n_vehicles, n_obstacles, n_pairs):
    """Return simple counts that describe MILP size."""
    horizon = idx['T']
    n_vars = idx['nvar']
    n_binaries = len(idx['intcon'])
    n_continuous = n_vars - n_binaries
    n_fuel = 4 * n_vehicles * horizon
    n_dyn = 4 * n_vehicles * horizon
    n_init = 4 * n_vehicles
    # Terminal constraints: 8 per vehicle if using terminal cost (slack constraints)
    # or 4 per vehicle if using hard constraints
    use_terminal_cost = idx.get('useTerminalCost', True)
    n_term = 8 * n_vehicles if use_terminal_cost else 4 * n_vehicles
    # Obstacle constraints now cover k=0 to T (T+1 positions)
    n_obs_cons = 5 * n_vehicles * n_obstacles * (horizon + 1)
    n_pair_cons = 5 * n_pairs * horizon
    n_constraints = n_fuel + n_dyn + n_init + n_term + n_obs_cons + n_pair_cons
    
    return {
        'n_vars': int(n_vars),
        'n_binaries': int(n_binaries),
        'n_continuous': int(n_continuous),
        'n_constraints': int(n_constraints),
        'n_obstacles': int(n_obstacles),
        'n_pairs': int(n_pairs),
        'horizon_steps': int(horizon),
        'use_terminal_cost': use_terminal_cost
    }


def summarize_milp_stats(stats_list):
    """Aggregate MILP stats collected over a receding-horizon rollout."""
    if not stats_list:
        return {
            'n_vars_peak': 0,
            'n_binaries_peak': 0,
            'n_constraints_peak': 0,
            'avg_active_obstacles': 0.0,
            'milp_calls': 0
        }
    
    n_vars_peak = max(s.get('n_vars', 0) for s in stats_list)
    n_binaries_peak = max(s.get('n_binaries', 0) for s in stats_list)
    n_constraints_peak = max(s.get('n_constraints', 0) for s in stats_list)
    avg_active_obstacles = float(np.mean([s.get('n_obstacles', 0) for s in stats_list]))
    
    return {
        'n_vars_peak': int(n_vars_peak),
        'n_binaries_peak': int(n_binaries_peak),
        'n_constraints_peak': int(n_constraints_peak),
        'avg_active_obstacles': avg_active_obstacles,
        'milp_calls': len(stats_list)
    }


def minimum_pairwise_distance(trajectory):
    """Compute the minimum inter-vehicle distance over the provided trajectory."""
    traj_arr = np.asarray(trajectory, dtype=float)
    if traj_arr.ndim != 3 or traj_arr.shape[0] < 2:
        return float('inf')
    
    positions = traj_arr[:, 0:2, :]
    n_vehicles = positions.shape[0]
    min_dist = float('inf')
    
    for i in range(n_vehicles):
        for j in range(i + 1, n_vehicles):
            diff = positions[i] - positions[j]
            dist = np.sqrt(np.sum(diff**2, axis=0))
            min_dist = min(min_dist, float(np.min(dist)))
    
    return min_dist


def pack_indices(T, n_obs, n_agents, n_pairs=0, use_terminal_cost=True):
    """Map every decision variable to a contiguous index range.
    
    Args:
        T: Horizon length (number of time steps)
        n_obs: Number of obstacles
        n_agents: Number of vehicles
        n_pairs: Number of vehicle pairs for separation constraints
        use_terminal_cost: If True, add slack variables for terminal cost (Schouwenaars eq. 4)
    """
    n_nodes = T + 1
    offset = 0
    
    idx = {}
    idx['T'] = T
    idx['nObstacles'] = n_obs
    idx['nVehicles'] = n_agents
    idx['useTerminalCost'] = use_terminal_cost
    
    # State variables (positions and velocities)
    idx['px'] = []
    idx['py'] = []
    idx['vx'] = []
    idx['vy'] = []
    for veh in range(n_agents):
        idx['px'].append(list(range(offset, offset + n_nodes)))
        offset += n_nodes
        idx['py'].append(list(range(offset, offset + n_nodes)))
        offset += n_nodes
        idx['vx'].append(list(range(offset, offset + n_nodes)))
        offset += n_nodes
        idx['vy'].append(list(range(offset, offset + n_nodes)))
        offset += n_nodes
    
    # Input variables
    idx['ux'] = []
    idx['uy'] = []
    for veh in range(n_agents):
        idx['ux'].append(list(range(offset, offset + T)))
        offset += T
        idx['uy'].append(list(range(offset, offset + T)))
        offset += T
    
    # Slack variables for L1 norm (fuel cost)
    idx['sx'] = []
    idx['sy'] = []
    for veh in range(n_agents):
        idx['sx'].append(list(range(offset, offset + T)))
        offset += T
        idx['sy'].append(list(range(offset, offset + T)))
        offset += T
    
    # Terminal cost slack variables (for soft terminal constraint)
    # Following Schouwenaars et al. (2001) Eq. 4: f(s_N) = p' |s_N - goal|
    idx['tpx'] = []  # Terminal position slack x
    idx['tpy'] = []  # Terminal position slack y
    idx['tvx'] = []  # Terminal velocity slack x
    idx['tvy'] = []  # Terminal velocity slack y
    if use_terminal_cost:
        for veh in range(n_agents):
            idx['tpx'].append(offset)
            offset += 1
            idx['tpy'].append(offset)
            offset += 1
            idx['tvx'].append(offset)
            offset += 1
            idx['tvy'].append(offset)
            offset += 1
    
    # Binary variables for obstacles (including k=0 for initial state check)
    # Changed: Now covers k=0 to T (all positions including initial)
    idx['a'] = []
    n_obs_steps = T + 1  # Include initial state
    a_cube = np.zeros((n_agents, 4, n_obs, n_obs_steps), dtype=int)
    for veh in range(n_agents):
        for side in range(4):
            for obs in range(n_obs):
                for k in range(n_obs_steps):
                    idx_val = offset
                    idx['a'].append(idx_val)
                    a_cube[veh, side, obs, k] = idx_val
                    offset += 1
    
    # Binary variables for vehicle separation
    idx['pairBins'] = []
    pair_cube = np.zeros((n_pairs, 4, T), dtype=int)
    for pair in range(n_pairs):
        for side in range(4):
            for k in range(T):
                idx_val = offset
                idx['pairBins'].append(idx_val)
                pair_cube[pair, side, k] = idx_val
                offset += 1
    
    idx['nvar'] = offset
    
    # Combined indices
    idx['state'] = [idx['px'], idx['py'], idx['vx'], idx['vy']]
    idx['input'] = [idx['ux'], idx['uy']]
    idx['slack'] = [idx['sx'], idx['sy']]
    idx['terminalSlack'] = [idx['tpx'], idx['tpy'], idx['tvx'], idx['tvy']]
    idx['intcon'] = idx['a'] + idx['pairBins']
    
    # Reshape binary variables for obstacle constraints
    idx['aCube'] = a_cube
    idx['pairCube'] = pair_cube
    
    return idx


def build_fuel_cost(idx, terminal_weights=None):
    """Implement ||u||_1 + terminal cost by introducing slack variables.
    
    Following Schouwenaars et al. (2001) Eq. 4 and 6:
    J_T = sum(q'|s_i|) + sum(r'|u_i|) + p'|s_N - goal|
    
    Args:
        idx: Index dictionary from pack_indices
        terminal_weights: Dict with 'position' and 'velocity' weights for terminal cost.
                         If None, defaults to {'position': 10.0, 'velocity': 5.0}
    """
    if terminal_weights is None:
        terminal_weights = {'position': 10.0, 'velocity': 5.0}
    
    # Cost function: minimize sum of slack variables
    f_cost = {}
    
    # Fuel cost (L1 norm of inputs)
    for veh in range(idx['nVehicles']):
        for k in range(idx['T']):
            f_cost[idx['sx'][veh][k]] = 1.0
            f_cost[idx['sy'][veh][k]] = 1.0
    
    # Terminal cost (L1 norm of terminal state error)
    # This replaces the hard terminal constraint with a soft penalty
    if idx.get('useTerminalCost', False):
        pos_weight = terminal_weights.get('position', 10.0)
        vel_weight = terminal_weights.get('velocity', 5.0)
        for veh in range(idx['nVehicles']):
            f_cost[idx['tpx'][veh]] = pos_weight
            f_cost[idx['tpy'][veh]] = pos_weight
            f_cost[idx['tvx'][veh]] = vel_weight
            f_cost[idx['tvy'][veh]] = vel_weight
    
    return f_cost


def build_bounds(idx, scn):
    """Apply simple min/max bounds to all states, controls, and binaries."""
    lb = {}
    ub = {}
    
    # Position bounds
    for veh in range(idx['nVehicles']):
        for i in idx['px'][veh]:
            lb[i] = scn['posBounds'][0, 0]
            ub[i] = scn['posBounds'][0, 1]
        for i in idx['py'][veh]:
            lb[i] = scn['posBounds'][1, 0]
            ub[i] = scn['posBounds'][1, 1]
    
    # Velocity bounds
    for veh in range(idx['nVehicles']):
        for i in idx['vx'][veh]:
            lb[i] = scn['velBounds'][0, 0]
            ub[i] = scn['velBounds'][0, 1]
        for i in idx['vy'][veh]:
            lb[i] = scn['velBounds'][1, 0]
            ub[i] = scn['velBounds'][1, 1]
    
    # Input bounds
    for veh in range(idx['nVehicles']):
        for i in idx['ux'][veh]:
            lb[i] = scn['inputBounds'][0, 0]
            ub[i] = scn['inputBounds'][0, 1]
        for i in idx['uy'][veh]:
            lb[i] = scn['inputBounds'][1, 0]
            ub[i] = scn['inputBounds'][1, 1]
    
    # Fuel slack variables (non-negative)
    for veh in range(idx['nVehicles']):
        for i in idx['sx'][veh]:
            lb[i] = 0.0
            ub[i] = None
        for i in idx['sy'][veh]:
            lb[i] = 0.0
            ub[i] = None
    
    # Terminal cost slack variables (non-negative)
    if idx.get('useTerminalCost', False):
        for veh in range(idx['nVehicles']):
            lb[idx['tpx'][veh]] = 0.0
            ub[idx['tpx'][veh]] = None
            lb[idx['tpy'][veh]] = 0.0
            ub[idx['tpy'][veh]] = None
            lb[idx['tvx'][veh]] = 0.0
            ub[idx['tvx'][veh]] = None
            lb[idx['tvy'][veh]] = 0.0
            ub[idx['tvy'][veh]] = None
    
    # Binary variables (for obstacles and separation)
    for i in idx['a']:
        lb[i] = 0.0
        ub[i] = 1.0
    for i in idx['pairBins']:
        lb[i] = 0.0
        ub[i] = 1.0
    
    return lb, ub


def assemble_milp_data(scn):
    """Assemble MILP data structure.
    
    Supports two modes for terminal conditions (following Schouwenaars et al. 2001):
    - useTerminalCost=True (default): Soft terminal constraint via cost function (Eq. 4)
    - useTerminalCost=False: Hard terminal equality constraint
    
    The terminal cost approach is preferred for receding horizon as it prevents
    infeasibility when the goal is unreachable within the horizon.
    """
    A, B = double_integrator_matrices(scn['dt'])
    start = np.atleast_2d(np.asarray(scn['start'], dtype=float))
    goal = np.atleast_2d(np.asarray(scn['goal'], dtype=float))
    if start.shape[0] != goal.shape[0]:
        raise ValueError('Start and goal must be defined for the same number of vehicles.')
    n_vehicles = int(scn.get('nVehicles', start.shape[0]))
    if start.shape[0] != n_vehicles:
        raise ValueError('Mismatch between start states and declared number of vehicles.')
    obstacles = np.asarray(scn.get('obstacles', []), dtype=float)
    if obstacles.size == 0:
        obstacles = np.zeros((0, 4))
    obstacles = obstacles.reshape((-1, 4))
    enforce_sep = bool(scn.get('enforceSeparation', False)) and n_vehicles > 1
    if enforce_sep:
        pair_margin = scn.get('pairPruneMargin', 0.5)
        vehicle_pairs = identify_active_pairs(start, goal, margin=pair_margin)
    else:
        vehicle_pairs = []
    
    # Terminal cost vs terminal constraint mode
    use_terminal_cost = scn.get('useTerminalCost', True)  # Default: use soft constraint
    
    idx = pack_indices(scn['horizonSteps'], obstacles.shape[0], n_vehicles, 
                       len(vehicle_pairs), use_terminal_cost=use_terminal_cost)
    
    # Terminal cost weights (Schouwenaars eq. 4: p' |s_N - goal|)
    terminal_weights = scn.get('terminalWeights', {'position': 10.0, 'velocity': 5.0})
    
    f_cost = build_fuel_cost(idx, terminal_weights)
    lb, ub = build_bounds(idx, scn)
    
    data = {
        'f_cost': f_cost,
        'lb': lb,
        'ub': ub,
        'idx': idx,
        'dt': scn['dt'],
        'start': start,
        'goal': goal,
        'obstacles': obstacles,
        'obstacleBuffer': scn.get('obstacleBuffer', 0.25),
        'safeSeparation': scn.get('safeSeparation', 0.0),
        'separationBigM': scn.get('separationBigM', 800.0),
        'vehicle_pairs': vehicle_pairs,
        'enforceSeparation': enforce_sep,
        'nVehicles': n_vehicles,
        'A': A,
        'B': B,
        'useTerminalCost': use_terminal_cost,
        'terminalWeights': terminal_weights
    }
    data['stats'] = compute_milp_stats(idx, n_vehicles, obstacles.shape[0], len(vehicle_pairs))
    
    return data


def solve_milp(data, time_limit=None):
    """Call SCIP solver and process the result.
    
    Args:
        data: MILP data dictionary
        time_limit: Time limit in seconds (None = no limit)
    """
    model = Model("MILP")
    model.setParam('display/verblevel', 0)  # Silent mode
    
    # Add timeout if specified
    if time_limit is not None and time_limit > 0:
        model.setParam('limits/time', float(time_limit))
    
    idx = data['idx']
    nvar = idx['nvar']
    A = data['A']
    B = data['B']
    n_vehicles = idx['nVehicles']
    horizon = idx['T']
    
    # Create variables
    vars_dict = {}
    intcon = set(idx['intcon'])
    for i in range(nvar):
        vtype = 'B' if i in intcon else 'C'  # Binary or Continuous
        lb_val = data['lb'].get(i, None)
        ub_val = data['ub'].get(i, None)
        
        if lb_val is None:
            lb_val = -model.infinity()
        if ub_val is None:
            ub_val = model.infinity()
        
        vars_dict[i] = model.addVar(name=f'x{i}', vtype=vtype, lb=lb_val, ub=ub_val)
    
    # Set objective
    obj_expr = quicksum(data['f_cost'].get(i, 0.0) * vars_dict[i] for i in range(nvar))
    model.setObjective(obj_expr, 'minimize')
    
    # Add constraints
    # Fuel cost constraints
    for veh in range(n_vehicles):
        for k in range(horizon):
            ux_idx = idx['ux'][veh][k]
            uy_idx = idx['uy'][veh][k]
            sx_idx = idx['sx'][veh][k]
            sy_idx = idx['sy'][veh][k]
            
            # ux <= sx, -ux <= sx
            model.addCons(vars_dict[ux_idx] <= vars_dict[sx_idx], name=f'fuel_ux1_v{veh}_{k}')
            model.addCons(-vars_dict[ux_idx] <= vars_dict[sx_idx], name=f'fuel_ux2_v{veh}_{k}')
            # uy <= sy, -uy <= sy
            model.addCons(vars_dict[uy_idx] <= vars_dict[sy_idx], name=f'fuel_uy1_v{veh}_{k}')
            model.addCons(-vars_dict[uy_idx] <= vars_dict[sy_idx], name=f'fuel_uy2_v{veh}_{k}')
    
    # Dynamics constraints
    for veh in range(n_vehicles):
        for k in range(horizon):
            # px[k+1] = px[k] + dt*vx[k] + 0.5*dt^2*ux[k]
            model.addCons(
                vars_dict[idx['px'][veh][k+1]] ==
                A[0, 0] * vars_dict[idx['px'][veh][k]] +
                A[0, 2] * vars_dict[idx['vx'][veh][k]] +
                B[0, 0] * vars_dict[idx['ux'][veh][k]],
                name=f'dyn_px_v{veh}_{k}'
            )
            
            # py[k+1] = py[k] + dt*vy[k] + 0.5*dt^2*uy[k]
            model.addCons(
                vars_dict[idx['py'][veh][k+1]] ==
                A[1, 1] * vars_dict[idx['py'][veh][k]] +
                A[1, 3] * vars_dict[idx['vy'][veh][k]] +
                B[1, 1] * vars_dict[idx['uy'][veh][k]],
                name=f'dyn_py_v{veh}_{k}'
            )
            
            # vx[k+1] = vx[k] + dt*ux[k]
            model.addCons(
                vars_dict[idx['vx'][veh][k+1]] ==
                A[2, 2] * vars_dict[idx['vx'][veh][k]] +
                B[2, 0] * vars_dict[idx['ux'][veh][k]],
                name=f'dyn_vx_v{veh}_{k}'
            )
            
            # vy[k+1] = vy[k] + dt*uy[k]
            model.addCons(
                vars_dict[idx['vy'][veh][k+1]] ==
                A[3, 3] * vars_dict[idx['vy'][veh][k]] +
                B[3, 1] * vars_dict[idx['uy'][veh][k]],
                name=f'dyn_vy_v{veh}_{k}'
            )
    
    # Initial state constraints
    start = data.get('start', None)
    if start is not None:
        for veh in range(n_vehicles):
            model.addCons(vars_dict[idx['px'][veh][0]] == start[veh, 0], name=f'init_px_v{veh}')
            model.addCons(vars_dict[idx['py'][veh][0]] == start[veh, 1], name=f'init_py_v{veh}')
            model.addCons(vars_dict[idx['vx'][veh][0]] == start[veh, 2], name=f'init_vx_v{veh}')
            model.addCons(vars_dict[idx['vy'][veh][0]] == start[veh, 3], name=f'init_vy_v{veh}')
    
    # Terminal state handling: cost (soft) vs constraint (hard)
    # Following Schouwenaars et al. (2001) Eq. 4: terminal COST is preferred for receding horizon
    goal = data.get('goal', None)
    use_terminal_cost = data.get('useTerminalCost', True)
    
    if goal is not None:
        if use_terminal_cost:
            # Soft terminal constraint via cost function (prevents infeasibility)
            # |px[T] - goal_x| <= tpx  =>  px[T] - goal_x <= tpx AND goal_x - px[T] <= tpx
            for veh in range(n_vehicles):
                # Position x slack
                model.addCons(
                    vars_dict[idx['px'][veh][horizon]] - goal[veh, 0] <= vars_dict[idx['tpx'][veh]],
                    name=f'term_px_pos_v{veh}'
                )
                model.addCons(
                    goal[veh, 0] - vars_dict[idx['px'][veh][horizon]] <= vars_dict[idx['tpx'][veh]],
                    name=f'term_px_neg_v{veh}'
                )
                # Position y slack
                model.addCons(
                    vars_dict[idx['py'][veh][horizon]] - goal[veh, 1] <= vars_dict[idx['tpy'][veh]],
                    name=f'term_py_pos_v{veh}'
                )
                model.addCons(
                    goal[veh, 1] - vars_dict[idx['py'][veh][horizon]] <= vars_dict[idx['tpy'][veh]],
                    name=f'term_py_neg_v{veh}'
                )
                # Velocity x slack
                model.addCons(
                    vars_dict[idx['vx'][veh][horizon]] - goal[veh, 2] <= vars_dict[idx['tvx'][veh]],
                    name=f'term_vx_pos_v{veh}'
                )
                model.addCons(
                    goal[veh, 2] - vars_dict[idx['vx'][veh][horizon]] <= vars_dict[idx['tvx'][veh]],
                    name=f'term_vx_neg_v{veh}'
                )
                # Velocity y slack
                model.addCons(
                    vars_dict[idx['vy'][veh][horizon]] - goal[veh, 3] <= vars_dict[idx['tvy'][veh]],
                    name=f'term_vy_pos_v{veh}'
                )
                model.addCons(
                    goal[veh, 3] - vars_dict[idx['vy'][veh][horizon]] <= vars_dict[idx['tvy'][veh]],
                    name=f'term_vy_neg_v{veh}'
                )
        else:
            # Hard terminal equality constraint (original behavior, may cause infeasibility)
            for veh in range(n_vehicles):
                model.addCons(vars_dict[idx['px'][veh][horizon]] == goal[veh, 0], name=f'term_px_v{veh}')
                model.addCons(vars_dict[idx['py'][veh][horizon]] == goal[veh, 1], name=f'term_py_v{veh}')
                model.addCons(vars_dict[idx['vx'][veh][horizon]] == goal[veh, 2], name=f'term_vx_v{veh}')
                model.addCons(vars_dict[idx['vy'][veh][horizon]] == goal[veh, 3], name=f'term_vy_v{veh}')
    
    # Obstacle constraints
    # Now covers k=0 to T (all positions including initial state)
    # This ensures the vehicle doesn't start inside an obstacle
    obstacles = data.get('obstacles', None)
    if obstacles is not None and len(obstacles) > 0:
        buffer = data.get('obstacleBuffer', 0.25)
        Mx = 800
        My = 800
        n_obs = len(obstacles)
        
        for veh in range(n_vehicles):
            # k=0 to horizon (T+1 positions total, including initial)
            for k in range(horizon + 1):
                for obs in range(n_obs):
                    rect = obstacles[obs, :]
                    bins = idx['aCube'][veh, :, obs, k]
                    
                    xmin = rect[0] - buffer
                    xmax = rect[1] + buffer
                    ymin = rect[2] - buffer
                    ymax = rect[3] + buffer
                    
                    eps = 1e-4  # small tolerance to avoid numerical ambiguity on edges
                    
                    # Ensure the state lies on one of the four sides outside the rectangle.
                    # Position at time k (not k+1 anymore, since we're iterating all positions)
                    model.addCons(
                        vars_dict[idx['px'][veh][k]] <= xmin - eps + Mx * vars_dict[bins[0]],
                        name=f'obs_px_left_v{veh}_{k}_{obs}'
                    )
                    
                    model.addCons(
                        vars_dict[idx['px'][veh][k]] >= xmax + eps - Mx * vars_dict[bins[1]],
                        name=f'obs_px_right_v{veh}_{k}_{obs}'
                    )
                    
                    model.addCons(
                        vars_dict[idx['py'][veh][k]] <= ymin - eps + My * vars_dict[bins[2]],
                        name=f'obs_py_bottom_v{veh}_{k}_{obs}'
                    )
                    
                    model.addCons(
                        vars_dict[idx['py'][veh][k]] >= ymax + eps - My * vars_dict[bins[3]],
                        name=f'obs_py_top_v{veh}_{k}_{obs}'
                    )
                    
                    # At least one binary must be 0 (i.e., at least one side constraint active)
                    model.addCons(
                        vars_dict[bins[0]] + vars_dict[bins[1]] +
                        vars_dict[bins[2]] + vars_dict[bins[3]] <= 3,
                        name=f'obs_bin_v{veh}_{k}_{obs}'
                    )

    # Pairwise vehicle separation constraints
    safe_sep = float(data.get('safeSeparation', 0.0))
    vehicle_pairs = data.get('vehicle_pairs', [])
    pair_cube = idx.get('pairCube', np.zeros((0, 4, horizon), dtype=int))
    Msep = float(data.get('separationBigM', 800.0))
    if (data.get('enforceSeparation', False) and safe_sep > 0 and
            len(vehicle_pairs) > 0 and pair_cube.size > 0):
        for p_idx, (veh_i, veh_j) in enumerate(vehicle_pairs):
            for k in range(horizon):
                bins = pair_cube[p_idx, :, k]
                px_i = idx['px'][veh_i][k+1]
                px_j = idx['px'][veh_j][k+1]
                py_i = idx['py'][veh_i][k+1]
                py_j = idx['py'][veh_j][k+1]
                
                model.addCons(
                    vars_dict[px_i] <= vars_dict[px_j] - safe_sep + Msep * vars_dict[bins[0]],
                    name=f'sep_px_left_p{p_idx}_{k}'
                )
                model.addCons(
                    vars_dict[px_i] >= vars_dict[px_j] + safe_sep - Msep * vars_dict[bins[1]],
                    name=f'sep_px_right_p{p_idx}_{k}'
                )
                model.addCons(
                    vars_dict[py_i] <= vars_dict[py_j] - safe_sep + Msep * vars_dict[bins[2]],
                    name=f'sep_py_bottom_p{p_idx}_{k}'
                )
                model.addCons(
                    vars_dict[py_i] >= vars_dict[py_j] + safe_sep - Msep * vars_dict[bins[3]],
                    name=f'sep_py_top_p{p_idx}_{k}'
                )
                model.addCons(
                    vars_dict[bins[0]] + vars_dict[bins[1]] +
                    vars_dict[bins[2]] + vars_dict[bins[3]] <= 3,
                    name=f'sep_bin_p{p_idx}_{k}'
                )
    
    # Solve
    model.optimize()
    
    solution = {}
    status = model.getStatus()
    solution['status'] = status
    
    # Check for timeout
    if status == 'timelimit':
        solution['status'] = 'timeout'
        solution['exitflag'] = -1
        solution['cost'] = None
        solution['energy'] = np.nan
        solution['state'] = []
        solution['input'] = []
        model.freeProb()
        return solution
    
    if status == 'optimal':
        solution['exitflag'] = 1
        solution['cost'] = model.getObjVal()
        
        # Extract solution
        z = np.zeros(nvar)
        for i in range(nvar):
            z[i] = model.getVal(vars_dict[i])
        
        energy = 0.0
        for veh in range(n_vehicles):
            energy += np.sum(z[idx['sx'][veh]]) + np.sum(z[idx['sy'][veh]])
        solution['energy'] = energy
        solution['state'] = np.zeros((n_vehicles, 4, horizon + 1))
        solution['input'] = np.zeros((n_vehicles, 2, horizon))
        
        for veh in range(n_vehicles):
            for k in range(horizon + 1):
                solution['state'][veh, 0, k] = z[idx['px'][veh][k]]
                solution['state'][veh, 1, k] = z[idx['py'][veh][k]]
                solution['state'][veh, 2, k] = z[idx['vx'][veh][k]]
                solution['state'][veh, 3, k] = z[idx['vy'][veh][k]]
            for k in range(horizon):
                solution['input'][veh, 0, k] = z[idx['ux'][veh][k]]
                solution['input'][veh, 1, k] = z[idx['uy'][veh][k]]
    else:
        solution['exitflag'] = 0
        solution['cost'] = None
        solution['energy'] = np.nan
        solution['state'] = []
        solution['input'] = []
    
    model.freeProb()
    return solution


def plot_trajectory(trajectory, scn, title_str):
    """Plot trajectory visualization."""
    traj_arr = np.asarray(trajectory)
    if traj_arr.ndim == 2:
        traj_arr = traj_arr[np.newaxis, :, :]
    if traj_arr.ndim != 3:
        raise ValueError('Trajectory must be a 2D or 3D array.')
    n_vehicles = traj_arr.shape[0]
    pos = traj_arr[:, 0:2, :]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('white')
    ax.grid(True)
    ax.set_aspect('equal')
    
    # Draw obstacles
    obstacles = np.asarray(scn.get('obstacles', []), dtype=float)
    if obstacles.size == 0:
        obstacles = np.zeros((0, 4))
    obstacles = obstacles.reshape((-1, 4))
    for i in range(len(obstacles)):
        rect = obstacles[i, :]
        ax.add_patch(plt.Rectangle(
            (rect[0], rect[2]), rect[1] - rect[0], rect[3] - rect[2],
            facecolor=[0.9, 0.9, 0.9], edgecolor='k', linewidth=1.2))
    
    # Draw trajectory
    cmap = plt.cm.get_cmap('tab10', max(n_vehicles, 1))
    starts = np.atleast_2d(np.asarray(scn['start'], dtype=float))
    goals = np.atleast_2d(np.asarray(scn['goal'], dtype=float))
    for veh in range(n_vehicles):
        color = cmap(veh % cmap.N)
        ax.plot(
            pos[veh, 0, :], pos[veh, 1, :], '-o',
            color=color, linewidth=1.5, markersize=4,
            label=f'Vehicle {veh + 1}'
        )
        start_label = 'Start' if veh == 0 else None
        goal_label = 'Goal' if veh == 0 else None
        start_idx = starts[veh if veh < starts.shape[0] else -1]
        goal_idx = goals[veh if veh < goals.shape[0] else -1]
        ax.plot(
            start_idx[0], start_idx[1], 'o',
            markersize=8, markerfacecolor='none',
            markeredgecolor=color, label=start_label
        )
        ax.plot(
            goal_idx[0], goal_idx[1], 's',
            markersize=7, markerfacecolor=color,
            markeredgecolor=color, label=goal_label
        )
    
    # Mark start and end points
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(title_str)
    ax.legend(loc='best')
    
    return fig


def run_receding_horizon(scenario, Thor):
    """True receding horizon simulation - applies only the first control input at each iteration."""
    dt = scenario['dt']
    horizon_steps = int(round(Thor / dt))  # Horizon length (number of steps)
    
    # Initial state
    start = np.atleast_2d(np.asarray(scenario['start'], dtype=float))
    goal = np.atleast_2d(np.asarray(scenario['goal'], dtype=float))
    if start.shape[0] != goal.shape[0]:
        raise ValueError('Start and goal must contain the same number of vehicles.')
    n_vehicles = start.shape[0]
    current_state = start.copy()
    trajectory = current_state[:, :, np.newaxis].copy()
    total_energy = 0.0
    total_computation = 0.0
    iterations = 0
    milp_stats_log = []
    
    # Target tolerances
    pos_tolerance = 0.1  # meters
    vel_tolerance = 0.1  # m/s
    max_iterations = 400  # Maximum number of iterations
    
    A, B = double_integrator_matrices(dt)
    
    while iterations < max_iterations:
        # Check if target is reached
        pos_error = np.linalg.norm(current_state[:, 0:2] - goal[:, 0:2], axis=1)
        vel_error = np.linalg.norm(current_state[:, 2:4] - goal[:, 2:4], axis=1)
        
        if np.all(pos_error < pos_tolerance) and np.all(vel_error < vel_tolerance):
            break
        
        # Plan from current state for horizon length
        local_scenario = scenario.copy()
        local_scenario['start'] = current_state.copy()
        local_scenario['goal'] = goal.copy()
        local_scenario['horizonSteps'] = horizon_steps
        local_scenario['nVehicles'] = n_vehicles
        prune_margin = scenario.get('obstaclePruneMargin', 2.0)
        local_scenario['obstacles'] = filter_active_obstacles(
            current_state, goal, scenario.get('obstacles', []), prune_margin)
        
        # Solve MILP problem
        milp_data = assemble_milp_data(local_scenario)
        milp_stats_log.append(milp_data.get('stats', {}))
        tic_solve = time.time()
        solution = solve_milp(milp_data)
        solve_time = time.time() - tic_solve
        
        if solution.get('exitflag', 0) <= 0:
            print(f'Warning: MILP infeasible at iteration {iterations + 1}')
            break
        
        # Apply only the first control input
        applied_input = solution.get('input', None)
        if isinstance(applied_input, np.ndarray) and applied_input.ndim == 3 and applied_input.shape[2] > 0:
            applied_input = applied_input[:, :, 0]
        else:
            applied_input = np.zeros((n_vehicles, 2))
        
        # Update state (double integrator dynamics)
        next_state = np.zeros_like(current_state)
        for veh in range(n_vehicles):
            next_state[veh, :] = A @ current_state[veh, :] + B @ applied_input[veh, :]
        current_state = next_state
        
        # Update metrics
        total_energy = total_energy + dt * np.sum(np.abs(applied_input))
        total_computation = total_computation + solve_time
        iterations = iterations + 1
        trajectory = np.concatenate([trajectory, current_state[:, :, np.newaxis]], axis=2)
    
    # Calculate results
    Tarr = iterations * dt
    Etot = total_energy
    Tcomp = total_computation
    Tit = total_computation / max(1, iterations)
    milp_stats = summarize_milp_stats(milp_stats_log)
    min_sep = minimum_pairwise_distance(trajectory)
    safe_sep = float(scenario.get('safeSeparation', 0.0))
    collision_flag = safe_sep > 0 and np.isfinite(min_sep) and (min_sep < safe_sep)
    
    return Tarr, Etot, Tcomp, Tit, trajectory, milp_stats, min_sep, collision_flag


def main():
    """Main function to run receding horizon sweep."""
    # Scenario definition
    # Following Schouwenaars et al. (2001) paper formulation
    base_scenario = {
        'dt': 0.2,
        'start': np.array([
            [5.5, 0.0, 0.0, 0.0],
            [6.0, -0.7, 0.0, 0.0],
            [5.8, 0.9, 0.0, 0.0]
        ]),
        'goal': np.array([
            [9, 8, 0.0, 0.0],
            [10, 8, 0.0, 0.0],
            [10, 7, 0.0, 0.0]
        ]),
        'obstacles': np.array([
            [-2.0, -0.5, -5.5, -1.5],
            [-2.0, -0.5, 0.0, 8.5],
            [2.5, 4.5, -3.5, 0.5],
            [2.5, 4.5, 2.0, 6.5],
            [5.5, 8.0, 2.0, 6.5],
            [9.0, 11.5, 0.0, 4.5],
            [11.5, 14.5, 4.5, 9.5],
            [5.0, 7.0, -5.8, -1.0],
            [8.2, 10.8, -6.2, -2.0],
            [12.0, 15.0, -4.5, -0.5]
        ]),
        'posBounds': np.array([[-50, 100], [-50, 200]]),
        'velBounds': np.array([[-22.5, 22.5], [-22.5, 22.5]]),
        'inputBounds': np.array([[-9.0, 9.0], [-9.0, 9.0]]),
        'obstacleBuffer': 0.25,
        'obstaclePruneMargin': 2.0,
        'safeSeparation': 0.35,
        'enforceSeparation': True,
        'pairPruneMargin': 0.6,
        'separationBigM': 25.0,
        'nVehicles': 3,
        # Terminal cost settings (Schouwenaars et al. 2001, Eq. 4)
        # useTerminalCost=True uses soft constraint (prevents infeasibility)
        # useTerminalCost=False uses hard equality constraint (original behavior)
        'useTerminalCost': True,
        'terminalWeights': {'position': 10.0, 'velocity': 5.0}
    }
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Receding horizon sweep
    horizon_times = [3.0]  # seconds
    sweep_results = []
    
    print('Running true receding horizon simulations...')
    
    for i, Thor in enumerate(horizon_times):
        print(f'Thor = {Thor:.1f} s... ', end='', flush=True)
        
        # Run receding horizon simulation
        Tarr, Etot, Tcomp, Tit, trajectory, milp_stats, min_sep, collision_flag = run_receding_horizon(base_scenario, Thor)
        
        result = {
            'horizonTime': Thor,
            'arrival': Tarr,
            'fuel': Etot,
            'solve_time': Tcomp,
            'iter_time': Tit,
            'trajectory': trajectory,
            'milp_stats': milp_stats,
            'min_sep': min_sep,
            'collision': collision_flag
        }
        sweep_results.append(result)
        
        print(
            f'Tarr={Tarr:.1f}, Etot={Etot:.1f}, Tcomp={Tcomp:.1f}, '
            f'nInt={milp_stats.get("n_binaries_peak", 0)}, '
            f'minSep={min_sep:.3f} m, safe={"YES" if not collision_flag else "NO"}')
    
    # Print results in table format
    heading = 'Table 1: Arrival times, fuel consumption and computation times for different horizon lengths.'
    print(f'\n{heading}')
    header_line = (
        f"{'Thor(s)':>8} {'Tarr(s)':>10} {'Etot':>12} {'Tcomp(s)':>12} "
        f"{'Tit(s)':>12} {'nVar':>10} {'nInt':>10} {'minSep(m)':>12} {'safe?':>8}"
    )
    print(header_line)
    table_lines = [heading, header_line]
    for result in sweep_results:
        stats = result.get('milp_stats', {})
        line = (f"{result['horizonTime']:>8.1f} {result['arrival']:>10.1f} "
                f"{result['fuel']:>12.1f} {result['solve_time']:>12.0f} "
                f"{result['iter_time']:>12.2f} {stats.get('n_vars_peak', 0):>10} "
                f"{stats.get('n_binaries_peak', 0):>10} "
                f"{result.get('min_sep', float('inf')):>12.3f} "
                f"{'OK' if not result.get('collision', False) else 'HIT':>8}")
        print(line)
        table_lines.append(line)
    
    # Persist results to a text file for later reference
    txt_path = os.path.join(output_dir, 'true_receding_horizon_results.txt')
    with open(txt_path, 'w', encoding='utf-8') as fh:
        fh.write("\n".join(table_lines) + "\n")
    print(f'Results saved to {txt_path}')
    
    # Visualization for the largest horizon
    main_result = sweep_results[-1]
    if main_result['trajectory'].size > 0:
        fig = plot_trajectory(
            main_result['trajectory'], base_scenario,
            f"True Receding Horizon (Thor = {main_result['horizonTime']:.1f} s)")
        png_path = os.path.join(output_dir, 'true_receding_horizon_snapshot.png')
        fig.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f'\nTrue receding horizon snapshot saved to {png_path}')
        
        # Keep figure open for inspection
        print('Figure remains open for inspection.')
        plt.show()


if __name__ == '__main__':
    main()
