"""
Generate branching samples with Ecole's StrongBranching expert.

- Expects LP instances via glob patterns.
- Saves NodeBipartite observations and expert actions to pickle files.
"""

import argparse
import glob
import pickle
import random
from pathlib import Path

import ecole


def collect_split(pattern: str, out_dir: Path, n_samples: int, seed: int, time_limit: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")

    scip_params = {}
    if time_limit > 0:
        scip_params["limits/time"] = time_limit

    obs_fn = {
        "bipartite": ecole.observation.NodeBipartite(),
        "scores": ecole.observation.StrongBranchingScores(),
    }
    env = ecole.environment.Branching(
        observation_function=obs_fn,
        reward_function=ecole.reward.NNodes(),
        scip_params=scip_params,
    )
    rng = random.Random(seed)

    i = 0
    episode = 0
    while i < n_samples:
        instance = rng.choice(files)
        obs_dict, action_set, _, done, info = env.reset(str(instance))
        while not done and i < n_samples:
            scores = obs_dict["scores"]
            # pick best score (scores aligned with action_set)
            best_idx = int(scores.argmax())
            action = action_set[best_idx]
            sample = {
                "episode": episode,
                "instance": instance,
                "observation": obs_dict["bipartite"],
                "action_set": action_set,
                "expert_action": action,
            }
            with open(out_dir / f"sample_{i+1}.pkl", "wb") as fh:
                pickle.dump(sample, fh)
            i += 1
            obs_dict, action_set, _, done, info = env.step(action)
        episode += 1
    print(f"[ok] {i} samples written to {out_dir}")


def parse_args():
    ap = argparse.ArgumentParser(description="Collect branching samples with Ecole StrongBranching expert.")
    ap.add_argument("--train-pattern", required=True, help="Glob for train LPs, e.g., /path/to/train/*.lp")
    ap.add_argument("--valid-pattern", required=True, help="Glob for valid LPs, e.g., /path/to/valid/*.lp")
    ap.add_argument("--test-pattern", required=True, help="Glob for test LPs, e.g., /path/to/test/*.lp")
    ap.add_argument("--out-dir", required=True, help="Output root directory for samples.")
    ap.add_argument("--train-samples", type=int, default=1000, help="Number of train samples.")
    ap.add_argument("--valid-samples", type=int, default=200, help="Number of valid samples.")
    ap.add_argument("--test-samples", type=int, default=200, help="Number of test samples.")
    ap.add_argument("--seed", type=int, default=0, help="Base seed (split seeds = seed, seed+1, seed+2).")
    ap.add_argument("--time-limit", type=float, default=600.0, help="SCIP time limit per episode (seconds). Use 0 to disable.")
    return ap.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.out_dir).resolve()

    collect_split(
        args.train_pattern,
        out_root / "train",
        args.train_samples,
        args.seed,
        args.time_limit,
    )
    collect_split(
        args.valid_pattern,
        out_root / "valid",
        args.valid_samples,
        args.seed + 1,
        args.time_limit,
    )
    collect_split(
        args.test_pattern,
        out_root / "test",
        args.test_samples,
        args.seed + 2,
        args.time_limit,
    )


if __name__ == "__main__":
    main()
