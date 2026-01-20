"""
Generate branching samples with Ecole's StrongBranching expert.

- Expects LP instances via glob patterns.
- Saves NodeBipartite observations and expert actions to pickle files.

Bu dosya Gasse et al. (2019) örneğindeki veri toplama mantığını
path-planning MILP'lerimize uyarlıyor. En önemli fark: LP instance'lar
diskte hazır (generate_instances_path_planning.py ile üretiliyor) ve
biz burada güçlü uzmanı (StrongBranching) sadece belli olasılıkla
çağırıp o düğümlerde etiket topluyoruz.
"""

import argparse
import glob
import pickle
import random
from pathlib import Path

import numpy as np
import ecole


class ExploreThenStrongBranch:
    """
    Custom observation function that randomly returns either:

    - strong branching scores (expensive expert) + NodeBipartite state (only when expert chosen), or
    - pseudocost scores (cheap exploration) + None (no NodeBipartite extraction).

    This avoids computing NodeBipartite on non-expert nodes, which can be the main bottleneck on large MILPs.
    """

    def __init__(self, expert_probability):
        self.expert_probability = float(expert_probability)
        self.pseudocosts_function = ecole.observation.Pseudocosts()
        self.strong_branching_function = ecole.observation.StrongBranchingScores()
        self.bipartite_function = ecole.observation.NodeBipartite()

    def before_reset(self, model):
        """Called at initialization of the environment (before dynamics are reset)."""
        self.pseudocosts_function.before_reset(model)
        self.strong_branching_function.before_reset(model)
        self.bipartite_function.before_reset(model)

    def extract(self, model, done):
        """Should we return strong branching or pseudocost scores at this node?"""
        probabilities = [1 - self.expert_probability, self.expert_probability]
        expert_chosen = bool(np.random.choice(np.arange(2), p=probabilities))
        if expert_chosen:
            scores = self.strong_branching_function.extract(model, done)
            node_observation = self.bipartite_function.extract(model, done)
            return scores, True, node_observation
        scores = self.pseudocosts_function.extract(model, done)
        return scores, False, None


def scores_are_valid(candidate_scores: np.ndarray) -> bool:
    """Filter out inconsistent expert scores (NaN/inf/negative)."""
    if candidate_scores.size == 0:
        return False
    if not np.all(np.isfinite(candidate_scores)):
        return False
    return bool(np.all(candidate_scores >= 0))


def collect_split(
    pattern: str,
    out_dir: Path,
    n_samples: int,
    seed: int,
    time_limit: float,
    split_name: str = "",
    max_ep_samples: int | None = None,
    max_samples_per_instance: int | None = None,
    expert_probability: float = 0.05,
    filter_invalid_scores: bool = False,
    resume: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")
    if max_samples_per_instance is not None:
        max_possible = max_samples_per_instance * len(files)
        if n_samples > max_possible:
            raise ValueError(
                f"Requested {n_samples} samples, but max per instance "
                f"{max_samples_per_instance} * {len(files)} instances = {max_possible}."
            )

    # SCIP parametreleri: notebook'taki gibi separations/presolve'u hafifletip
    # veri toplamayı hızlandırıyoruz.
    seed = int(seed) % 2147483648
    scip_params = {
        "separating/maxrounds": 0,
        "presolving/maxrestarts": 0,
        "randomization/permutevars": True,
        "randomization/permutationseed": seed,
        "randomization/randomseedshift": seed,
        "timing/clocktype": 2,
    }
    if time_limit > 0:
        scip_params["limits/time"] = time_limit

    # Explore-then-strong-branch: her düğümde expert_probability olasılıkla
    # StrongBranchingScores (yavaş ama doğru) ve o düğümün NodeBipartite state'i.
    # Aksi halde Pseudocosts (hızlı keşif) ve NodeBipartite çıkarımı yapılmaz (hız için).
    obs_fn = ExploreThenStrongBranch(expert_probability=expert_probability)
    env = ecole.environment.Branching(
        observation_function=obs_fn,
        reward_function=ecole.reward.NNodes(),
        scip_params=scip_params,
    )
    # Ortamı deterministik hale getir (SCIP iç rastgelelikleri + bizim expert seçimimiz).
    env.seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)
    rng.shuffle(files)  # Shuffle once at start for randomness

    # Resume: mevcut sample'lardan devam et
    existing_samples = list(out_dir.glob("sample_*.pkl"))
    inst_sample_counts = {path: 0 for path in files}
    capped_instances: set[str] = set()
    if resume and existing_samples:
        i = len(existing_samples)
        print(f"[{split_name}] Resuming from {i} existing samples...")
        if max_samples_per_instance is not None:
            for sample_path in existing_samples:
                try:
                    with open(sample_path, "rb") as fh:
                        sample = pickle.load(fh)
                    inst = sample.get("instance")
                    if inst in inst_sample_counts:
                        inst_sample_counts[inst] += 1
                except Exception as exc:
                    print(f"[warn] Failed to read {sample_path.name}: {exc}")
            capped_instances = {
                inst for inst, count in inst_sample_counts.items()
                if count >= max_samples_per_instance
            }
    else:
        i = 0
    
    episode = 0
    file_idx = 0  # Round-robin index to cycle through all instances
    
    while i < n_samples:
        if max_samples_per_instance is not None and len(capped_instances) == len(files):
            print(f"[{split_name}] All instances reached the per-instance cap; stopping early.")
            break
        # Round-robin: cycle through all instances to ensure diversity
        instance = files[file_idx % len(files)]
        file_idx += 1
        if max_samples_per_instance is not None and inst_sample_counts[instance] >= max_samples_per_instance:
            capped_instances.add(instance)
            continue
        
        ep_label = split_name or "split"
        inst_name = instance.split('/')[-1]
        print(f"[{ep_label}] Episode {episode}, {i} samples collected so far (instance: {inst_name})")
        
        # Bozuk LP dosyalarını atla
        try:
            obs_dict, action_set, _, done, info = env.reset(str(instance))
        except Exception as e:
            print(f"[skip] Error loading {inst_name}: {e}")
            episode += 1
            continue
        
        ep_samples = 0
        while not done and i < n_samples:
            scores, scores_are_expert, node_observation = obs_dict
            if max_samples_per_instance is not None and inst_sample_counts[instance] >= max_samples_per_instance:
                capped_instances.add(instance)
                break
            if action_set is None or len(action_set) == 0:
                print(f"[warn] Empty action_set in {inst_name}; ending episode early.")
                break
            if scores is None:
                print(f"[warn] Missing scores in {inst_name}; ending episode early.")
                break
            action_set = np.asarray(action_set, dtype=np.int64)
            scores = np.asarray(scores)
            # pick best score (scores aligned with action_set)
            # Filter NaN scores: only consider valid scores for expert selection
            candidate_scores = scores[action_set]
            valid_mask = ~np.isnan(candidate_scores)
            if valid_mask.sum() == 0:
                # All scores are NaN, skip this sample
                action = action_set[0]
            else:
                # Find best among valid scores
                valid_scores = candidate_scores[valid_mask]
                valid_indices = np.where(valid_mask)[0]
                best_valid_idx = int(valid_scores.argmax())
                best_idx = valid_indices[best_valid_idx]
                action = action_set[best_idx]

            # Only save samples if they are coming from the expert (strong branching)
            if scores_are_expert and (i < n_samples):
                if node_observation is None:
                    raise RuntimeError("Expected NodeBipartite observation when expert was queried.")
                if max_samples_per_instance is not None and inst_sample_counts[instance] >= max_samples_per_instance:
                    capped_instances.add(instance)
                    break
                if (not filter_invalid_scores) or scores_are_valid(candidate_scores):
                    # Expand scores to full variable set for consistency
                    full_scores = np.full(len(node_observation.variable_features), np.nan, dtype=np.float32)
                    full_scores[action_set] = scores[action_set]
                    sample = {
                        "episode": episode,
                        "instance": instance,
                        "observation": node_observation,
                        "action_set": action_set,
                        "expert_action": action,
                        "scores": full_scores,
                    }
                    with open(out_dir / f"sample_{i+1}.pkl", "wb") as fh:
                        pickle.dump(sample, fh)
                    i += 1
                    ep_samples += 1
                    inst_sample_counts[instance] += 1
                    if max_samples_per_instance is not None and inst_sample_counts[instance] >= max_samples_per_instance:
                        capped_instances.add(instance)
                        break

            # Eğer epizod başına örnek sınırı varsa ve dolduysa epizodu bitirip
            # yeni instance'a geçiyoruz. None ise sınır yok (tam solve).
            if max_ep_samples is not None and ep_samples >= max_ep_samples:
                break
            obs_dict, action_set, _, done, info = env.step(action)
        episode += 1
    print(f"[ok] {i} samples written to {out_dir}")


def parse_args():
    ap = argparse.ArgumentParser(description="Collect branching samples with Ecole StrongBranching expert.")
    ap.add_argument("--train-pattern", default="CO_to_Path_MILP/pp_instances_gasse/train/*.lp", help="Glob for train LPs.")
    ap.add_argument("--valid-pattern", default="CO_to_Path_MILP/pp_instances_gasse/valid/*.lp", help="Glob for valid LPs.")
    ap.add_argument("--test-pattern", default="CO_to_Path_MILP/pp_instances_gasse/test/*.lp", help="Glob for test LPs.")
    ap.add_argument("--out-dir", default="CO_to_Path_MILP/pp_samples_ecole", help="Output root directory for samples.")
    ap.add_argument("--train-samples", type=int, default=2000, help="Number of train samples.")
    ap.add_argument("--valid-samples", type=int, default=400, help="Number of valid samples.")
    ap.add_argument("--test-samples", type=int, default=400, help="Number of test samples.")
    ap.add_argument("--seed", type=int, default=0, help="Base seed (split seeds = seed, seed+1, seed+2).")
    ap.add_argument(
        "--time-limit",
        type=float,
        default=120,
        help="SCIP time limit per episode (seconds). Default 600. Use 0 to disable.",
    )
    ap.add_argument(
        "--expert-probability",
        type=float,
        default=0.05,
        help="Her düğümde StrongBranching kullanılma olasılığı (0-1).",
    )
    ap.add_argument(
        "--max-ep-samples",
        type=int,
        default=0,
        help="Bir episodedan en fazla kaç expert örneği kaydedilsin. 0 = sınır yok.",
    )
    ap.add_argument(
        "--max-samples-per-instance",
        type=int,
        default=0,
        help="Instance başına en fazla kaç expert örneği kaydedilsin. 0 = sınır yok.",
    )
    ap.add_argument(
        "--filter-invalid-scores",
        action="store_true",
        help="NaN/inf/negatif expert skorlarını filtrele (example.ipynb'de kapalı).",
    )
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help="Mevcut sample'lardan devam etme, baştan başla.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.out_dir).resolve()

    resume = not args.no_resume
    
    # Train first, then valid, then test
    collect_split(
        args.train_pattern,
        out_root / "train",
        args.train_samples,
        args.seed,
        args.time_limit,
        split_name="train",
        max_ep_samples=None if args.max_ep_samples <= 0 else args.max_ep_samples,
        max_samples_per_instance=None
        if args.max_samples_per_instance <= 0
        else args.max_samples_per_instance,
        expert_probability=args.expert_probability,
        filter_invalid_scores=args.filter_invalid_scores,
        resume=resume,
    )
    collect_split(
        args.valid_pattern,
        out_root / "valid",
        args.valid_samples,
        args.seed + 1,
        args.time_limit,
        split_name="valid",
        max_ep_samples=None if args.max_ep_samples <= 0 else args.max_ep_samples,
        max_samples_per_instance=None
        if args.max_samples_per_instance <= 0
        else args.max_samples_per_instance,
        expert_probability=args.expert_probability,
        filter_invalid_scores=args.filter_invalid_scores,
        resume=resume,
    )
    collect_split(
        args.test_pattern,
        out_root / "test",
        args.test_samples,
        args.seed + 2,
        args.time_limit,
        split_name="test",
        max_ep_samples=None if args.max_ep_samples <= 0 else args.max_ep_samples,
        max_samples_per_instance=None
        if args.max_samples_per_instance <= 0
        else args.max_samples_per_instance,
        expert_probability=args.expert_probability,
        filter_invalid_scores=args.filter_invalid_scores,
        resume=resume,
    )


if __name__ == "__main__":
    main()
