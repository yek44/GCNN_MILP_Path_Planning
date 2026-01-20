"""
Eğitilmiş GNN branching policy'sini test LP'lerinde değerlendir.

Amaç:
- Sadece "accuracy" değil, gerçek çözüm performansını ölçmek:
  - Toplam B&B node sayısı (NNodes)
  - Çözüm zamanı (SolvingTime)

Bu script, Ecole'un resmi branching-imitation örneğindeki evaluation hücrelerinin
aynı mantığını takip eder, sadece instance'lar SetCover generator yerine
diskteki .lp dosyalarımızdan gelir.
"""

from __future__ import annotations

import argparse
import csv
import glob
import random
from pathlib import Path

import numpy as np

# Bu scripti çalıştırdığınız environment'ta torch ve ecole kurulu olmalı.
import torch
import ecole

from train_gnn_path_planning import GNNPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained GNN brancher vs SCIP default on LP instances."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="CO_to_Path_MILP/gnn_policy_path_planning.pt",
        help="train_gnn_path_planning.py tarafından kaydedilmiş .pt dosyası.",
    )
    parser.add_argument(
        "--instances-pattern",
        type=str,
        default="CO_to_Path_MILP/pp_instances_gasse/test/*.lp",
        help="Değerlendirilecek LP dosyaları (glob).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Kaç instance üzerinde ölçelim? (0 veya negatif = hepsi)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=True,
        help="Instance listesini karıştır (seed ile deterministik).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Ecole/SCIP seedi (instance başına seed+i).",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=3600.0,
        help="Her instance için time limit (s). 0 = limitsiz.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda / cpu",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Özet metrikleri en sonda yazdır.",
    )
    parser.add_argument(
        "--log-csv",
        type=str,
        default="CO_to_Path_MILP/evaluation_results.csv",
        help="CSV dosyasına sonuçları kaydet.",
    )
    return parser.parse_args()


def make_branching_env(scip_params: dict) -> ecole.environment.Branching:
    info_fns = {
        "nb_nodes": ecole.reward.NNodes(),
        "time": ecole.reward.SolvingTime(),
    }
    try:
        return ecole.environment.Branching(
            observation_function=ecole.observation.NodeBipartite(),
            information_function=info_fns,
            scip_params=scip_params,
        )
    except TypeError:
        # Bazı Ecole sürümlerinde reward_function argümanı zorunlu olabiliyor.
        return ecole.environment.Branching(
            observation_function=ecole.observation.NodeBipartite(),
            reward_function=ecole.reward.NNodes(),
            information_function=info_fns,
            scip_params=scip_params,
        )


def make_default_env(scip_params: dict) -> ecole.environment.Configuring:
    info_fns = {
        "nb_nodes": ecole.reward.NNodes(),
        "time": ecole.reward.SolvingTime(),
    }
    try:
        return ecole.environment.Configuring(
            observation_function=None,
            information_function=info_fns,
            scip_params=scip_params,
        )
    except TypeError:
        # Bazı Ecole sürümlerinde reward_function argümanı zorunlu olabiliyor.
        return ecole.environment.Configuring(
            observation_function=None,
            reward_function=ecole.reward.NNodes(),
            information_function=info_fns,
            scip_params=scip_params,
        )


def load_policy(model_path: Path, device: torch.device) -> GNNPolicy:
    # PyTorch 2.1+ güvenlik uyarısı: `torch.load` pickle kullandığı için, sadece
    # kendi ürettiğiniz model dosyalarını yükleyin. Burada mümkünse
    # `weights_only=True` ile daha güvenli modu kullanıyoruz.
    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(model_path, map_location="cpu")
    except Exception:
        # Bazı checkpoint formatlarında weights_only kısıtı sorun çıkarabilir.
        ckpt = torch.load(model_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        saved_args = ckpt.get("args", {}) or {}
        num_conv_layers = int(saved_args.get("num_conv_layers", 1))
        norm = str(saved_args.get("norm", "layer"))
    else:
        # Eski format: direkt state_dict kaydedilmiş olabilir.
        state_dict = ckpt
        num_conv_layers = 1
        norm = "layer"

    policy = GNNPolicy(num_conv_layers=num_conv_layers, norm=norm).to(device)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


@torch.no_grad()
def select_action(
    policy: GNNPolicy,
    observation: ecole.observation.NodeBipartite,
    action_set: np.ndarray,
    device: torch.device,
) -> int:
    # NodeBipartite observation -> torch tensor
    constraint_features = torch.from_numpy(observation.row_features.astype(np.float32)).to(device)
    edge_index = torch.from_numpy(observation.edge_features.indices.astype(np.int64)).to(device)
    edge_attr = (
        torch.from_numpy(observation.edge_features.values.astype(np.float32)).view(-1, 1).to(device)
    )
    variable_features = torch.from_numpy(observation.variable_features.astype(np.float32)).to(device)

    # GraphNorm seçiliyse policy forward içinde batch vektörleri beklenir.
    constraint_batch = torch.zeros(constraint_features.size(0), dtype=torch.int64, device=device)
    variable_batch = torch.zeros(variable_features.size(0), dtype=torch.int64, device=device)

    logits = policy(
        constraint_features,
        edge_index,
        edge_attr,
        variable_features,
        constraint_batch,
        variable_batch,
    )

    candidates = torch.as_tensor(action_set, dtype=torch.int64, device=device)
    best_idx = int(logits[candidates].argmax().item())
    return int(action_set[best_idx])


def main() -> None:
    args = parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    instance_files = sorted(glob.glob(args.instances_pattern))
    if not instance_files:
        raise FileNotFoundError(f"No LP files matched: {args.instances_pattern}")

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(instance_files)

    if args.limit and args.limit > 0:
        instance_files = instance_files[: args.limit]

    device = torch.device(args.device)
    policy = load_policy(model_path, device)

    # Notebook'taki evaluation ayarları:
    # - presolve/separating'i azalt: daha hızlı ve daha stabil karşılaştırma
    scip_seed = int(args.seed) % 2147483648
    scip_params = {
        "display/verblevel": 0,
        "separating/maxrounds": 0,
        "presolving/maxrestarts": 0,
        "randomization/permutevars": True,
        "randomization/permutationseed": scip_seed,
        "randomization/randomseedshift": scip_seed,
        "timing/clocktype": 1,
    }
    if args.time_limit > 0:
        scip_params["limits/time"] = float(args.time_limit)

    env = make_branching_env(scip_params)
    default_env = make_default_env(scip_params)

    policy_nodes: list[float] = []
    policy_times: list[float] = []
    default_nodes: list[float] = []
    default_times: list[float] = []

    # CSV logging setup
    csv_path = Path(args.log_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "instance_idx", "instance_path", 
        "scip_nodes", "scip_time", 
        "gnn_nodes", "gnn_time",
        "gain_nodes_pct", "gain_time_pct"
    ])

    for i, lp_path in enumerate(instance_files):
        # Her instance'ı aynı seed ile (policy vs default) kıyaslamak önemli.
        env.seed(args.seed + i)
        default_env.seed(args.seed + i)

        # --- GNN brancher ---
        gnn_nodes = 0.0
        gnn_time = 0.0
        observation, action_set, _, done, info = env.reset(lp_path)
        gnn_nodes += float(info["nb_nodes"])
        gnn_time += float(info["time"])
        if action_set is None:
            action_set = np.empty((0,), dtype=np.int64)
        else:
            action_set = np.asarray(action_set, dtype=np.int64)

        while not done:
            if action_set.size == 0:
                print("  [warn] action_set empty/None; ending early.", flush=True)
                break
            action = select_action(policy, observation, action_set, device)
            observation, action_set, _, done, info = env.step(action)
            gnn_nodes += float(info["nb_nodes"])
            gnn_time += float(info["time"])
            if action_set is None:
                action_set = np.empty((0,), dtype=np.int64)
            else:
                action_set = np.asarray(action_set, dtype=np.int64)

        # --- SCIP default ---
        default_env.reset(lp_path)
        _, _, _, _, base_info = default_env.step({})
        b_nodes = float(base_info["nb_nodes"])
        b_time = float(base_info["time"])

        policy_nodes.append(gnn_nodes)
        policy_times.append(gnn_time)
        default_nodes.append(b_nodes)
        default_times.append(b_time)

        gain_nodes = 100.0 * (1.0 - gnn_nodes / max(b_nodes, 1.0))
        gain_time = 100.0 * (1.0 - gnn_time / max(b_time, 1e-9))

        print(f"Instance {i: >3} | SCIP nb nodes {int(b_nodes): >5d}  | SCIP time {b_time: >6.2f}")
        print(f"             | GNN  nb nodes {int(gnn_nodes): >5d}  | GNN  time {gnn_time: >6.2f}")
        print(f"             | Gain {gain_nodes: >8.2f}% | Gain {gain_time: >8.2f}%")

        # Write to CSV
        csv_writer.writerow([
            i, str(lp_path),
            int(b_nodes), b_time,
            int(gnn_nodes), gnn_time,
            gain_nodes, gain_time
        ])
        csv_file.flush()  # Ensure data is written immediately

    # Close CSV file
    csv_file.close()
    print(f"\nEvaluation results saved to {csv_path.resolve()}")

    if args.summary:
        mean_nodes = float(np.mean(np.asarray(default_nodes, dtype=np.float64)))
        mean_time = float(np.mean(np.asarray(default_times, dtype=np.float64)))
        mean_g_nodes = float(np.mean(np.asarray(policy_nodes, dtype=np.float64)))
        mean_g_time = float(np.mean(np.asarray(policy_times, dtype=np.float64)))
        mean_gain_nodes = 100.0 * (1.0 - mean_g_nodes / max(mean_nodes, 1.0))
        mean_gain_time = 100.0 * (1.0 - mean_g_time / max(mean_time, 1e-9))
        
        print("\n--- SUMMARY (mean over instances) ---")
        print(f"SCIP nodes {mean_nodes:.2f} | time {mean_time:.2f}s")
        print(f"GNN  nodes {mean_g_nodes:.2f} | time {mean_g_time:.2f}s")
        print(f"GAIN nodes {mean_gain_nodes:.2f}% | time {mean_gain_time:.2f}%")
        
        # Append summary to CSV
        csv_file = open(csv_path, "a", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([])  # Empty row
        csv_writer.writerow(["SUMMARY", "", "", "", "", "", "", ""])
        csv_writer.writerow([
            "mean", "",
            mean_nodes, mean_time,
            mean_g_nodes, mean_g_time,
            mean_gain_nodes, mean_gain_time
        ])
        csv_file.close()


if __name__ == "__main__":
    main()
