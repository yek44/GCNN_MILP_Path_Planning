"""
Train a GNN brancher on the path-planning MILP dataset.

- Compatible with LP instances built via `generate_instances_path_planning.py`
  and branching samples collected by `generate_dataset_ecole.py`.
- Mirrors the Gasse et al. (2019) bipartite GNN from `example.ipynb`, but
  tailored to our receding-horizon formulation in `true_receding_horizon.py`.
"""

import argparse
import csv
import os
import pickle
import random
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.loader import DataLoader
try:
    # GraphNorm PyG içerisinde; batch bazlı normalizasyon yapar.
    from torch_geometric.nn.norm import GraphNorm
except Exception:  # pragma: no cover
    GraphNorm = None  # type: ignore

# Ecole is only needed for unpickling NodeBipartite observations
import ecole  # noqa: F401  pylint: disable=unused-import


# ---------
# Data utils
# ---------
class BipartiteNodeData(torch_geometric.data.Data):
    """Wrap a NodeBipartite observation in a PyG-friendly container."""

    def __inc__(self, key, value, store, *args, **kwargs):
        # Tell PyG how to increment indices when batching graphs.
        if key == "edge_index":
            # edge_index[0] refers to constraint nodes, edge_index[1] to variables.
            return torch.tensor(
                [[self.constraint_features.size(0)], [self.variable_features.size(0)]]
            )
        if key == "candidates":
            # Candidate indices are over variables only.
            return self.variable_features.size(0)
        if key in ("constraint_batch", "variable_batch"):
            # Her graph kendi içinde 0 ile başlıyor; batch'ler birleşirken +1 kaydır.
            return 1
        return super().__inc__(key, value, store, *args, **kwargs)


class GraphDataset(torch_geometric.data.Dataset):
    """Load StrongBranching samples saved by `generate_dataset_ecole.py`."""

    def __init__(self, sample_files: Sequence[str]):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files: List[str] = list(sample_files)

    def len(self) -> int:
        return len(self.sample_files)

    def get(self, index: int) -> BipartiteNodeData:
        with open(self.sample_files[index], "rb") as f:
            sample = pickle.load(f)

        obs = sample["observation"]
        action_set = np.asarray(sample["action_set"], dtype=np.int64)
        expert_action = int(sample["expert_action"])
        scores = np.asarray(sample["scores"]).reshape(-1)

        # Features from NodeBipartite
        constraint_features = np.nan_to_num(np.asarray(obs.row_features, dtype=np.float32))
        if hasattr(obs, "variable_features"):
            variable_features = np.asarray(obs.variable_features, dtype=np.float32)
        elif hasattr(obs, "col_features"):
            variable_features = np.asarray(obs.col_features, dtype=np.float32)
        else:
            raise AttributeError("NodeBipartite observation missing variable features.")
        variable_features = np.nan_to_num(variable_features)
        edge_indices = np.asarray(obs.edge_features.indices, dtype=np.int64)
        edge_features = np.nan_to_num(np.asarray(obs.edge_features.values, dtype=np.float32)).reshape(
            -1, 1
        )

        n_vars = variable_features.shape[0]
        if action_set.max(initial=-1) >= n_vars:
            raise ValueError(
                f"Candidate index exceeds variable count in {self.sample_files[index]}"
            )

        # Scores sometimes cover the full variable set; align to candidates if needed.
        if scores.shape[0] == action_set.shape[0]:
            candidate_scores = scores
        elif scores.shape[0] >= n_vars:
            candidate_scores = scores[action_set]
        else:
            raise ValueError(
                f"Unexpected score shape {scores.shape} for {self.sample_files[index]}"
            )
        candidate_scores = np.nan_to_num(
            candidate_scores, nan=-1e8, posinf=-1e8, neginf=-1e8
        )

        # Expert choice index within the candidate list; fall back to best score on mismatch.
        match = np.where(action_set == expert_action)[0]
        if match.size == 0:
            candidate_choice = int(np.argmax(candidate_scores))
        else:
            candidate_choice = int(match[0])

        graph = BipartiteNodeData(
            constraint_features=torch.tensor(constraint_features, dtype=torch.float32),
            edge_index=torch.tensor(edge_indices, dtype=torch.int64),
            edge_attr=torch.tensor(edge_features, dtype=torch.float32),
            variable_features=torch.tensor(variable_features, dtype=torch.float32),
            candidates=torch.tensor(action_set, dtype=torch.int64),
            nb_candidates=torch.tensor([len(action_set)], dtype=torch.int64),
            candidate_choices=torch.tensor([candidate_choice], dtype=torch.int64),
            candidate_scores=torch.tensor(candidate_scores, dtype=torch.float32),
            # GraphNorm için gerekli: her düğümün hangi graph'a ait olduğu.
            constraint_batch=torch.zeros(
                constraint_features.shape[0], dtype=torch.int64
            ),
            variable_batch=torch.zeros(
                variable_features.shape[0], dtype=torch.int64
            ),
        )

        # PyG uses this to size the combined bipartite graph.
        graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]
        return graph


def pad_tensor(input_: torch.Tensor, pad_sizes: torch.Tensor, pad_value: float = -1e8) -> torch.Tensor:
    """Pad a stacked tensor to max candidate size across the batch."""
    max_pad = int(pad_sizes.max().item())
    # Split by graph, pad each slice, then stack.
    output = input_.split(pad_sizes.cpu().tolist())
    output = torch.stack(
        [
            F.pad(slice_, (0, max_pad - slice_.size(0)), "constant", pad_value)
            for slice_ in output
        ],
        dim=0,
    )
    return output


# --------------
# Model definition
# --------------
class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """Half-convolution for constraint/variable message passing."""

    def __init__(self):
        super().__init__(aggr="add")
        emb_size = 64

        self.feature_module_left = torch.nn.Linear(emb_size, emb_size)
        self.feature_module_edge = torch.nn.Linear(1, emb_size, bias=False)
        self.feature_module_right = torch.nn.Linear(emb_size, emb_size, bias=False)
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )
        self.post_conv_module = torch.nn.Sequential(torch.nn.LayerNorm(emb_size))
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(
        self,
        left_features: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_features: torch.Tensor,
        right_features: torch.Tensor,
    ) -> torch.Tensor:
        messages = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )
        return self.output_module(
            torch.cat([self.post_conv_module(messages), right_features], dim=-1)
        )

    def message(self, node_features_i, node_features_j, edge_features):
        return self.feature_module_final(
            self.feature_module_left(node_features_i)
            + self.feature_module_edge(edge_features)
            + self.feature_module_right(node_features_j)
        )


class GNNPolicy(torch.nn.Module):
    """Bipartite GNN from the Ecole example notebook (Gasse et al.)."""

    def __init__(self, num_conv_layers: int = 1, norm: str = "layer"):
        super().__init__()
        emb_size = 64
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 19
        self.num_conv_layers = max(1, int(num_conv_layers))

        # Normalizasyon tipi: "layer" (eski davranış) veya "graph" (hocanın önerisi).
        use_graph_norm = norm.lower().startswith("graph") and GraphNorm is not None
        self.use_graph_norm = use_graph_norm
        ConsNorm = GraphNorm if use_graph_norm else torch.nn.LayerNorm
        VarNorm = GraphNorm if use_graph_norm else torch.nn.LayerNorm

        # Ön embedding: ham NodeBipartite özelliklerini 64-boyutlu ortak uzaya taşır.
        self.cons_norm_in = ConsNorm(cons_nfeats)
        self.var_norm_in = VarNorm(var_nfeats)
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )
        # Edge'ler için GraphNorm anlamlı değil; LayerNorm bırakıyoruz.
        self.edge_embedding = torch.nn.Sequential(torch.nn.LayerNorm(edge_nfeats))
        self.var_embedding = torch.nn.Sequential(
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # Her katmanda iki "yarım" konvolüsyon var: v->c ve c->v.
        self.conv_v_to_c = torch.nn.ModuleList(
            [BipartiteGraphConvolution() for _ in range(self.num_conv_layers)]
        )
        self.conv_c_to_v = torch.nn.ModuleList(
            [BipartiteGraphConvolution() for _ in range(self.num_conv_layers)]
        )
        # Katman sonrası normalizasyon (GraphNorm veya LayerNorm).
        self.cons_norms = torch.nn.ModuleList(
            [ConsNorm(emb_size) for _ in range(self.num_conv_layers)]
        )
        self.var_norms = torch.nn.ModuleList(
            [VarNorm(emb_size) for _ in range(self.num_conv_layers)]
        )
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(
        self,
        constraint_features: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_features: torch.Tensor,
        variable_features: torch.Tensor,
        constraint_batch: torch.Tensor | None = None,
        variable_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Reverse edge direction for constraint -> variable convolution.
        reversed_edges = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # Giriş norm + embedding
        if self.use_graph_norm:
            constraint_features = self.cons_norm_in(constraint_features, constraint_batch)
            variable_features = self.var_norm_in(variable_features, variable_batch)
        else:
            constraint_features = self.cons_norm_in(constraint_features)
            variable_features = self.var_norm_in(variable_features)

        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        # Mesaj geçiş katmanları (L defa tekrar).
        for layer in range(self.num_conv_layers):
            constraint_features = self.conv_v_to_c[layer](
                variable_features, reversed_edges, edge_features, constraint_features
            )
            if self.use_graph_norm:
                constraint_features = self.cons_norms[layer](constraint_features, constraint_batch)
            else:
                constraint_features = self.cons_norms[layer](constraint_features)

            variable_features = self.conv_c_to_v[layer](
                constraint_features, edge_indices, edge_features, variable_features
            )
            if self.use_graph_norm:
                variable_features = self.var_norms[layer](variable_features, variable_batch)
            else:
                variable_features = self.var_norms[layer](variable_features)

        return self.output_module(variable_features).squeeze(-1)


# --------------
# Training / eval
# --------------
def process_epoch(
    policy: GNNPolicy,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    loss_mode: str = "tie",
    tie_eps: float = 0.0,
    top_k: Sequence[int] | None = None,
) -> tuple[float, float]:
    """Run one training or evaluation epoch."""
    mean_loss = 0.0
    mean_acc = 0.0
    if top_k is None:
        top_k = []
    top_k = sorted({int(k) for k in top_k if int(k) > 0})
    topk_sums = {k: 0.0 for k in top_k}
    n_samples_processed = 0

    grad_enabled = optimizer is not None
    with torch.set_grad_enabled(grad_enabled):
        for batch in data_loader:
            batch = batch.to(device)
            logits = policy(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
                batch.constraint_batch,
                batch.variable_batch,
            )

            logits = pad_tensor(logits[batch.candidates], batch.nb_candidates)
            if loss_mode == "ce":
                # Klasik öğrenme: tek bir expert_action etiketini hedefle.
                loss = F.cross_entropy(logits, batch.candidate_choices)
            elif loss_mode == "tie":
                # Tie-aware (set-based) öğrenme:
                # Eğer birden fazla aday aynı en-iyi score'a sahipse, hepsini "doğru" say.
                true_scores = pad_tensor(batch.candidate_scores, batch.nb_candidates)
                true_best = true_scores.max(dim=-1, keepdim=True).values

                # Pad edilen skorlar -1e8 civarında; bunları maske dışı bırak.
                valid_mask = true_scores > -1e7
                tie_mask = (true_scores >= (true_best - tie_eps)) & valid_mask

                targets = tie_mask.float()
                targets = targets / targets.sum(dim=-1, keepdim=True).clamp_min(1.0)
                loss = -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            elif loss_mode == "kl":
                # KL: expert skor dağılımını hedefle (tie’leri ve “birden fazla iyi aday”ı doğal modeller).
                true_scores = pad_tensor(batch.candidate_scores, batch.nb_candidates)
                valid_mask = true_scores > -1e7
                true_scores_safe = true_scores.clone()
                true_scores_safe[~valid_mask] = float("-inf")
                target_probs = F.softmax(true_scores_safe.float(), dim=-1)
                target_probs = (target_probs + 1e-8) / (target_probs + 1e-8).sum(dim=-1, keepdim=True)
                log_model = F.log_softmax(logits, dim=-1)
                loss = F.kl_div(log_model, target_probs, reduction="batchmean")
            else:
                raise ValueError(f"Unknown loss_mode: {loss_mode}")

            if grad_enabled:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Accuracy: notebook'taki gibi "tie" durumlarını doğru say (en iyi skor birden fazla olabilir).
            true_scores = pad_tensor(batch.candidate_scores, batch.nb_candidates)
            true_bestscore = true_scores.max(dim=-1, keepdims=True).values
            predicted_bestindex = logits.max(dim=-1, keepdims=True).indices
            accuracy = (
                (true_scores.gather(-1, predicted_bestindex) == true_bestscore)
                .float()
                .mean()
                .item()
            )

            # Top-k accuracy (learn2branch tarzı): en iyi skor top-k içinde mi?
            for k in top_k:
                pred_topk = logits.topk(k, dim=-1).indices
                pred_topk_scores = true_scores.gather(-1, pred_topk)
                kacc = (pred_topk_scores == true_bestscore).any(dim=-1).float().mean().item()
                topk_sums[k] += kacc * batch.num_graphs

            mean_loss += loss.item() * batch.num_graphs
            mean_acc += accuracy * batch.num_graphs
            n_samples_processed += batch.num_graphs

    mean_loss /= n_samples_processed
    mean_acc /= n_samples_processed
    mean_topk = {k: (topk_sums[k] / n_samples_processed) for k in top_k}
    return mean_loss, mean_acc, mean_topk


# -----
# Helpers
# -----
def list_samples(folder: Path, limit: int | None = None) -> List[str]:
    files = sorted(folder.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No samples found in {folder}")
    if limit is not None:
        files = files[:limit]
    return [str(f) for f in files]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----
# CLI
# -----
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a GNN brancher on path-planning MILP samples (Ecole StrongBranching)."
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default="CO_to_Path_MILP/pp_samples_ecole/train",
        help="Directory with training pickles.",
    )
    parser.add_argument(
        "--valid-dir",
        type=str,
        default="CO_to_Path_MILP/pp_samples_ecole/valid",
        help="Directory with validation pickles.",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="CO_to_Path_MILP/pp_samples_ecole/test",
        help="Directory with test pickles.",
    )
    parser.add_argument("--limit-train", type=int, default=None, help="Optional cap on train samples.")
    parser.add_argument("--limit-valid", type=int, default=None, help="Optional cap on valid samples.")
    parser.add_argument("--limit-test", type=int, default=None, help="Optional cap on test samples.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--patience", type=int, default=10, help="LR decay patience (epochs without improvement before reducing LR).")
    parser.add_argument("--early-stopping", type=int, default=20, help="Early stopping patience (epochs without improvement before stopping).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--lr-decay", type=float, default=0.2, help="LR decay factor when patience is reached (learn2branch uses 0.2).")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument(
        "--epoch-size",
        type=int,
        default=312,
        help="Number of batches per epoch (learn2branch uses 312; set 0 to use all data).",
    )
    parser.add_argument(
        "--num-conv-layers",
        type=int,
        default=1,
        help="GNN'de kaç mesaj-geçiş katmanı olsun (1 = orijinal Gasse).",
    )
    parser.add_argument(
        "--norm",
        type=str,
        default="graph",
        choices=["layer", "graph"],
        help="Normalizasyon tipi: layer veya graph.",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="tie",
        choices=["ce", "tie", "kl"],
        help="Kayıp tipi: ce, tie veya kl (KL divergence, expert skor dağılımı).",
    )
    parser.add_argument(
        "--tie-eps",
        type=float,
        default=0.0,
        help="Tie loss için tolerans: skor >= (en_iyi - eps) ise 'doğru'.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="CO_to_Path_MILP/gnn_policy_path_planning_ep6.pt",
        help="Where to store the trained weights.",
    )
    parser.add_argument(
        "--log-csv",
        type=str,
        default="CO_to_Path_MILP/training_log.csv",
        help="CSV file to log epoch-by-epoch metrics.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_files = list_samples(Path(args.train_dir), args.limit_train)
    valid_files = list_samples(Path(args.valid_dir), args.limit_valid)
    test_files = list_samples(Path(args.test_dir), args.limit_test)
    
    print(f"Loaded {len(train_files)} train, {len(valid_files)} valid, {len(test_files)} test samples.")

    # Validation loader (sabit, her epoch'ta aynı)
    valid_loader = DataLoader(GraphDataset(valid_files), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(GraphDataset(test_files), batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)
    policy = GNNPolicy(num_conv_layers=args.num_conv_layers, norm=args.norm).to(device)
    
    # Dinamik LR için current_lr kullan (learn2branch tarzı)
    current_lr = args.lr
    optimizer = torch.optim.Adam(
        policy.parameters(), lr=current_lr, weight_decay=args.weight_decay
    )

    # Early stopping and best model tracking (learn2branch: loss'a göre takip)
    best_valid_loss = float('inf')
    best_valid_acc = 0.0
    best_epoch = 0
    best_state_dict = None
    plateau_count = 0  # learn2branch naming

    # Hyperparameters log
    print(f"Hyperparameters:")
    print(f"  batch_size: {args.batch_size}")
    print(f"  epoch_size: {args.epoch_size if args.epoch_size > 0 else 'all'}")
    print(f"  lr: {args.lr}")
    print(f"  lr_decay: {args.lr_decay}")
    print(f"  patience: {args.patience}")
    print(f"  early_stopping: {args.early_stopping}")
    print(f"  weight_decay: {args.weight_decay}")
    print(f"  num_conv_layers: {args.num_conv_layers}")
    print(f"  norm: {args.norm}")
    print()

    # CSV logging setup
    csv_path = Path(args.log_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    top_k = [1, 3, 5, 10]
    csv_header = ["epoch", "train_loss", "train_acc", "valid_loss", "valid_acc"]
    csv_header += [f"train_acc@{k}" for k in top_k]
    csv_header += [f"valid_acc@{k}" for k in top_k]
    csv_header += ["lr", "is_best"]
    csv_writer.writerow(csv_header)

    # RNG for epoch sampling (learn2branch tarzı)
    rng = np.random.RandomState(args.seed)
    for epoch in range(1, args.epochs + 1):
        # Epoch size: her epoch'ta belirli sayıda sample kullan (learn2branch tarzı)
        if args.epoch_size > 0:
            # Her epoch'ta rastgele epoch_size * batch_size sample seç
            n_samples = args.epoch_size * args.batch_size
            epoch_indices = rng.choice(len(train_files), min(n_samples, len(train_files)), replace=True)
            epoch_train_files = [train_files[i] for i in epoch_indices]
            train_loader = DataLoader(
                GraphDataset(epoch_train_files), 
                batch_size=args.batch_size, 
                shuffle=True
            )
        else:
            # Tüm train data'yı kullan
            train_loader = DataLoader(
                GraphDataset(train_files), 
                batch_size=args.batch_size, 
                shuffle=True
            )

        train_loss, train_acc, train_topk = process_epoch(
            policy,
            train_loader,
            optimizer,
            device,
            loss_mode=args.loss,
            tie_eps=args.tie_eps,
            top_k=top_k,
        )
        valid_loss, valid_acc, valid_topk = process_epoch(
            policy,
            valid_loader,
            None,
            device,
            loss_mode=args.loss,
            tie_eps=args.tie_eps,
            top_k=top_k,
        )
        
        # Track best model (learn2branch: loss'a göre)
        improved = valid_loss < best_valid_loss
        if improved:
            plateau_count = 0
            best_valid_loss = valid_loss
            best_valid_acc = valid_acc
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in policy.state_dict().items()}
        else:
            plateau_count += 1
            
            # Learning rate scheduling (learn2branch tarzı)
            if plateau_count % args.patience == 0 and plateau_count > 0:
                current_lr *= args.lr_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                print(f"  {plateau_count} epochs without improvement, decreasing learning rate to {current_lr:.6f}")
            
            # Early stopping check (learn2branch tarzı)
            if plateau_count >= args.early_stopping:
                print(f"Early stopping at epoch {epoch} ({plateau_count} epochs without improvement)")
                break
        
        topk_msg = " ".join(
            [f"acc@{k} {train_topk.get(k, 0.0):.3f}/{valid_topk.get(k, 0.0):.3f}" for k in top_k]
        )
        print(
            f"[epoch {epoch:03d}] "
            f"train loss {train_loss:.4f}, acc {train_acc:.3f} | "
            f"valid loss {valid_loss:.4f}, acc {valid_acc:.3f} | "
            f"{topk_msg} | lr {current_lr:.6f}"
            f"{' *' if improved else ''}"
        )
        
        # Write to CSV
        csv_row = [epoch, train_loss, train_acc, valid_loss, valid_acc]
        csv_row += [train_topk.get(k, 0.0) for k in top_k]
        csv_row += [valid_topk.get(k, 0.0) for k in top_k]
        csv_row += [current_lr, improved]
        csv_writer.writerow(csv_row)
        csv_file.flush()  # Ensure data is written immediately

    # Close CSV file
    csv_file.close()
    print(f"Training log saved to {csv_path.resolve()}")

    # Restore best model for final test evaluation
    if best_state_dict is not None:
        policy.load_state_dict(best_state_dict)
        print(f"Restored best model from epoch {best_epoch} (valid loss: {best_valid_loss:.4f}, acc: {best_valid_acc:.3f})")

    test_loss, test_acc, _ = process_epoch(
        policy,
        test_loader,
        None,
        device,
        loss_mode=args.loss,
        tie_eps=args.tie_eps,
        top_k=None,
    )
    print(f"[test] loss {test_loss:.4f}, acc {test_acc:.3f}")

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "args": vars(args),
            "best_epoch": best_epoch,
            "best_valid_loss": best_valid_loss,
            "best_valid_acc": best_valid_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        },
        save_path,
    )
    print(f"Model saved to {save_path.resolve()}")


if __name__ == "__main__":
    main()
