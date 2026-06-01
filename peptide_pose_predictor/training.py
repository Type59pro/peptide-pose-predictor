import argparse
import bisect
import csv
import math
import os
import json
import time
import warnings
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, Subset, random_split
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# ───────────── Parameters ──────────────
DATA_PT = "data_r3.pt"
BATCH_SIZE = 32
LR = 3e-5
MAX_EPOCHS = 400
EARLY_PATIENCE = 25

NUM_LAYERS = 3
HIDDEN_DIM = 256
ROT_K = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────── Custom EGNN Layer (PyG compatible, no coordinate updates) ──────────────
from torch_geometric.nn import MessagePassing

class EGNNLayer(MessagePassing):
    def __init__(self, in_dim, edge_dim, out_dim):
        super().__init__(aggr='add', node_dim=0)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * in_dim + edge_dim + 1, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU()
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x, pos, edge_index, edge_attr):
        # pos is not updated, only used to calculate relative distance
        return self.propagate(edge_index, x=x, pos=pos, edge_attr=edge_attr)

    def message(self, x_i, x_j, pos_i, pos_j, edge_attr):
        rel_pos = pos_j - pos_i
        dist2 = (rel_pos ** 2).sum(dim=-1, keepdim=True)  # squared distance
        input_feats = torch.cat([x_i, x_j, edge_attr, dist2], dim=-1)
        return self.edge_mlp(input_feats)

    def update(self, aggr_out, x):
        return self.node_mlp(torch.cat([x, aggr_out], dim=-1))

# ───────────── Model Definition ──────────────
class EGNNModel(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        num_layers,
        hidden_dim,
        out_dim=1,
        drop=0.1,
        output_init=None,
        pooling="mean_max_add",
        residual=False,
        layer_norm=False,
        hidden_dims=None,
    ):
        super().__init__()
        self.pooling = pooling
        self.residual = residual
        if hidden_dims is None:
            hidden_dims = [hidden_dim] * num_layers
        if len(hidden_dims) != num_layers:
            raise ValueError("hidden_dims length must match num_layers")
        self.hidden_dims = hidden_dims
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(drop)
        )
        layer_in_dims = [hidden_dims[0]] + hidden_dims[:-1]
        self.egnn_layers = nn.ModuleList([
            EGNNLayer(in_dim, edge_dim if edge_dim > 0 else 0, out_dim)
            for in_dim, out_dim in zip(layer_in_dims, hidden_dims)
        ])
        self.residual_projections = nn.ModuleList([
            nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
            for in_dim, out_dim in zip(layer_in_dims, hidden_dims)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(out_dim)
            for out_dim in hidden_dims
        ]) if layer_norm else None

        final_dim = hidden_dims[-1]
        readout_dim = hidden_dim
        pooling_dim = {
            "mean_max_add": final_dim * 3,
            "mean_max": final_dim * 2,
            "mean": final_dim,
        }[pooling]

        self.readout = nn.Sequential(
            nn.Linear(pooling_dim, readout_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(readout_dim, readout_dim // 2),
            nn.ReLU(),
            nn.Linear(readout_dim // 2, out_dim),
            nn.Softplus()
        )
        if output_init is not None:
            self.init_output(output_init)

    def init_output(self, output_init):
        output_init = max(float(output_init), 1e-6)
        final_linear = self.readout[-2]
        nn.init.zeros_(final_linear.weight)
        nn.init.constant_(final_linear.bias, math.log(math.expm1(output_init)))

    def forward(self, x, pos, edge_index, edge_attr, batch):
        h = self.node_encoder(x)
        coords = pos  # Not updated, passed directly

        for idx, layer in enumerate(self.egnn_layers):
            h_next = layer(h, coords, edge_index, edge_attr)
            h = self.residual_projections[idx](h) + h_next if self.residual else h_next
            if self.layer_norms is not None:
                h = self.layer_norms[idx](h)

        if self.pooling == "mean_max_add":
            pooled = torch.cat([
                global_mean_pool(h, batch),
                global_max_pool(h, batch),
                global_add_pool(h, batch)
            ], dim=-1)
        elif self.pooling == "mean_max":
            pooled = torch.cat([
                global_mean_pool(h, batch),
                global_max_pool(h, batch),
            ], dim=-1)
        else:
            pooled = global_mean_pool(h, batch)

        return self.readout(pooled)

# ───────────── Augmentation: Random Rigid Transform ──────────────
def random_rigid_transform(pos, batch, translate_scale=0.0):
    pos = pos.clone()
    for g_id in torch.unique(batch):
        sel = (batch == g_id)
        theta = torch.rand(3, device=pos.device) * 2 * math.pi
        c, s = torch.cos(theta), torch.sin(theta)
        R_x = torch.tensor([[1, 0, 0], [0, c[0], -s[0]], [0, s[0], c[0]]], device=pos.device)
        R_y = torch.tensor([[c[1], 0, s[1]], [0, 1, 0], [-s[1], 0, c[1]]], device=pos.device)
        R_z = torch.tensor([[c[2], -s[2], 0], [s[2], c[2], 0], [0, 0, 1]], device=pos.device)
        R = R_z @ R_y @ R_x
        pos[sel] = (R @ pos[sel].T).T
        if translate_scale > 0:
            translation = (torch.rand(3, device=pos.device) * 2 - 1) * translate_scale
            pos[sel] = pos[sel] + translation
    return pos

# ───────────── Dataset ──────────────
def load_pt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class PtDataset(Dataset):
    def __init__(self, pt_path):
        super().__init__()
        self.pt_path = Path(pt_path)
        if not self.pt_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.pt_path}")

        if self.pt_path.is_dir():
            metadata_path = self.pt_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as handle:
                    metadata = json.load(handle)
                self.shard_paths = [self.pt_path / name for name in metadata["shards"]]
                total_graphs = metadata.get("total_graphs")
                shard_size = metadata.get("shard_size")
            else:
                self.shard_paths = sorted(self.pt_path.glob("*.pt"))
                total_graphs = None
                shard_size = None
        else:
            self.shard_paths = [self.pt_path]
            total_graphs = None
            shard_size = None

        if not self.shard_paths:
            raise FileNotFoundError(f"No .pt shards found in {self.pt_path}")

        first_blob = load_pt(self.shard_paths[0])
        first_graph = first_blob["data_list"][0] if first_blob["data_list"] else None
        if total_graphs is not None and shard_size is not None:
            self.shard_lengths = [shard_size] * len(self.shard_paths)
            last_shard_length = total_graphs - shard_size * (len(self.shard_paths) - 1)
            self.shard_lengths[-1] = last_shard_length
        else:
            self.shard_lengths = []
            for shard_path in self.shard_paths:
                blob = load_pt(shard_path)
                self.shard_lengths.append(len(blob["data_list"]))

        self.cumulative_lengths = []
        total = 0
        for shard_length in self.shard_lengths:
            total += shard_length
            self.cumulative_lengths.append(total)

        if first_graph is None:
            raise ValueError(f"No graphs found in {self.pt_path}")

        self._cache_shard_index = None
        self._cache_blob = None
        self.is_sharded = len(self.shard_paths) > 1
        self.node_dim = first_graph.x.size(1) - 3
        self.edge_dim = first_graph.edge_attr.size(1) if first_graph.edge_attr is not None else 0
        self.out_dim = 1

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        if idx < 0:
            idx = len(self) + idx
        shard_index = bisect.bisect_right(self.cumulative_lengths, idx)
        shard_start = self.cumulative_lengths[shard_index - 1] if shard_index > 0 else 0
        local_index = idx - shard_start

        if self._cache_shard_index != shard_index:
            self._cache_blob = load_pt(self.shard_paths[shard_index])
            self._cache_shard_index = shard_index

        graph = self._cache_blob["data_list"][local_index].clone()
        graph.pos = graph.x[:, -3:].clone()
        graph.x = graph.x[:, :-3].clone()
        graph.y = graph.y.view(-1)[:1].clone().float()
        return graph

    def estimate_target_mean(self, max_samples=4096):
        values = []
        remaining = max_samples
        for shard_path in self.shard_paths:
            if remaining <= 0:
                break
            blob = load_pt(shard_path)
            shard_values = blob.get("target_values")
            if shard_values is None:
                shard_values = [graph.y.tolist() for graph in blob["data_list"]]
            for target in shard_values[:remaining]:
                values.append(float(target[0]))
            remaining = max_samples - len(values)
        if not values:
            raise ValueError("Cannot estimate target mean from an empty dataset.")
        return sum(values) / len(values)

# ───────────── Single Epoch Training/Validation ──────────────
def run_epoch(model, loader, opt=None, rot_k=ROT_K, translate_scale=0.0, desc=None):
    train = opt is not None
    model.train() if train else model.eval()
    total_loss, n_graph = 0., 0
    loss_fn = torch.nn.SmoothL1Loss()  # Use standard Smooth L1 Loss

    iterator = tqdm(loader, desc=desc, leave=False) if desc else loader
    for batch in iterator:
        batch = batch.to(DEVICE)
        target = batch.y.view(-1)
        k = rot_k if train else 0
        k = max(k, 1)
        loss = 0.
        for _ in range(k):
            pos_rot = random_rigid_transform(batch.pos, batch.batch, translate_scale) if (train and rot_k > 0) else batch.pos
            pred = model(batch.x, pos_rot, batch.edge_index, batch.edge_attr, batch.batch).view(-1)
            loss += loss_fn(pred, target)  # Calculate loss
        loss /= k
        if train:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        total_loss += loss.item() * batch.num_graphs
        n_graph += batch.num_graphs
    return total_loss / n_graph


def evaluate(model, loader):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            pred = model(batch.x, batch.pos, batch.edge_index, batch.edge_attr, batch.batch).view(-1)
            preds.append(pred.cpu())
            gts.append(batch.y.view(-1).cpu())
    preds = torch.cat(preds).clamp_min(0.)
    gts = torch.cat(gts)
    rmse = ((preds - gts).pow(2).mean()).sqrt().item()
    mae = (preds - gts).abs().mean().item()
    return rmse, mae


def append_metrics(metrics_file, row):
    metrics_path = Path(metrics_file)
    is_new = not metrics_path.exists()
    with metrics_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def parse_hidden_dims(hidden_dims):
    if hidden_dims is None:
        return None
    dims = [int(part.strip()) for part in hidden_dims.split(",") if part.strip()]
    if not dims or any(dim <= 0 for dim in dims):
        raise ValueError("--hidden-dims must be a comma-separated list of positive integers")
    return dims


# ───────────── Main Process ──────────────
def main():
    parser = argparse.ArgumentParser(description="Train Peptide Pose Predictor EGNN model.")
    parser.add_argument("--data", type=str, default=DATA_PT, help="Training .pt file or shard directory.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=EARLY_PATIENCE)
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default=None,
        help="Comma-separated EGNN layer widths, e.g. 256,128,64. Overrides --num-layers for message-passing widths.",
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--pooling",
        choices=["mean_max_add", "mean_max", "mean"],
        default="mean_max_add",
    )
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--layer-norm", action="store_true")
    parser.add_argument("--rot-k", type=int, default=ROT_K)
    parser.add_argument("--translate-scale", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--output-init",
        type=str,
        default="auto",
        help="Initial prediction value for the Softplus output head. Use 'auto' to estimate label mean, or 'none' to keep random init.",
    )
    parser.add_argument("--output-init-samples", type=int, default=4096)
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle training samples. Leave disabled for large sharded datasets to avoid excessive shard reloads.",
    )
    parser.add_argument("--output-model", type=str, default="best_model_egnn.pth")
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Load model weights from an existing checkpoint and continue training with a fresh optimizer.",
    )
    parser.add_argument(
        "--resume-best",
        action="store_true",
        help="Use the resumed checkpoint epoch/val as the current best for early stopping.",
    )
    parser.add_argument("--history-file", type=str, default="loss_history_egnn.json")
    parser.add_argument("--metrics-file", type=str, default="metrics_egnn.csv")
    parser.add_argument("--results-file", type=str, default="results_egnn.txt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    hidden_dims = parse_hidden_dims(args.hidden_dims)
    if hidden_dims is not None:
        args.num_layers = len(hidden_dims)

    ds = PtDataset(args.data)
    N = len(ds)
    if N < 3:
        raise ValueError("Need at least 3 graphs to create train/val/test splits.")

    n_val = max(1, int(0.1 * N))
    n_test = max(1, int(0.1 * N))
    n_train = N - n_val - n_test
    if n_train <= 0:
        raise ValueError("Dataset is too small for the configured split.")

    if ds.is_sharded:
        # Shards are already produced from parallel jobs in a mixed order. Keep access contiguous
        # so each shard is loaded once instead of repeatedly reloading large shard files.
        tr_ds = Subset(ds, range(0, n_train))
        va_ds = Subset(ds, range(n_train, n_train + n_val))
        te_ds = Subset(ds, range(n_train + n_val, N))
    else:
        tr_ds, va_ds, te_ds = random_split(ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))

    train_shuffle = args.shuffle and not ds.is_sharded
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=train_shuffle, num_workers=args.num_workers)
    va_ld = DataLoader(va_ds, batch_size=args.batch_size, num_workers=args.num_workers)
    te_ld = DataLoader(te_ds, batch_size=args.batch_size, num_workers=args.num_workers)

    if args.output_init.lower() == "auto":
        output_init = ds.estimate_target_mean(args.output_init_samples)
    elif args.output_init.lower() == "none":
        output_init = None
    else:
        output_init = float(args.output_init)

    model = EGNNModel(
        ds.node_dim,
        ds.edge_dim,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        drop=args.dropout,
        output_init=output_init,
        pooling=args.pooling,
        residual=args.residual,
        layer_norm=args.layer_norm,
        hidden_dims=hidden_dims,
    ).to(DEVICE)
    print(f"Dataset: {N} graphs | Train: {n_train} | Val: {n_val} | Test: {n_test}", flush=True)
    print(f"Device: {DEVICE} | Hidden Dim: {args.hidden_dim} | Hidden Dims: {model.hidden_dims} | Num Layers: {args.num_layers} | Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"Pooling: {args.pooling} | Residual: {args.residual} | LayerNorm: {args.layer_norm}", flush=True)
    print(f"Dropout: {args.dropout} | Output init: {output_init}", flush=True)
    print(f"Rotation augmentations per batch: {args.rot_k} | Translation scale: {args.translate_scale}", flush=True)

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    model_config = {
        "node_dim": ds.node_dim,
        "edge_dim": ds.edge_dim,
        "num_layers": args.num_layers,
        "hidden_dim": args.hidden_dim,
        "hidden_dims": model.hidden_dims,
        "dropout": args.dropout,
        "pooling": args.pooling,
        "residual": args.residual,
        "layer_norm": args.layer_norm,
    }

    start_epoch = 1
    best_val_loss, wait = float('inf'), 0
    if args.resume_checkpoint is not None:
        ckpt = torch.load(args.resume_checkpoint, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        print(
            f"Resumed model weights from {args.resume_checkpoint} "
            f"(epoch={ckpt.get('epoch', 'N/A')}, val={ckpt.get('val', 'N/A')}).",
            flush=True,
        )
        if args.resume_best:
            best_val_loss = float(ckpt["val"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            torch.save({
                "model": model.state_dict(),
                "epoch": int(ckpt.get("epoch", 0)),
                "val": best_val_loss,
                "model_config": model_config,
            }, args.output_model)

    hist = {"train": [], "val": []}
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        tr_loss = run_epoch(model, tr_ld, opt, rot_k=args.rot_k, translate_scale=args.translate_scale, desc=f"Train {epoch:03d}")

        with torch.no_grad():
            va_loss = run_epoch(model, va_ld, rot_k=0, desc=f"Val {epoch:03d}")

        sch.step()
        hist["train"].append(tr_loss)
        hist["val"].append(va_loss)

        is_best = va_loss < best_val_loss - 1e-5
        if is_best:
            best_val_loss, wait = va_loss, 0
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val": va_loss,
                "model_config": model_config,
            }, args.output_model)
        else:
            wait += 1

        epoch_seconds = time.time() - epoch_start
        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "best_val_loss": best_val_loss,
            "lr": opt.param_groups[0]["lr"],
            "is_best": int(is_best),
            "wait": wait,
            "epoch_seconds": epoch_seconds,
        }
        append_metrics(args.metrics_file, row)
        json.dump(hist, open(args.history_file, "w"), indent=2)

        print(f"Epoch {epoch:03d} | Train Loss: {tr_loss:.4f} | Val Loss: {va_loss:.4f} | Best Val: {best_val_loss:.4f} | Wait: {wait}/{args.patience} | LR: {opt.param_groups[0]['lr']:.2e} | Time: {epoch_seconds:.1f}s {'** New Best **' if is_best else ''}", flush=True)
        if wait >= args.patience:
            print(f"Early stopping after {args.patience} epochs of no improvement.", flush=True)
            break

    # ---------- Testing ----------
    print("\n--- Testing on the best model ---", flush=True)
    if not os.path.exists(args.output_model):
        print(f"Error: {args.output_model} not found. Training did not save a model.", flush=True)
        return
        
    ckpt = torch.load(args.output_model, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    rmse, mae = evaluate(model, te_ld)
    print(f"Test RMSE: {rmse:.4f} | Test MAE: {mae:.4f}", flush=True)

    # ---------- Save Results ----------
    json.dump(hist, open(args.history_file, "w"), indent=2)
    with open(args.results_file, "w") as f:
        f.write(f"Test RMSE: {rmse:.4f}\n")
        f.write(f"Test MAE: {mae:.4f}\n")
        if 'epoch' in ckpt:
            f.write(f"Best epoch: {ckpt['epoch']}\n")
        if 'val' in ckpt:
            f.write(f"Best validation loss: {ckpt['val']:.4f}\n")
    print("Results and loss history saved.", flush=True)


if __name__ == "__main__":
    main()

