import os
import math
import json
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import softmax

warnings.filterwarnings("ignore", category=UserWarning)

# ───────────── Parameters ──────────────
DATA_PT = "data_r3.pt"
BATCH_SIZE = 32
LR = 3e-4
MAX_EPOCHS = 400
EARLY_PATIENCE = 25

NUM_LAYERS = 3
HIDDEN_DIM = 256
ROT_K = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────── Custom EGNN Layer (PyG compatible, no coordinate updates) ──────────────
from torch_geometric.nn import MessagePassing

class EGNNLayer(MessagePassing):
    def __init__(self, in_dim, edge_dim, hidden_dim):
        super().__init__(aggr='add', node_dim=0)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * in_dim + edge_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim)
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
    def __init__(self, node_dim, edge_dim, num_layers, hidden_dim, out_dim=1, drop=0.1):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop)
        )
        self.egnn_layers = nn.ModuleList([
            EGNNLayer(hidden_dim, edge_dim if edge_dim > 0 else 0, hidden_dim)
            for _ in range(num_layers)
        ])

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim),
            nn.Softplus()
        )

    def forward(self, x, pos, edge_index, edge_attr, batch):
        h = self.node_encoder(x)
        coords = pos  # Not updated, passed directly

        for layer in self.egnn_layers:
            h = layer(h, coords, edge_index, edge_attr)

        pooled = torch.cat([
            global_mean_pool(h, batch),
            global_max_pool(h, batch),
            global_add_pool(h, batch)
        ], dim=-1)

        return self.readout(pooled)

# ───────────── Augmentation: Random Rotation ──────────────
def random_rotate(pos, batch):
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
    return pos

# ───────────── Dataset ──────────────
class PtDataset(InMemoryDataset):
    def __init__(self, pt_path):
        super().__init__(os.path.dirname(pt_path))
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"Data file not found at {pt_path}")
        blob = torch.load(pt_path)
        self.graphs = blob['data_list']
        raw_vals = torch.tensor([t[0] for t in blob['target_values']], dtype=torch.float32)
        for g, raw in zip(self.graphs, raw_vals):
            g.pos = g.x[:, -3:]
            g.x = g.x[:, :-3]
            g.y = raw.unsqueeze(0)  # Use raw value directly as label
        self.node_dim = self.graphs[0].x.size(1)
        self.edge_dim = self.graphs[0].edge_attr.size(1) if self.graphs[0].edge_attr is not None else 0
        self.out_dim = 1

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

# ───────────── Single Epoch Training/Validation ──────────────
def run_epoch(model, loader, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    total_loss, n_graph = 0., 0
    loss_fn = torch.nn.SmoothL1Loss()  # Use standard Smooth L1 Loss

    for batch in loader:
        batch = batch.to(DEVICE)
        target = batch.y.view(-1)
        k = ROT_K if train else 0
        k = max(k, 1)
        loss = 0.
        for _ in range(k):
            pos_rot = random_rotate(batch.pos, batch.batch) if (train and ROT_K > 0) else batch.pos
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

# ───────────── Main Process ──────────────
def main():
    ds = PtDataset(DATA_PT)
    N = len(ds)
    n_val = n_test = int(0.1 * N)
    n_train = N - n_val - n_test
    tr_ds, va_ds, te_ds = random_split(ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))
    # tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    # va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE)
    te_ld = DataLoader(te_ds, batch_size=BATCH_SIZE)

    model = EGNNModel(ds.node_dim, ds.edge_dim, num_layers=NUM_LAYERS, hidden_dim=HIDDEN_DIM).to(DEVICE)
    print(f"Device: {DEVICE} | Hidden Dim: {HIDDEN_DIM} | Num Layers: {NUM_LAYERS} | Parameters: {sum(p.numel() for p in model.parameters()):,}")
    # print(f"Rotation Augmentations (K): {ROT_K}")

    # opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    # sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=1e-6)

    # best_val_loss, wait = float('inf'), 0
    # hist = {"train": [], "val": []}
    # for epoch in range(1, MAX_EPOCHS + 1):
    #     tr_loss = run_epoch(model, tr_ld, opt)

    #     with torch.no_grad():
    #         va_loss = run_epoch(model, va_ld)

    #     sch.step()
    #     hist["train"].append(tr_loss)
    #     hist["val"].append(va_loss)

    #     is_best = va_loss < best_val_loss - 1e-5
    #     if is_best:
    #         best_val_loss, wait = va_loss, 0
    #         torch.save({"model": model.state_dict(), "epoch": epoch, "val": va_loss}, "best_model_egnn.pth")
    #     else:
    #         wait += 1

    #     print(f"Epoch {epoch:03d} | Train Loss: {tr_loss:.4f} | Val Loss: {va_loss:.4f} | LR: {opt.param_groups[0]['lr']:.2e} {'** New Best **' if is_best else ''}")
    #     if wait >= EARLY_PATIENCE:
    #         print(f"Early stopping after {EARLY_PATIENCE} epochs of no improvement.")
    #         break

    # ---------- Testing ----------
    print("\n--- Testing on the best model ---")
    if not os.path.exists("best_model_egnn.pth"):
        print("Error: best_model_egnn.pth not found. Please ensure the trained model exists.")
        return
        
    ckpt = torch.load("best_model_egnn.pth", map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for batch in te_ld:
            batch = batch.to(DEVICE)
            p = model(batch.x, batch.pos, batch.edge_index, batch.edge_attr, batch.batch).view(-1)
            preds.append(p.cpu())
            gts.append(batch.y.view(-1).cpu()) # Ground truth is now in batch.y
    preds = torch.cat(preds)
    gts = torch.cat(gts)
    preds = preds.clamp_min(0.) # Remove inverse transform

    rmse = ((preds - gts).pow(2).mean()).sqrt().item()
    mae = (preds - gts).abs().mean().item()
    print(f"Test RMSE: {rmse:.4f} | Test MAE: {mae:.4f}")

    # ---------- Save Results ----------
    # json.dump(hist, open("loss_history_egnn.json", "w"), indent=2)
    with open("results_egnn.txt", "w") as f:
        f.write(f"Test RMSE: {rmse:.4f}\n")
        f.write(f"Test MAE: {mae:.4f}\n")
        if 'epoch' in ckpt:
            f.write(f"Best epoch: {ckpt['epoch']}\n")
        if 'val' in ckpt:
            f.write(f"Best validation loss: {ckpt['val']:.4f}\n")
    print("Results and loss history saved.")


if __name__ == "__main__":
    main()

