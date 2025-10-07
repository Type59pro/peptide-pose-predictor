import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

# This script is based on training.py.
# It includes the necessary model and data handling classes and functions for inference.

# ───────────── Parameters (from training.py) ──────────────
# These should match the parameters used during training.
NUM_LAYERS = 3
HIDDEN_DIM = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────── Custom EGNN Layer (PyG compatible, no coordinate updates) ──────────────
class EGNNLayer(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden_dim):
        super().__init__()
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
        self.message_passing = self.build_message_passing()

    def build_message_passing(self):
        from torch_geometric.nn import MessagePassing
        class CustomMessagePassing(MessagePassing):
            def __init__(self, edge_mlp, node_mlp):
                super().__init__(aggr='add', node_dim=0)
                self.edge_mlp = edge_mlp
                self.node_mlp = node_mlp

            def forward(self, x, pos, edge_index, edge_attr):
                return self.propagate(edge_index, x=x, pos=pos, edge_attr=edge_attr)

            def message(self, x_i, x_j, pos_i, pos_j, edge_attr):
                rel_pos = pos_j - pos_i
                dist2 = (rel_pos ** 2).sum(dim=-1, keepdim=True)
                input_feats = torch.cat([x_i, x_j, edge_attr, dist2], dim=-1)
                return self.edge_mlp(input_feats)

            def update(self, aggr_out, x):
                return self.node_mlp(torch.cat([x, aggr_out], dim=-1))
        return CustomMessagePassing(self.edge_mlp, self.node_mlp)

    def forward(self, x, pos, edge_index, edge_attr):
        return self.message_passing(x, pos, edge_index, edge_attr)


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
        for layer in self.egnn_layers:
            h = layer(h, pos, edge_index, edge_attr)
        pooled = torch.cat([
            global_mean_pool(h, batch),
            global_max_pool(h, batch),
            global_add_pool(h, batch)
        ], dim=-1)
        return self.readout(pooled)


# ───────────── Dataset ──────────────
class PtDataset(InMemoryDataset):
    def __init__(self, pt_path):
        super().__init__(os.path.dirname(pt_path))
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"Data file not found at {pt_path}")
        blob = torch.load(pt_path, map_location='cpu')
        self.graphs = blob['data_list']
        
        # Handle targets if they exist, otherwise create placeholders
        if 'target_values' in blob:
            raw_vals = torch.tensor([t[0] for t in blob['target_values']], dtype=torch.float32)
        else:
            raw_vals = torch.full((len(self.graphs),), float('nan'))

        for i, g in enumerate(self.graphs):
            g.pos = g.x[:, -3:]
            g.x = g.x[:, :-3]
            if 'target_values' in blob:
                g.y = raw_vals[i].unsqueeze(0)
            else:
                g.y = torch.tensor([float('nan')]) # Placeholder for ground truth

        self.node_dim = self.graphs[0].x.size(1)
        self.edge_dim = self.graphs[0].edge_attr.size(1) if self.graphs[0].edge_attr is not None else 0
        self.out_dim = 1

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


# ───────────── Main Inference Function ──────────────
def inference(model_path, data_path, batch_size, output_file):
    print(f"Using device: {DEVICE}")
    
    # 1. Load Dataset
    print(f"Loading data from {data_path}...")
    try:
        dataset = PtDataset(data_path)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    print(f"Data loaded. Found {len(dataset)} graphs.")

    # 2. Initialize and Load Model
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
        
    model = EGNNModel(
        node_dim=dataset.node_dim, 
        edge_dim=dataset.edge_dim, 
        num_layers=NUM_LAYERS, 
        hidden_dim=HIDDEN_DIM
    ).to(DEVICE)
    
    try:
        ckpt = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        model.eval()
        print("Model loaded successfully.")
        print(f"Model was trained for {ckpt.get('epoch', 'N/A')} epochs with validation loss {ckpt.get('val', 'N/A'):.4f}")
    except Exception as e:
        print(f"Error loading model state: {e}")
        return

    # 3. Run Inference
    print("Running inference...")
    all_preds = []
    all_gts = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            pred = model(batch.x, batch.pos, batch.edge_index, batch.edge_attr, batch.batch).view(-1)
            all_preds.append(pred.cpu())
            all_gts.append(batch.y.view(-1).cpu())

    preds_tensor = torch.cat(all_preds)
    gts_tensor = torch.cat(all_gts)
    
    print(f"Inference complete. Generated {len(preds_tensor)} predictions.")

    # 4. Save Results
    results_df = pd.DataFrame({
        'prediction': preds_tensor.numpy(),
        'ground_truth': gts_tensor.numpy()
    })
    
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    # 5. Calculate and print metrics if ground truth is available
    if not gts_tensor.isnan().all():
        # Filter out NaN values for metric calculation
        valid_mask = ~gts_tensor.isnan()
        valid_preds = preds_tensor[valid_mask]
        valid_gts = gts_tensor[valid_mask]

        if len(valid_preds) > 0:
            rmse = ((valid_preds - valid_gts).pow(2).mean()).sqrt().item()
            mae = (valid_preds - valid_gts).abs().mean().item()
            print(f"\nMetrics (based on available ground truth):")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
        else:
            print("No valid ground truth values found to calculate metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for EGNN model.")
    parser.add_argument("model_path", type=str, help="Path to the trained model file (e.g., best_model_egnn.pth).")
    parser.add_argument("data_path", type=str, help="Path to the input data file (e.g., data.pt).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
    parser.add_argument("--output_file", type=str, default="predictions.csv", help="Path to save the output predictions CSV file.")
    
    args = parser.parse_args()
    
    inference(args.model_path, args.data_path, args.batch_size, args.output_file)
