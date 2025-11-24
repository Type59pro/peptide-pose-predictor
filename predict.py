import argparse
import torch
import os
import sys
import warnings

# Filter warnings
warnings.filterwarnings("ignore")

try:
    from build_graph import build_graph, convert_nx_to_pyg
    from inference import EGNNModel, DEVICE, HIDDEN_DIM, NUM_LAYERS
except ImportError:
    # Fallback for when running directly without installation or path issues
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from build_graph import build_graph, convert_nx_to_pyg
    from inference import EGNNModel, DEVICE, HIDDEN_DIM, NUM_LAYERS

def main():
    parser = argparse.ArgumentParser(description="Peptide Pose Predictor")
    parser.add_argument("--prot", type=str, required=True, help="Path to protein PDB file")
    parser.add_argument("--pep", type=str, required=True, help="Path to peptide PDB file")
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint (default: best_model_egnn.pth in package dir)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.prot):
        print(f"Error: Protein file {args.prot} not found.")
        sys.exit(1)
    if not os.path.exists(args.pep):
        print(f"Error: Peptide file {args.pep} not found.")
        sys.exit(1)
        
    # Determine model path
    if args.model is None:
        # Look for model in the same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.model = os.path.join(script_dir, "best_model_egnn.pth")
    
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found. Please specify with --model.")
        sys.exit(1)

    # Build graph
    print(f"Processing {args.prot} and {args.pep}...")
    try:
        nx_graph = build_graph(args.prot, args.pep)
        # Pass dummy RMSD values [0.0, 0.0] as we are in inference mode and don't have ground truth
        data = convert_nx_to_pyg(nx_graph, [0.0, 0.0])
    except Exception as e:
        print(f"Error building graph: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    # Load model
    # print(f"Loading model from {args.model}...")
    
    # Determine dimensions from data
    node_dim = data.x.size(1)
    edge_dim = data.edge_attr.size(1) if data.edge_attr is not None else 0
    
    model = EGNNModel(
        node_dim=node_dim, 
        edge_dim=edge_dim, 
        num_layers=NUM_LAYERS, 
        hidden_dim=HIDDEN_DIM
    ).to(DEVICE)
    
    try:
        ckpt = torch.load(args.model, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    # Run inference
    # print("Running inference...")
    with torch.no_grad():
        data = data.to(DEVICE)
        # The model expects a batch, so we need to add a batch index
        # Since it's a single graph, batch is all zeros
        batch = torch.zeros(data.x.size(0), dtype=torch.long).to(DEVICE)
        
        pred = model(data.x, data.pos, data.edge_index, data.edge_attr, batch).view(-1)
        
    print(f"Predicted RMSD: {pred.item():.4f}")

if __name__ == "__main__":
    main()
