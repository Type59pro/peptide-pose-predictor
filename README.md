# Peptide Pose Predictor

This project provides a tool to predict the quality (RMSD) of a peptide pose docked to a protein using an EGNN model.

## Installation

1. Clone the repository.
2. Install the package and dependencies:
   ```bash
   pip install .
   ```
   
   *Note: For `torch` and `torch_geometric`, it is recommended to install them manually first to ensure compatibility with your CUDA version (if applicable).*

## Usage

The installation provides a command-line tool `ppp`.

```bash
ppp --prot <path_to_protein.pdb> --pep <path_to_peptide.pdb>
```

### Arguments

- `--prot`: Path to the protein PDB file.
- `--pep`: Path to the peptide PDB file.
- `--model`: (Optional) Path to the trained model file (`best_model_egnn.pth`). If not provided, the script attempts to find it in the installation directory or current directory.

### Example

```bash
ppp --prot examples/protein.pdb --pep examples/peptide.pdb
```

## Files

- `peptide_pose_predictor/predict.py`: The main entry point for the CLI.
- `peptide_pose_predictor/build_graph.py`: Logic for converting PDB files to graphs.
- `peptide_pose_predictor/inference.py`: Inference logic and model definition.
- `peptide_pose_predictor/training.py`: Training script.
- `peptide_pose_predictor/best_model_egnn.pth`: Pre-trained model weights.
