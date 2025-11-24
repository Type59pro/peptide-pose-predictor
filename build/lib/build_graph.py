import os
import numpy as np
import concurrent.futures
from tqdm import tqdm
import pickle
from Bio.PDB import Structure, Model, Chain, Residue, Atom, PDBParser, NeighborSearch, Selection, SASA
from Bio.PDB.SASA import ShrakeRupley
import networkx as nx
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from concurrent.futures import ProcessPoolExecutor, as_completed

import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning

warnings.simplefilter('ignore', PDBConstructionWarning)

def calculate_sasa(structure):
    # sasa: Solvent Accessible Surface Area
    sasa_calculator = ShrakeRupley()
    sasa_calculator.compute(structure)

    sasas = {}

    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    residue_id = residue.get_id()[1] 
                    key = (residue.get_resname(), residue_id, chain.id, atom.get_id())
                    sasas[key] = (atom.sasa, atom.get_coord())
    max_sasa = 45
    for key in sasas:
        original_sasa = sasas[key][0]
        normalized_sasa = min(original_sasa/max_sasa, 1.0)
        sasas[key] = (normalized_sasa, sasas[key][1])
    return sasas

def cal_rmsd(str1, str2):
    def get_corresponding_atoms(structure1, structure2, backbone_only=False):
        """Get corresponding atoms in two structures"""
        atoms1 = []
        atoms2 = []
        for residue1, residue2 in zip(structure1.get_residues(), structure2.get_residues()):
            for atom1, atom2 in zip(residue1, residue2):
                if backbone_only:
                    if atom1.get_name() in ['N', 'CA', 'C', 'O'] and atom1.element != 'H':
                        atoms1.append(atom1)
                        atoms2.append(atom2)
                elif atom1.element != 'H':
                    atoms1.append(atom1)
                    atoms2.append(atom2)
        return atoms1, atoms2

    def calculate_rmsd(atoms1, atoms2):
        """Calculate RMSD between two sets of atoms"""
        diff = np.array([atom1.coord - atom2.coord for atom1, atom2 in zip(atoms1, atoms2)])
        return np.sqrt((np.sum(diff ** 2)) / len(atoms1))

    # Parse two PDB files
    # print(str1, str2)
    parser = PDBParser()
    structure1 = parser.get_structure("structure1", str1)
    structure2 = parser.get_structure("structure2", str2)

    # Calculate RMSD for non-hydrogen atoms
    all_atoms1, all_atoms2 = get_corresponding_atoms(structure1, structure2)
    rmsd_all = calculate_rmsd(all_atoms1, all_atoms2)

    # Calculate RMSD for backbone atoms
    backbone_atoms1, backbone_atoms2 = get_corresponding_atoms(structure1, structure2, backbone_only=True)
    rmsd_backbone = calculate_rmsd(backbone_atoms1, backbone_atoms2)
    rmsd_value = [rmsd_all, rmsd_backbone]
    return rmsd_value

def remove_hydrogen_atoms(structure):
    """Remove hydrogen atoms from the structure."""
    for model in structure:
        for chain in model:
            for residue in chain:
                # Get and remove hydrogen atoms from current residue
                hydrogen_atoms = [atom.id for atom in residue if atom.element == 'H']
                for atom_id in hydrogen_atoms:
                    residue.detach_child(atom_id)  # Remove hydrogen atom
    return structure

def rename_chains(structure, base_name='A'):
    existing_ids = set()  # Store existing chain IDs
    for model in structure:
        for chain in model:
            # Find the next unused chain ID
            while base_name in existing_ids:
                base_name = chr(ord(base_name) + 1)  # Move to next letter
            
            chain.id = base_name
            existing_ids.add(base_name)  # Add to existing ID set
            base_name = chr(ord(base_name) + 1)  # Move to next letter
            
    return structure

def rename_peptide_chains(structure, chain_id='_'):
    for model in structure:
        for chain in model:
            chain.id = chain_id  # Set chain ID to '_'
    return structure

def combine_structures(protein_structure, peptide_structure):
    """Combine two PDB structures."""
    # Create a new structure, must have a model
    combined_structure = Structure.Structure("combined_structure")
    combined_model = Model.Model(0)  # Use model ID 0

    # Add chains from protein structure
    for model in protein_structure:
        for chain in model:
            new_chain = Chain.Chain(chain.id)  # Create new chain
            for residue in chain:
                new_residue = Residue.Residue(residue.id, residue.resname, residue.segid)
                # Add atoms to new residue
                for atom in residue:
                    new_atom = Atom.Atom(atom.name, atom.coord, atom.occupancy, atom.bfactor,
                                         atom.fullname, atom.altloc, atom.element)  # Remove `atom.chain`
                    new_residue.add(new_atom)  # Add atom to new residue
                new_chain.add(new_residue)  # Add new residue to new chain
            combined_model.add(new_chain)  # Add new chain to model

    # Add chains from peptide structure
    for model in peptide_structure:
        for chain in model:
            new_chain = Chain.Chain(chain.id)  # Create new chain
            for residue in chain:
                new_residue = Residue.Residue(residue.id, residue.resname, residue.segid)
                for atom in residue:
                    new_atom = Atom.Atom(atom.name, atom.coord, atom.occupancy, atom.bfactor,
                                         atom.fullname, atom.altloc, atom.element)  # Also remove `atom.chain`
                    new_residue.add(new_atom)  # Add atom to new residue
                new_chain.add(new_residue)  # Add new residue to new chain
            combined_model.add(new_chain)  # Add new chain to model

    combined_structure.add(combined_model)  # Add model to combined structure
    return combined_structure


def build_graph(input_pdb, input_peptide):
    parser = PDBParser()
    protein = parser.get_structure('protein', input_pdb)
    peptide = parser.get_structure('peptide', input_peptide)

    protein_no_h = remove_hydrogen_atoms(protein)
    peptide_no_h = remove_hydrogen_atoms(peptide)

    # protein_no_h = rename_chains(protein_no_h)
    peptide_no_h = rename_peptide_chains(peptide_no_h)

    combine_structure = combine_structures(protein_no_h, peptide_no_h)

    sasas = calculate_sasa(combine_structure)

    protein_atoms = list(protein_no_h.get_atoms())
    peptide_atoms = list(peptide_no_h.get_atoms())

    protein_coords = np.array([atom.get_coord() for atom in protein_atoms])
    peptide_coords = np.array([atom.get_coord() for atom in peptide_atoms])

    protein_sasas = {}
    peptide_sasas = {}
    for atom in protein.get_atoms():
        residue = atom.get_parent()
        residue_name = residue.get_resname()
        residue_id = residue.get_id()[1]  # Get residue number
        chain_id = residue.get_parent().get_id()  # Get chain ID
        key = (residue_name, residue_id, chain_id, atom.get_id())
        protein_sasas[key] = sasas.get(key, (0.0, None))[0]

    for atom in peptide.get_atoms():
        residue = atom.get_parent()
        residue_name = residue.get_resname()
        residue_id = residue.get_id()[1]  # Get residue number
        chain_id = residue.get_parent().get_id()  # Get chain ID
        key = (residue_name, residue_id, chain_id, atom.get_id())
        peptide_sasas[key] = sasas.get(key, (0.0, None))[0]
    
    protein_pocket = set()
    distance_threshold = 10.0
    distance_threshold_squared = distance_threshold ** 2

    diff = peptide_coords[:, np.newaxis, :] - protein_coords[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    close_pairs = np.where(dist_sq < distance_threshold_squared)
    for protein_idx in close_pairs[1]:
        atom = protein_atoms[protein_idx]
        protein_pocket.add(atom)
    
    pocket_atoms = []
    for atom in protein_pocket:
        residue = atom.get_parent()
        residue_name = residue.get_resname()
        residue_id = residue.get_id()[1]
        chain_id = residue.get_parent().get_id()
        element = atom.element
        atom_idx = atom.get_serial_number()
        atom_name = atom.get_name()
        coords = atom.get_coord()
        sasa_value = protein_sasas.get((residue_name, residue_id, chain_id, atom.get_id()), 0.0)
        pocket_atoms.append([residue_name, residue_id, chain_id, atom_name, atom_idx, element, coords, sasa_value])
        # print(residue_name, residue_id, chain_id, atom_name, atom_idx, element, coords, sasa_value)

    pep_atoms = []
    for atom in peptide_atoms:
        residue = atom.get_parent()
        residue_name = residue.get_resname()
        residue_id = residue.get_id()[1]
        chain_id = residue.get_parent().get_id()
        element = atom.element
        atom_idx = atom.get_serial_number()
        atom_name = atom.get_name()
        coords = atom.get_coord()
        sasa_value = peptide_sasas.get((residue_name, residue_id, chain_id, atom.get_id()), 0.0)
        pep_atoms.append([residue_name, residue_id, chain_id, atom_name, atom_idx, element, coords, sasa_value])
        # print(residue_name, residue_id, chain_id, atom_name, atom_idx, element, coords, sasa_value)

    G = nx.Graph()
    global_idx = 0

    all_atoms = pocket_atoms + pep_atoms

    def amino_type_encoding(aa):
    # Encode by residue name
        if aa in ['GLY', 'ALA', 'PRO', 'VAL', 'LEU', 'ILE']:
            aa_type =  [1,0,0,0]
        elif aa in ['SER', 'THR', 'GLN', 'ASN', 'MET', 'CYS']:
            aa_type = [0,1,0,0]
        elif aa in ['PHE','TYR', 'TRP']:
            aa_type = [0,0,1,0]
        elif aa in ['ASP', 'GLU', 'ARG', 'LYS', 'HIS', 'HIE', 'HID', 'HIP']:
            aa_type = [0,0,0,1]
        else:
            aa_type = [0,0,0,0]
        amino_acids = [
        'GLY', 'ALA', 'PRO', 'VAL', 'LEU', 'ILE',
        'SER', 'THR', 'GLN', 'ASN', 'MET', 'CYS',
        'PHE','TYR', 'TRP',
        'ASP', 'GLU', 'ARG', 'LYS', 'HIS'
        ]
        amino_acid_index = {aa: idx for idx, aa in enumerate(amino_acids)}
        if aa in [ 'HIE', 'HID', 'HIP']: # Different protonation states
            aa = 'HIS'
        
        if aa in [ 'CYX', 'CYS']: # CYX is CYS with disulfide bond
            aa = 'CYS'
        
        one_hot = np.zeros(len(amino_acids), dtype=int)

        if aa in amino_acid_index:
            index = amino_acid_index[aa]
            one_hot[index] = 1
            return one_hot.tolist() + aa_type
        
    def atom_charge(aa, atom_name):
        '''
        charge from ff14SB force field
        '''
        charge_dict = {
            'GLY':{ 'N': -0.4157, 'CA': -0.0252, 'C': 0.5973, 'O': -0.5679},
            'ALA':{ 'N': -0.4157, 'CA': -0.0252, 'C': 0.5973, 'O': -0.5679, 'CB': -0.1825},
            'PRO':{ 'N': -0.2548, 'CA': -0.0266, 'C': 0.5896, 'O': -0.5748, 'CB': -0.0070,
                    'CD': 0.0192, 'CG': 0.0189},
            'VAL':{ 'N': -0.4157, 'CA': -0.0875, 'C': 0.5973, 'O': -0.5679, 'CB': 0.2985,
                    'CG1': -0.3192, 'CG2': -0.3192},
            'LEU':{ 'N': -0.4157, 'CA': -0.0518, 'C': 0.5973, 'O': -0.5679, 'CB': -0.1102,
                    'CG': 0.3531, 'CD1': -0.4121, 'CD2': -0.4121},
            'ILE':{ 'N': -0.4157, 'CA': -0.0597, 'C': 0.5973, 'O': -0.5679, 'CB': 0.1303,
                    'CG1': -0.0430, 'CG2': -0.3204, 'CD1': -0.0660},
            'SER':{ 'N': -0.4157, 'CA': -0.0249, 'C': 0.5973, 'O': -0.5679, 'CB': 0.2117,
                    'OG': -0.6546},
            'THR':{ 'N': -0.4157, 'CA': -0.0389, 'C': 0.5973, 'O': -0.5679, 'CB': 0.3654,
                    'OG1': -0.6761, 'CG2': -0.2438},
            'GLN':{ 'N': -0.4157, 'CA': -0.0031, 'C': 0.5973, 'O': -0.5679, 'CB': -0.0036,
                    'CG': -0.0645, 'CD': 0.6951, 'OE1': -0.6086, 'NE2': -0.9407},
            'ASN':{ 'N': -0.4157, 'CA': 0.0143, 'C': 0.5973, 'O': -0.5679, 'CB': -0.2041,
                    'CG': 0.7130, 'OD1': -0.5931, 'ND2': -0.9191},
            'MET':{ 'N': -0.4157, 'CA': -0.0237, 'C': 0.5973, 'O': -0.5679, 'CB': 0.0342,
                    'CG': 0.0018, 'SD': -0.2737, 'CE': -0.0536},
            'CYS':{ 'N': -0.4157, 'CA': 0.0213, 'C': 0.5973, 'O': -0.5679, 'CB': -0.1231,
                    'SG': -0.3119},
            'PHE':{ 'N': -0.4157, 'CA': -0.0024, 'C': 0.5973, 'O': -0.5679, 'CB': -0.0343,
                    'CG': 0.0118, 'CD1': -0.1256, 'CD2': -0.1256, 'CE1': -0.1704,
                    'CE2': -0.1704, 'CZ': -0.1072},
            'TYR':{ 'N': -0.4157, 'CA': -0.0014, 'C': 0.5973, 'O': -0.5679, 'CB': -0.0152,
                    'CG': -0.0011, 'CD1': -0.1906, 'CD2': -0.1906, 'CE1': -0.2341,
                    'CE2': -0.2341, 'CZ': 0.3226, 'OH': -0.5579},
            'TRP':{ 'N': -0.4157, 'CA': -0.0275, 'C': 0.5973, 'O': -0.5679, 'CB': -0.0050,
                    'CG': -0.1415, 'CD1': -0.1638, 'CD2': 0.1243, 'CE2': 0.1380,
                    'CE3': -0.2387, 'CZ2': -0.2601, 'CZ3': -0.1972, 'CH2': -0.1134,
                    'NE1': -0.3418},
            'ASP':{ 'N': -0.5163, 'CA': 0.0381, 'C': 0.5366, 'O': -0.5819, 'CB': -0.0303,
                    'CG': 0.7994, 'OD1': -0.8014, 'OD2': -0.8014},
            'GLU':{ 'N': -0.5163, 'CA': 0.0397, 'C': 0.5366, 'O': -0.5819, 'CB': 0.0560,
                    'CG': 0.0136, 'CD': 0.8054, 'OE1': -0.8188, 'OE2': -0.8188},
            'ARG':{ 'N': -0.3479, 'CA': -0.2637, 'C': 0.7341, 'O': -0.5894, 'CB': -0.0007,
                    'CG': 0.0390, 'CD': 0.0486, 'NE': -0.5295, 'CZ': 0.8076,
                    'NH1': -0.8627, 'NH2': -0.8627},
            'LYS':{ 'N': -0.3479, 'CA': -0.2400, 'C': 0.7341, 'O': -0.5894, 'CB': -0.0094,
                    'CG': 0.0187, 'CD': -0.0479, 'CE': -0.0143, 'NZ': -0.3854},
            'HIS':{ 'N': -0.4157, 'CA': -0.0581, 'C': 0.5973, 'O': -0.5679, 'CB': -0.0074,
                    'CG': 0.1868, 'ND1': -0.5432, 'CD2': -0.2207, 'CE1': 0.1635,
                    'NE2': -0.2795, 'CE2': 0.3226}
        }
        if aa in [ 'HIE', 'HID', 'HIP']:
            aa = 'HIS'
        
        if aa in [ 'CYX', 'CYS']:
            aa = 'CYS'

        residue_atom_charge = charge_dict[aa][atom_name] if aa in charge_dict and atom_name in charge_dict[aa] else 0.0
        return [residue_atom_charge]
    
    def atom_connectivity(aa, atom_name):
        '''
        [all connectivity atoms, heavy atom connectivity, hydrogen atom connectivity]
        '''
        # all atom 4(1,2,3,4), heavy atom 4(1,2,3,4), hydrogen atom 4(0,1,2,3)
        atom_connectivity_dict = {
            "GLY": {
                "N": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], 
                "CA": [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], 
                "C": [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0], 
                "O": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                  "OXT": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]}, 
            "ALA": {
                "N": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], 
                "CA": [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], 
                "C": [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0], 
                "O": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 
                "OXT": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 
                "CB": [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]}, 
            "PRO": {
                "N": [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0], 
                "CA": [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], 
                "C": [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                "O": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                "OXT": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 
                "CB": [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0], 
                "CD": [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0], 
                "CG": [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0]}, 
            "VAL": {
                "N": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], 
                "CA": [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], 
                "C": [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0], 
                "O": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                "OXT": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 
                "CB": [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], 
                "CG1": [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1], 
                "CG2": [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]}, 
            "LEU": {
                "N": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], 
                "CA": [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], 
                "C": [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                "O": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 
                "OXT": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 
                "CB": [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0], 
                "CG": [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], 
                "CD1": [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1], 
                "CD2": [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]}, 
            "ILE": {
                "N": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], 
                "CA": [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], 
                "C": [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0], 
                "O": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                "OXT": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 
                "CB": [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], 
                "CG1": [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0], 
                "CG2": [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1], 
                "CD1": [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]}, 
            "SER": {
                "N": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], 
                "CA": [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], 
                "C": [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0], 
                "O": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 
                "OXT": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 
                "CB": [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0], 
                "OG": [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]},
            "THR": {
                "N": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], 
                "CA": [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                "C": [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                "O": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 
                "OXT": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                "CB": [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                "OG1": [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], 
                "CG2": [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]}, 
            "GLN": {
                "N": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], 
                "CA": [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], 
                "C": [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0], 
                "O": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 
                "OXT": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 
                "CB": [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0], 
                "CG": [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0], 
                "CD": [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0], 
                "OE1": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 
                "NE2": [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0]}, 
            "ASN": {
                "N": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], 
                "CA": [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], 
                "C": [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                "O": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 
                "OXT": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 
                "CB": [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0], 
                "CG": [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0], 
                "OD1": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 
                "ND2": [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0]}, 
            "MET": {
                "N": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], 
                "CA": [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], 
                "C": [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0], 
                "O": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 
                "OXT": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 
                "CB": [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0], 
                "CG": [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0], 
                "SD": [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0], 
                "CE": [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]}, 
            "CYS": {
                "N": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], 
                "CA": [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], 
                "C": [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0], 
                "O": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 
                "OXT": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 
                "CB": [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0], 
                "SG": [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]}, 
            "PHE": {
                "N": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], 
                "CA": [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], 
                "C": [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0], 
                "O": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 
                "OXT": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 
                "CB": [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0], 
                "CG": [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0], 
                "CD1": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], 
                "CD2": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], 
                "CE1": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], 
                "CE2": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], 
                "CZ": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0]}, 
            "TYR": {
                "N": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], 
                "CA": [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], 
                "C": [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0], 
                "O": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 
                "OXT": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 
                "CB": [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0], 
                "CG": [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0], 
                "CD1": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], 
                "CD2": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], 
                "CE1": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], 
                "CE2": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                "CZ": [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                "OH": [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]}, 
            "TRP": {
                "N": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], 
                "CA": [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                "C": [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0], 
                "O": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 
                "OXT": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 
                "CB": [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0], 
                "CG": [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0], 
                "CD1": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], 
                "CD2": [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0], 
                "CE2": [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0], 
                "CE3": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], 
                "CZ2": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], 
                "CZ3": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], 
                "CH2": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], 
                "NE1": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0]},
            "ASP": {
                "N": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], 
                "CA": [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], 
                "C": [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0], 
                "O": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 
                "OXT": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 
                "CB": [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0], 
                "CG": [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0], 
                "OD1": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 
                "OD2": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]}, 
            "GLU": {
                "N": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], 
                "CA": [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], 
                "C": [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0], 
                "O": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 
                "OXT": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 
                "CB": [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0], 
                "CG": [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0], 
                "CD": [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0], 
                "OE1": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 
                "OE2": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]}, 
            "ARG": {
                "N": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], 
                "CA": [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], 
                "C": [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0], 
                "O": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 
                "OXT": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 
                "CB": [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0], 
                "CG": [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0], 
                "CD": [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0], 
                "NE": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], 
                "CZ": [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0], 
                "NH1": [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0], 
                "NH2": [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0]}, 
            "LYS": {
                "N": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], 
                "CA": [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], 
                "C": [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0], 
                "O": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 
                "OXT": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 
                "CB": [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0], 
                "CG": [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0], 
                "CD": [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0], 
                "CE": [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0], 
                "NZ": [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]}, 
            "HIS": {
                "N": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], 
                "CA": [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], 
                "C": [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0], 
                "O": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 
                "OXT": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 
                "CB": [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0], 
                "CG": [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0], 
                "ND1": [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0], 
                "CD2": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], 
                "CE1": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], 
                "NE2": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0]}
            }

        if aa in [ 'HIE', 'HID', 'HIP']:
            aa = 'HIS'
        
        if aa in [ 'CYX', 'CYS']:
            aa = 'CYS'

        atom_connectivity_key = atom_connectivity_dict[aa][atom_name] if aa in atom_connectivity_dict and atom_name in atom_connectivity_dict[aa] else 0.0
        return atom_connectivity_key

    def element_encoding(ele):
        element_all = ['C','N','O','S']
        return [int(ele == r) for r in element_all]

    def interaction_type(res1, res2, name1, name2, ele1, ele2):
    # Aromatic ring atom list
        aromatic_atoms = {'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'CH2', 'CZ2', 'CZ3', 'CD', 'CE',
                        'ND1', 'NE1', 'NE2', 'CD2'}

    # Aromatic residues
        aromatic_residues = {'PHE', 'TYR', 'TRP', 'HIS'}

    # Positively charged residues and their atoms
        pos_sidechain_atoms = {
            'ARG': {'NH1', 'NH2', 'NE'},
            'LYS': {'NZ'},
            'HIS': {'ND1', 'NE2'}  
        }

    # Negatively charged residues and their atoms
        neg_sidechain_atoms = {
            'ASP': {'OD1', 'OD2'},
            'GLU': {'OE1', 'OE2'},
        }
    # Determine if aromatic atom
        is_aromatic_1 = res1 in aromatic_residues and name1 in aromatic_atoms
        is_aromatic_2 = res2 in aromatic_residues and name2 in aromatic_atoms
        is_pos_1 = res1 in pos_sidechain_atoms and name1 in pos_sidechain_atoms[res1]
        is_pos_2 = res2 in pos_sidechain_atoms and name2 in pos_sidechain_atoms[res2]
        is_neg_1 = res1 in neg_sidechain_atoms and name1 in neg_sidechain_atoms[res1]
        is_neg_2 = res2 in neg_sidechain_atoms and name2 in neg_sidechain_atoms[res2]
        # PI-PI stacking
        if is_aromatic_1 and is_aromatic_2:
            return [1,0,0,0,0,0,0]
        # PI cation
        if (is_aromatic_1 and is_pos_2) or (is_aromatic_2 and is_pos_1):
            return [0,1,0,0,0,0,0]
        # electrostatic attraction
        if (is_pos_1 and is_neg_2) or (is_neg_1 and is_pos_2):
            return [0,0,1,0,0,0,0]
        # electrostatic repulsion
        if (is_pos_1 and is_pos_2) or (is_neg_1 and is_neg_2):
            return [0,0,0,1,0,0,0]
        # hydrophobic interaction
        if ele1 in [ 'C', 'S' ] and ele2 in [ 'C', 'S']:
            return [0,0,0,0,1,0,0]
        # hydrophobic repulsion
        if (ele1 in [ 'C', 'S' ] and ele2 not in [ 'C', 'S']) or (ele1 not in [ 'C', 'S' ] and ele2 in [ 'C', 'S']):
            return [0,0,0,0,0,1,0]
        # hbond
        if ele1 in ['N','O'] and ele2 in ['N', 'O']:
            return [0,0,0,0,0,0,1]   
        raise ValueError('error residue')
    
    def peptide_localtion(atom_name):
        if atom_name in ['N', 'CA', 'C', 'O', 'OXT']:
            return [0] 
        else:
            return [1]
        
    # ---------- protein pocket ----------  
    for residue_name, residue_id, chain_id, atom_name, atom_idx, element, coords, sasa_value in pocket_atoms:  
        node_name = f"POKT_{residue_name}_{residue_id}_{chain_id}_{atom_name}_{atom_idx}"
        # print(element, element_encoding(element), residue_name, amino_type_encoding(residue_name), atom_charge(residue_name, atom_name), atom_connectivity(residue_name, atom_name), sasa_value)
        G.add_node(node_name,  
                feature = element_encoding(element) + [ 0 ] + peptide_localtion(atom_name) + amino_type_encoding(residue_name) + atom_charge(residue_name, atom_name) + atom_connectivity(residue_name, atom_name) + [sasa_value],
                # element=element_encoding(element),  
                # source=0,  
                # aa=amino_type_encoding(residue_name),  
                # coords=coords,  
                # idx=global_idx)  
                pos = coords)
        global_idx += 1  

    # ---------- peptide ----------  
    for residue_name, residue_id, chain_id, atom_name, atom_idx, element, coords, sasa_value in pep_atoms:  
        node_name = f"PEPT_{residue_name}_{residue_id}_{chain_id}_{atom_name}_{atom_idx}" 
        G.add_node(node_name,  
                feature = element_encoding(element) + [ 1 ] + peptide_localtion(atom_name) + amino_type_encoding(residue_name) + atom_charge(residue_name, atom_name) + atom_connectivity(residue_name, atom_name) + [sasa_value],
                # element_encoding: 4 
                # position: 1
                # peptide_location: 1
                # amino_type_encoding: 24
                # atom_charge: 1
                # atom_connectivity: 12
                # sasa_value: 1
                # total: 44
                pos = coords )
        global_idx += 1 

    n_atoms = len(all_atoms)
    for i in range(n_atoms):  
        residue_name_i, residue_id_i, chain_i, atom_name_i, atom_idx_i, element_i, coords_i, _ = all_atoms[i]  
        node_i = f"{'POKT' if i < len(pocket_atoms) else 'PEPT'}_{residue_name_i}_{residue_id_i}_{chain_i}_{atom_name_i}_{atom_idx_i}"
        for j in range(i+1, n_atoms):  
            residue_name_j, residue_id_j, chain_j, atom_name_j, atom_idx_j, element_j, coord_j, _ = all_atoms[j] 
            node_j = f"{'POKT' if j < len(pocket_atoms) else 'PEPT'}_{residue_name_j}_{residue_id_j}_{chain_j}_{atom_name_j}_{atom_idx_j}" 
            if node_i == node_j:
                continue
            d = np.linalg.norm(coords_i - coord_j)  
            if d < 1.9:  
                G.add_edge(node_i, node_j, 
                           feature =  [ 1, 0,0,0,0,0,0,0, 1]
                        #    bond_type=1,  
                        #    interaction_type=[0,0,0,0,0,0,0,1],  
                        #    distance=1.0
                        )  
            elif d < 5.0:  
                G.add_edge(node_i, node_j, 
                           feature = [0] + interaction_type(residue_name_i,residue_name_j,atom_name_i,atom_name_j,element_i,element_j) + [1 - (d-1.9)/(5.0-1.9)]
                        #    bond_type=0,  
                        #    interaction_type=interaction_type(res_i,res_j,name_i,name_j,ele_i,ele_j),  
                        #    distance= 1 - (d-1.9)/(5.0-1.9)
                           ) 
    # print(G.nodes(data= True))
    # print(G.edges(data= True))
    return G

def convert_nx_to_pyg(nx_graph, rmsd_list):
    # Convert networkx graph to PyG data object
    node_mapping = {node: idx for idx, node in enumerate(nx_graph.nodes())}
    node_features = []
    for node in nx_graph.nodes(data=True):
        feature = node[1].get('feature', [0] * 40)
        pos = node[1].get('pos', [0.0, 0.0, 0.0]) 
        combine_features = feature[:]
        combine_features.extend(pos)
        node_features.append(combine_features)

    x = torch.tensor(node_features, dtype=torch.float)

    # Get edge info and build edge_index
    edges = list(nx_graph.edges())
    edge_index = torch.tensor([[node_mapping[u], node_mapping[v]] for u, v in edges], dtype=torch.long).t()

    # Get edge features
    edge_features = []
    for edge in nx_graph.edges(data=True):
        edge_feature = edge[2].get('feature', [0] * 44)  # Default edge feature is 44 elements
        edge_features.append(edge_feature)

    edge_attr = torch.tensor(edge_features, dtype=torch.float) if edge_features else None

    # Convert target values to tensor
    y_tensor = torch.tensor(rmsd_list, dtype=torch.float)

    # Create PyG data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y_tensor)
    # print(y_tensor)
    return data

def process_docking(protein_file, peptide_file, init_pdb):
    nx_graph = build_graph(protein_file, peptide_file)    
    rmsd_list = cal_rmsd(peptide_file, init_pdb)    
    pyg_data = convert_nx_to_pyg(nx_graph, rmsd_list)
    return pyg_data
    
def main():  
    work_path = 'cases'  
    system_fold = [f for f in os.listdir(work_path) if os.path.isdir(os.path.join(work_path, f))]  

    all_tasks = []  
    for fold_id in system_fold:  
        all_pdb = [f for f in os.listdir(os.path.join(work_path, fold_id)) if f.startswith('upper_ranked') or f.startswith('lower_ranked') or f.startswith('min_peptide_')]  
        protein_f = os.path.join(work_path, fold_id, 'protein.pdb') 
        ligand_f = os.path.join(work_path, fold_id, 'ligand.pdb')  
        
        for docking_ligand in all_pdb:  
            docking_results_f = os.path.join(work_path, fold_id, docking_ligand)  
            all_tasks.append((protein_f, docking_results_f, ligand_f))  

    graphs = []
    target_values = []

    with ProcessPoolExecutor(max_workers= 28) as executor:
        futures = {executor.submit(process_docking, *task_input): task_input for task_input in all_tasks}
        for future in tqdm(as_completed(futures), total= len(futures), desc="Processing docking tasks"):
            task_input = futures[future]
            try:
                result = future.result()
                if result is not None:
                    graphs.append(result)
                    target_values.append(result.y.tolist())
            except Exception as e:
                print(f"Error occur: {task_input} error: {e}")
    output_file = 'data_long.pt'
    torch.save({
        'data_list': graphs,
        'target_values': target_values
    }, output_file)

if __name__ == '__main__':
    main()