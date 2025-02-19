import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdDepictor
import matplotlib.pyplot as plt
import seaborn as sns

def get_pdb_structure(pdb_id, chain_id):
    """Download and load PDB structure."""
    pdb_list = PDBList()
    pdb_path = pdb_list.retrieve_pdb_file(pdb_id)
    parser = PDBParser()
    structure = parser.get_structure(pdb_id, pdb_path)
    return structure[0][chain_id], pdb_path

def prepare_molecule_2d(smiles):
    """Prepare a 2D representation of the molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Generate 2D coordinates
    rdDepictor.Compute2DCoords(mol)
    
    # Try to add Hs and optimize, but don't fail if it doesn't work
    try:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
    except:
        # If 3D optimization fails, just use 2D coordinates
        mol = Chem.MolFromSmiles(smiles)
        rdDepictor.Compute2DCoords(mol)
    
    return mol

def highlight_interaction_sites(mol, attention_weights):
    """Highlight atoms based on attention weights."""
    # Normalize attention weights to [0, 1]
    if len(attention_weights.shape) > 2:
        attention_weights = attention_weights.mean(axis=0)
    attention_scores = attention_weights.mean(axis=0)
    attention_scores = (attention_scores - attention_scores.min()) / (attention_scores.max() - attention_scores.min())
    
    # Map attention scores to atoms (simplified mapping)
    n_atoms = mol.GetNumAtoms()
    atom_scores = attention_scores[:n_atoms] if len(attention_scores) > n_atoms else np.ones(n_atoms)
    
    return atom_scores

def create_interaction_visualization(smiles, protein_info, attention_weights, output_dir):
    """Create a visualization showing potential interaction sites."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure
    fig = plt.figure(figsize=(15, 5))
    
    # 1. Original molecule
    plt.subplot(1, 3, 1)
    mol = prepare_molecule_2d(smiles)
    if mol is not None:
        img = Draw.MolToImage(mol)
        plt.imshow(img)
        plt.axis('off')
        plt.title("Drug Molecule Structure")
    
    # 2. Molecule with highlighted interaction sites
    plt.subplot(1, 3, 2)
    if mol is not None:
        atom_scores = highlight_interaction_sites(mol, attention_weights)
        
        # Create atom highlights
        atom_colors = {}
        for i, score in enumerate(atom_scores):
            # Use a red-yellow gradient for better visibility
            atom_colors[i] = (1, 1-score, 0)  # Red to yellow gradient
        
        # Draw molecule with highlights
        img = Draw.MolToImage(mol, highlightAtoms=range(mol.GetNumAtoms()),
                            highlightColor=None,
                            highlightAtomColors=atom_colors)
        plt.imshow(img)
        plt.axis('off')
        plt.title("Predicted Interaction Sites\n(Red = High attention)")
    
    # 3. Attention heatmap
    plt.subplot(1, 3, 3)
    if len(attention_weights.shape) > 2:
        attention_weights = attention_weights.mean(axis=0)
    sns.heatmap(attention_weights, cmap='YlOrRd')
    plt.title("Attention Weights")
    plt.xlabel("Molecule Position")
    plt.ylabel("Attention Head")
    
    # Save the visualization
    output_path = os.path.join(output_dir, f"interaction_visualization_{len(os.listdir(output_dir))}.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_path
