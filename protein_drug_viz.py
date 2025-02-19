import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdDepictor
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_molecule_2d(smiles):
    """Prepare a 2D representation of the molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Generate 2D coordinates
    rdDepictor.Compute2DCoords(mol)
    return mol

def create_protein_drug_interaction(smiles, protein_seq, contact_map, attention_weights, output_dir, prefix="interaction"):
    """
    Create a comprehensive visualization showing:
    1. Drug molecule
    2. Protein contact map
    3. Attention-based interaction between drug and protein
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure
    fig = plt.figure(figsize=(20, 10))
    
    # 1. Drug molecule (left panel)
    ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=2)
    mol = prepare_molecule_2d(smiles)
    if mol is not None:
        img = Draw.MolToImage(mol)
        ax1.imshow(img)
        ax1.axis('off')
        ax1.set_title("Drug Structure", fontsize=12, pad=10)
    
    # 2. Protein contact map (right panel)
    ax2 = plt.subplot2grid((2, 4), (0, 2), colspan=2)
    contact_map_plot = sns.heatmap(contact_map, cmap='viridis', ax=ax2)
    plt.colorbar(contact_map_plot.collections[0], ax=ax2, label='Contact Strength')
    ax2.set_title("Protein Contact Map", fontsize=12, pad=10)
    
    # 3. Interaction visualization (bottom panel)
    ax3 = plt.subplot2grid((2, 4), (1, 0), colspan=4)
    
    # Process attention weights
    if len(attention_weights.shape) > 2:
        attention_weights = attention_weights.mean(axis=0)  # Average over heads
    
    # Create interaction heatmap
    im = ax3.imshow(attention_weights, aspect='auto', cmap='YlOrRd')
    plt.colorbar(im, ax=ax3, label='Attention Weight')
    
    # Add labels
    ax3.set_xlabel("Protein Sequence Position", fontsize=10)
    ax3.set_ylabel("Drug SMILES Position", fontsize=10)
    ax3.set_title("Drug-Protein Interaction Map", fontsize=12, pad=10)
    
    # Highlight top interactions
    n_top = 5  # Number of top interactions to highlight
    attention_flat = attention_weights.flatten()
    top_indices = np.argsort(attention_flat)[-n_top:]
    top_positions = np.unravel_index(top_indices, attention_weights.shape)
    
    # Add markers for top interactions
    for y, x in zip(*top_positions):
        ax3.plot(x, y, 'o', color='blue', markersize=10, alpha=0.5)
    
    # Add overall title
    plt.suptitle(f"Drug-Protein Interaction Analysis\nProtein Length: {len(protein_seq)}", 
                fontsize=14, y=0.95)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save visualization
    output_path = os.path.join(output_dir, f"{prefix}_protein_drug_interaction.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def create_detailed_interaction_view(smiles, protein_seq, attention_weights, output_dir, prefix="detailed"):
    """
    Create a detailed view of specific interactions between drug atoms and protein residues
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Process attention weights
    if len(attention_weights.shape) > 2:
        attention_weights = attention_weights.mean(axis=0)
    
    # Find top interactions
    n_top = 3  # Number of top interactions to show
    attention_flat = attention_weights.flatten()
    top_indices = np.argsort(attention_flat)[-n_top:]
    top_positions = np.unravel_index(top_indices, attention_weights.shape)
    
    # Create figure
    fig = plt.figure(figsize=(15, 8))
    
    # 1. Drug molecule with highlighted atoms
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
    mol = prepare_molecule_2d(smiles)
    if mol is not None:
        # Highlight atoms involved in top interactions
        highlight_atoms = [int(x) for x in list(set(top_positions[0]))]  # Convert to int
        atom_colors = {int(atom): (1, 0, 0) for atom in highlight_atoms}  # Red color for highlighted atoms
        img = Draw.MolToImage(mol, highlightAtoms=highlight_atoms,
                            highlightColor=None,
                            highlightAtomColors=atom_colors)
        ax1.imshow(img)
        ax1.axis('off')
        ax1.set_title("Drug Structure\n(Red atoms show strong interactions)", fontsize=12)
    
    # 2. Interaction strength plot
    ax2 = plt.subplot2grid((2, 3), (1, 0), colspan=3)
    attention_plot = sns.heatmap(attention_weights, cmap='YlOrRd', ax=ax2)
    ax2.set_title("Interaction Strength Map", fontsize=12)
    ax2.set_xlabel("Protein Sequence Position")
    ax2.set_ylabel("Drug SMILES Position")
    
    # Add markers for top interactions
    for y, x in zip(*top_positions):
        ax2.plot(x, y, 'o', color='blue', markersize=10, alpha=0.5)
        
        # Add text annotation for protein residue if sequence is available
        if protein_seq and x < len(protein_seq):
            residue = protein_seq[x]
            ax2.text(x, y, residue, color='white', ha='center', va='center')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save visualization
    output_path = os.path.join(output_dir, f"{prefix}_detailed_interaction.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path
