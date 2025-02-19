import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

def plot_contact_map(contact_map, title="Protein Contact Map", save_path=None):
    """Visualize the protein contact map."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(contact_map, cmap='viridis', square=True)
    plt.title(title)
    plt.xlabel("Residue Position")
    plt.ylabel("Residue Position")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_molecule(smiles, title="Drug Molecule", save_path=None):
    """Visualize the drug molecule from SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Error: Could not parse SMILES string: {smiles}")
        return None
    
    # Generate 2D coordinates for the molecule
    AllChem.Compute2DCoords(mol)
    
    # Create the image with white background
    drawer = Draw.rdDepictor.MolDraw2DCairo(800, 800)
    drawer.drawOptions().clearBackground = True
    Draw.MolToImage(mol, size=(800, 800))
    
    plt.figure(figsize=(8, 8))
    img = Draw.MolToImage(mol)
    plt.imshow(img)
    plt.axis('off')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()
    
    return img

def plot_attention_weights(attention_weights, save_path=None):
    """Visualize attention weights."""
    # Reshape attention weights if needed
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.cpu().detach().numpy()
    
    # Ensure we're working with 2D array
    if len(attention_weights.shape) > 2:
        # If we have multiple attention heads, average them
        attention_weights = attention_weights.mean(axis=0)
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(attention_weights, cmap='viridis', cbar_kws={'label': 'Attention Weight'})
    plt.title("Attention Weights Heatmap")
    plt.xlabel("Sequence Position")
    plt.ylabel("Attention Head")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_full_visualization(smiles, contact_map, attention_weights, output_dir, prefix="sample"):
    """Create a full visualization of input data and model attention."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set figure style
    plt.style.use('default')
    
    # Create a combined figure
    fig = plt.figure(figsize=(20, 6))
    
    # 1. Molecule visualization
    plt.subplot(1, 3, 1)
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        AllChem.Compute2DCoords(mol)
        img = Draw.MolToImage(mol)
        plt.imshow(img)
        plt.axis('off')
        plt.title("Drug Molecule Structure")
    
    # 2. Contact map visualization
    plt.subplot(1, 3, 2)
    sns.heatmap(contact_map, cmap='viridis', square=True)
    plt.title("Protein Contact Map")
    plt.xlabel("Residue Position")
    plt.ylabel("Residue Position")
    
    # 3. Attention weights visualization
    plt.subplot(1, 3, 3)
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.cpu().detach().numpy()
    if len(attention_weights.shape) > 2:
        attention_weights = attention_weights.mean(axis=0)
    sns.heatmap(attention_weights, cmap='viridis')
    plt.title("Attention Weights")
    plt.xlabel("Sequence Position")
    plt.ylabel("Attention Head")
    
    # Adjust layout and save
    plt.tight_layout()
    combined_path = os.path.join(output_dir, f"{prefix}_combined.png")
    plt.savefig(combined_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return combined_path
