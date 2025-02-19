import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from Bio.PDB import *
import pymol
from pymol import cmd
import tempfile
import py3Dmol
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def prepare_molecule_3d(smiles):
    """Generate 3D conformer for the drug molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Add hydrogens
    mol = Chem.AddHs(mol)
    
    # Generate 3D conformer
    try:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        return mol
    except:
        print("Failed to generate 3D conformer, trying alternative method...")
        try:
            AllChem.EmbedMolecule(mol, maxAttempts=1000, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            return mol
        except:
            print("Failed to generate 3D conformer")
            return None

def get_protein_structure(pdb_id):
    """Download and prepare protein structure."""
    pdbl = PDBList()
    pdb_file = pdbl.retrieve_pdb_file(pdb_id, pdir='./temp', file_format='pdb')
    return pdb_file

def create_3d_interaction_view(smiles, pdb_id, attention_weights, output_dir, prefix="3d_interaction"):
    """
    Create a 3D visualization of drug-protein interaction using py3Dmol.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare drug molecule
    mol = prepare_molecule_3d(smiles)
    if mol is None:
        print("Failed to generate 3D conformer for the drug molecule")
        return None
    
    # Save drug molecule as PDB
    drug_pdb = os.path.join(output_dir, f"{prefix}_drug.pdb")
    writer = Chem.PDBWriter(drug_pdb)
    writer.write(mol)
    writer.close()
    
    # Get protein structure
    protein_pdb = get_protein_structure(pdb_id)
    
    # Create py3Dmol visualization
    view = py3Dmol.view(width=800, height=600)
    
    # Load protein
    with open(protein_pdb, 'r') as f:
        protein_data = f.read()
    view.addModel(protein_data, "pdb")
    
    # Load drug
    with open(drug_pdb, 'r') as f:
        drug_data = f.read()
    view.addModel(drug_data, "pdb")
    
    # Style protein
    view.setStyle({'model': 0}, {
        'cartoon': {'color': 'spectrum'},
        'surface': {'opacity': 0.3}
    })
    
    # Style drug
    view.setStyle({'model': 1}, {
        'stick': {'colorscheme': 'greenCarbon'}
    })
    
    # Process attention weights to highlight interaction sites
    if len(attention_weights.shape) > 2:
        attention_weights = attention_weights.mean(axis=0)
    
    # Find top interaction sites
    n_top = 5
    attention_flat = attention_weights.flatten()
    top_indices = np.argsort(attention_flat)[-n_top:]
    top_positions = np.unravel_index(top_indices, attention_weights.shape)
    
    # Add spheres for interaction sites
    for pos in zip(*top_positions):
        view.addSphere({
            'center': {'x': pos[0], 'y': pos[1], 'z': 0},
            'radius': 0.5,
            'color': 'red',
            'opacity': 0.7
        })
    
    # Set camera
    view.zoomTo()
    
    # Save visualization
    output_path = os.path.join(output_dir, f"{prefix}_3d.html")
    view.save(output_path)
    
    # Create a more detailed PyMOL visualization
    cmd.load(protein_pdb, "protein")
    cmd.load(drug_pdb, "drug")
    
    # Set visualization style
    cmd.hide("everything")
    cmd.show("cartoon", "protein")
    cmd.show("surface", "protein")
    cmd.set("transparency", 0.5, "protein")
    cmd.show("sticks", "drug")
    cmd.color("cyan", "drug")
    
    # Highlight interaction sites
    for i, pos in enumerate(zip(*top_positions)):
        cmd.select(f"interaction_{i}", f"resi {pos[1]} and protein")
        cmd.show("spheres", f"interaction_{i}")
        cmd.color("red", f"interaction_{i}")
    
    # Save PyMOL session and image
    cmd.save(os.path.join(output_dir, f"{prefix}_session.pse"))
    cmd.png(os.path.join(output_dir, f"{prefix}_pymol.png"), width=1200, height=1000, dpi=300)
    cmd.delete("all")
    
    return output_path

def create_ligand_binding_site_view(smiles, pdb_id, binding_site_residues, output_dir, prefix="binding_site"):
    """
    Create a focused view of the ligand binding site.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare drug molecule
    mol = prepare_molecule_3d(smiles)
    if mol is None:
        print("Failed to generate 3D conformer for the drug molecule")
        return None
    
    # Save drug molecule
    drug_pdb = os.path.join(output_dir, f"{prefix}_drug.pdb")
    writer = Chem.PDBWriter(drug_pdb)
    writer.write(mol)
    writer.close()
    
    # Get protein structure
    protein_pdb = get_protein_structure(pdb_id)
    
    # Create PyMOL visualization
    cmd.load(protein_pdb, "protein")
    cmd.load(drug_pdb, "drug")
    
    # Select binding site residues
    binding_site_sel = " or ".join([f"resi {res}" for res in binding_site_residues])
    cmd.select("binding_site", f"protein and ({binding_site_sel})")
    
    # Set visualization style
    cmd.hide("everything")
    cmd.show("cartoon", "protein")
    cmd.color("white", "protein")
    cmd.show("surface", "binding_site")
    cmd.color("yellow", "binding_site")
    cmd.show("sticks", "drug")
    cmd.color("cyan", "drug")
    
    # Show hydrogen bonds
    cmd.distance("hbonds", "drug", "binding_site", mode=2)
    
    # Center and zoom on binding site
    cmd.zoom("binding_site", 5)
    
    # Save PyMOL session and image
    cmd.save(os.path.join(output_dir, f"{prefix}_session.pse"))
    cmd.png(os.path.join(output_dir, f"{prefix}_pymol.png"), width=1200, height=1000, dpi=300)
    cmd.delete("all")
    
    return os.path.join(output_dir, f"{prefix}_pymol.png")

def create_3d_interaction_visualization(smiles, attention_weights, output_dir, prefix="3d_interaction"):
    """
    Create a 3D visualization of the drug molecule with attention weights.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare molecule
    mol = prepare_molecule_3d(smiles)
    if mol is None:
        print("Failed to prepare 3D molecule")
        return None
    
    # Get atom coordinates
    conf = mol.GetConformer()
    positions = conf.GetPositions()
    
    # Process attention weights
    if len(attention_weights.shape) > 2:
        attention_weights = attention_weights.mean(axis=0)
    
    # Get number of atoms in molecule
    num_atoms = mol.GetNumAtoms()
    
    # Resize attention weights if necessary
    if attention_weights.shape[0] > num_atoms:
        # Take average of excess attention weights
        resized_attention = np.zeros(num_atoms)
        for i in range(num_atoms):
            start_idx = int(i * attention_weights.shape[0] / num_atoms)
            end_idx = int((i + 1) * attention_weights.shape[0] / num_atoms)
            resized_attention[i] = attention_weights[start_idx:end_idx].mean()
        atom_attention = resized_attention
    else:
        # Pad with zeros if attention weights are fewer than atoms
        atom_attention = np.pad(attention_weights.mean(axis=1), 
                              (0, num_atoms - attention_weights.shape[0]),
                              mode='constant')
    
    # Normalize attention weights to [0,1]
    atom_attention = (atom_attention - atom_attention.min()) / (atom_attention.max() - atom_attention.min() + 1e-6)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot atoms
    elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
    element_colors = {'C': 'gray', 'N': 'blue', 'O': 'red', 'S': 'yellow', 'F': 'green', 
                     'Cl': 'green', 'Br': 'brown', 'I': 'purple', 'H': 'white'}
    
    # Create scatter plots for each element type
    unique_elements = set(elements)
    scatter_plots = {}
    
    for element in unique_elements:
        element_positions = []
        element_sizes = []
        for i, (pos, el) in enumerate(zip(positions, elements)):
            if el == element:
                element_positions.append(pos)
                element_sizes.append(100 + 400 * atom_attention[i])
        
        if element_positions:
            element_positions = np.array(element_positions)
            scatter = ax.scatter(element_positions[:, 0], 
                               element_positions[:, 1], 
                               element_positions[:, 2],
                               c=element_colors.get(element, 'gray'),
                               s=element_sizes,
                               alpha=0.6,
                               label=element)
            scatter_plots[element] = scatter
    
    # Plot bonds
    for bond in mol.GetBonds():
        id1 = bond.GetBeginAtomIdx()
        id2 = bond.GetEndAtomIdx()
        pos1 = positions[id1]
        pos2 = positions[id2]
        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]], 'k-', alpha=0.3)
    
    # Add legend
    ax.legend(title="Atom Types")
    
    # Add colorbar for attention weights
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Attention Weight')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Drug Structure with Attention Weights\n(Larger spheres indicate higher attention)')
    
    # Save visualization
    output_path = os.path.join(output_dir, f"{prefix}_3d.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create additional views
    angles = [(0, 0), (90, 0), (0, 90), (45, 45)]
    fig = plt.figure(figsize=(16, 4))
    
    for i, (elev, azim) in enumerate(angles, 1):
        ax = fig.add_subplot(1, 4, i, projection='3d')
        
        # Plot atoms for each element type
        for element in unique_elements:
            element_positions = []
            element_sizes = []
            for j, (pos, el) in enumerate(zip(positions, elements)):
                if el == element:
                    element_positions.append(pos)
                    element_sizes.append(50 + 200 * atom_attention[j])
            
            if element_positions:
                element_positions = np.array(element_positions)
                ax.scatter(element_positions[:, 0], 
                          element_positions[:, 1], 
                          element_positions[:, 2],
                          c=element_colors.get(element, 'gray'),
                          s=element_sizes,
                          alpha=0.6)
        
        # Plot bonds
        for bond in mol.GetBonds():
            id1 = bond.GetBeginAtomIdx()
            id2 = bond.GetEndAtomIdx()
            pos1 = positions[id1]
            pos2 = positions[id2]
            ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]], 'k-', alpha=0.3)
        
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f'View {i}\n(elev={elev}°, azim={azim}°)')
    
    # Save multi-view visualization
    output_path_views = os.path.join(output_dir, f"{prefix}_3d_views.png")
    plt.savefig(output_path_views, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path, output_path_views

def create_interaction_surface_plot(attention_weights, output_dir, prefix="interaction_surface"):
    """
    Create a 3D surface plot showing the interaction landscape.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Process attention weights
    if len(attention_weights.shape) > 2:
        attention_weights = attention_weights.mean(axis=0)
    
    # Create coordinate grids
    x = np.arange(attention_weights.shape[0])
    y = np.arange(attention_weights.shape[1])
    X, Y = np.meshgrid(x, y)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create surface plot
    surf = ax.plot_surface(X, Y, attention_weights.T, cmap='viridis', 
                          linewidth=0, antialiased=True)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, label='Attention Weight')
    
    # Set labels and title
    ax.set_xlabel('Drug SMILES Position')
    ax.set_ylabel('Protein Sequence Position')
    ax.set_zlabel('Attention Weight')
    ax.set_title('3D Interaction Landscape')
    
    # Save visualization
    output_path = os.path.join(output_dir, f"{prefix}_surface.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path
