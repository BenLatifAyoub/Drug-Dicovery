import os
import py3Dmol
import tempfile
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio.PDB import *
from Bio import PDB
import requests

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
    pdb_list = PDB.PDBList()
    try:
        pdb_file = pdb_list.retrieve_pdb_file(pdb_id.lower(), pdir='.', file_format='pdb')
        return pdb_file
    except:
        print(f"Failed to download PDB structure for {pdb_id}")
        return None

def save_view_to_html(view, output_path, pdb_data, style_commands):
    """Save py3Dmol view to HTML file."""
    html_content = f'''
    <html>
    <head>
        <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <style>
            .mol-container {{ 
                width: 800px;
                height: 600px;
                position: relative;
            }}
        </style>
    </head>
    <body>
        <div id="container" class="mol-container"></div>
        <script>
            let viewer = $3Dmol.createViewer($("#container"), {{
                backgroundColor: "white"
            }});
            let pdbData = `{pdb_data}`;
            viewer.addModel(pdbData, "pdb");
            {style_commands}
            viewer.zoomTo();
            viewer.render();
        </script>
    </body>
    </html>
    '''
    
    with open(output_path, 'w') as f:
        f.write(html_content)

def create_interactive_visualization(smiles, attention_weights=None, output_dir="interactive_visualizations", prefix="interaction"):
    """
    Create an interactive 3D visualization of the drug molecule.
    """
    try:
        print(f"\nCreating visualization in directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare drug molecule
        print("Preparing drug molecule...")
        mol = prepare_molecule_3d(smiles)
        if mol is None:
            print("Failed to prepare drug molecule")
            return None
        
        # Convert molecule to PDB format
        print("Converting to PDB format...")
        pdb_path = os.path.join(output_dir, "temp_drug.pdb")
        writer = Chem.PDBWriter(pdb_path)
        writer.write(mol)
        writer.close()
        
        # Read PDB data
        with open(pdb_path, 'r') as f:
            pdb_data = f.read()
        
        # Process attention weights if provided
        style_commands = []
        if attention_weights is not None:
            print("Processing attention weights...")
            if len(attention_weights.shape) > 2:
                attention_weights = attention_weights.mean(axis=0)
            
            # Get number of atoms
            num_atoms = mol.GetNumAtoms()
            print(f"Number of atoms: {num_atoms}")
            
            # Resize attention weights if necessary
            if attention_weights.shape[0] > num_atoms:
                resized_attention = np.zeros(num_atoms)
                for i in range(num_atoms):
                    start_idx = int(i * attention_weights.shape[0] / num_atoms)
                    end_idx = int((i + 1) * attention_weights.shape[0] / num_atoms)
                    resized_attention[i] = attention_weights[start_idx:end_idx].mean()
                atom_attention = resized_attention
            else:
                atom_attention = np.pad(attention_weights.mean(axis=1), 
                                      (0, num_atoms - attention_weights.shape[0]),
                                      mode='constant')
            
            # Normalize to [0,1]
            atom_attention = (atom_attention - atom_attention.min()) / (atom_attention.max() - atom_attention.min() + 1e-6)
            
            # Convert to Python float for JSON serialization
            atom_attention = [float(x) for x in atom_attention]
            
            # Generate style commands for each atom
            for i, atom in enumerate(mol.GetAtoms()):
                color = f'rgb({int(255*atom_attention[i])},0,{int(255*(1-atom_attention[i]))})'
                style_commands.append(
                    f'viewer.setStyle({{serial: {i+1}}}, {{stick: {{color: "{color}", radius: 0.2}}}});'
                )
        else:
            style_commands.append('viewer.setStyle({}, {stick: {colorscheme: "spectral"}});')
        
        # Save visualization
        output_path = os.path.join(output_dir, f"{prefix}_interactive.html")
        print(f"Saving visualization to: {output_path}")
        save_view_to_html(None, output_path, pdb_data, '\n            '.join(style_commands))
        
        # Verify file was created
        if os.path.exists(output_path):
            print(f"File successfully created at: {output_path}")
            file_size = os.path.getsize(output_path)
            print(f"File size: {file_size} bytes")
        else:
            print(f"Warning: File was not created at: {output_path}")
        
        # Clean up temporary files
        os.remove(pdb_path)
        print("Temporary files cleaned up")
        
        return output_path
        
    except Exception as e:
        print(f"Error in create_interactive_visualization: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

def create_protein_ligand_interaction(protein_pdb_id, smiles, attention_weights=None, output_dir="interactive_visualizations", prefix="complex"):
    """
    Create an interactive visualization of protein-ligand complex with binding site highlight.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Get protein structure
        print(f"Downloading PDB structure '{protein_pdb_id}'...")
        pdb_file = get_protein_structure(protein_pdb_id)
        if not pdb_file:
            return None
        
        # Prepare ligand
        mol = prepare_molecule_3d(smiles)
        if mol is None:
            print("Failed to prepare ligand")
            return None
        
        # Convert ligand to PDB format
        pdb_path = os.path.join(output_dir, "temp_ligand.pdb")
        writer = Chem.PDBWriter(pdb_path)
        writer.write(mol)
        writer.close()
        
        # Read protein and ligand PDB data
        with open(pdb_file, 'r') as f:
            protein_data = f.read()
        with open(pdb_path, 'r') as f:
            ligand_data = f.read()
        
        # Combine PDB data
        combined_data = protein_data + "\n" + ligand_data
        
        # Generate style commands
        style_commands = []
        
        # Style protein
        style_commands.append('''
            viewer.setStyle({model: 0}, {
                cartoon: {color: "spectrum", opacity: 0.8},
                stick: {radius: 0.2, opacity: 0.6}
            });
        ''')
        
        # Style ligand
        if attention_weights is not None:
            # Process attention weights
            if len(attention_weights.shape) > 2:
                attention_weights = attention_weights.mean(axis=0)
            
            # Get number of atoms
            num_atoms = mol.GetNumAtoms()
            
            # Resize attention weights if necessary
            if attention_weights.shape[0] > num_atoms:
                resized_attention = np.zeros(num_atoms)
                for i in range(num_atoms):
                    start_idx = int(i * attention_weights.shape[0] / num_atoms)
                    end_idx = int((i + 1) * attention_weights.shape[0] / num_atoms)
                    resized_attention[i] = attention_weights[start_idx:end_idx].mean()
                atom_attention = resized_attention
            else:
                atom_attention = np.pad(attention_weights.mean(axis=1), 
                                      (0, num_atoms - attention_weights.shape[0]),
                                      mode='constant')
            
            # Normalize to [0,1]
            atom_attention = (atom_attention - atom_attention.min()) / (atom_attention.max() - atom_attention.min() + 1e-6)
            
            # Convert to Python float for JSON serialization
            atom_attention = [float(x) for x in atom_attention]
            
            # Color atoms by attention
            for i, atom in enumerate(mol.GetAtoms()):
                color = f'rgb({int(255*atom_attention[i])},0,{int(255*(1-atom_attention[i]))})'
                style_commands.append(f'''
                    viewer.setStyle({{model: 1, serial: {i+1}}}, {{
                        stick: {{color: "{color}", radius: 0.3}},
                        sphere: {{color: "{color}", radius: {0.5*atom_attention[i] + 0.2}}}
                    }});
                ''')
        else:
            style_commands.append('''
                viewer.setStyle({model: 1}, {
                    stick: {colorscheme: "spectral", radius: 0.3},
                    sphere: {colorscheme: "spectral", radius: 0.4}
                });
            ''')
        
        # Add surface
        style_commands.append('viewer.addSurface($3Dmol.VDW, {opacity: 0.6, color: "white"});')
        
        # Save visualization
        output_path = os.path.join(output_dir, f"{prefix}_interactive.html")
        print(f"Saving visualization to: {output_path}")
        save_view_to_html(None, output_path, combined_data, '\n            '.join(style_commands))
        
        # Verify file was created
        if os.path.exists(output_path):
            print(f"File successfully created at: {output_path}")
            file_size = os.path.getsize(output_path)
            print(f"File size: {file_size} bytes")
        else:
            print(f"Warning: File was not created at: {output_path}")
        
        # Clean up temporary files
        os.remove(pdb_path)
        print("Temporary files cleaned up")
        
        return output_path
        
    except Exception as e:
        print(f"Error in create_protein_ligand_interaction: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None
