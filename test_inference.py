import os
import re
import random
import torch
import requests
import matplotlib.pyplot as plt
from PIL import Image

# RDKit imports for 2D drug drawing
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

# Import your custom modules (ensure these are in your PYTHONPATH)
from model import DrugVQA, ResidualBlock
from dataPre import ProDataset, getTrainDataSet, getSeqContactDict, getLetters
from utils import make_variables, make_variables_seq
from interactive_viz import create_interactive_visualization, create_protein_ligand_interaction
from protein_structure import ProteinStructurePredictor

########################################
# Helper: Sanitize file names
########################################
def sanitize_filename(filename):
    """
    Remove any characters that are not alphanumeric, underscore, or hyphen.
    """
    return re.sub(r'[^\w\-]', '_', filename)

########################################
# Helper Function: Download PDB from RCSB
########################################
def download_pdb(pdb_id, output_dir, protein_name=None):
    """
    Downloads a PDB file from RCSB using the given pdb_id and saves it to output_dir.
    The saved file name will be 'protein_<protein_name>_<PDBID>.pdb' if protein_name is provided,
    otherwise just '<PDBID>.pdb'. If the file already exists, it is not re-downloaded.
    """
    if protein_name:
        sanitized_name = sanitize_filename(protein_name)
        filename = f"protein_{sanitized_name}_{pdb_id.upper()}.pdb"
    else:
        filename = f"{pdb_id.upper()}.pdb"
    pdb_file = os.path.join(output_dir, filename)
    if os.path.exists(pdb_file):
        print(f"PDB file for {pdb_id} already exists: {pdb_file}")
        return pdb_file
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    print(f"Downloading PDB from {url} ...")
    response = requests.get(url)
    if response.status_code == 200:
        with open(pdb_file, 'w') as f:
            f.write(response.text)
        print(f"PDB file downloaded and saved to {pdb_file}")
        return pdb_file
    else:
        raise Exception(f"Failed to download PDB file for {pdb_id}. HTTP status code: {response.status_code}")

########################################
# 1. 3D Visualization using NGL.js (Separate Components)
########################################
def generate_ngl_visualization(protein_name, protein_seq, smiles, output_dir, index, pdb_id=None):
    """
    Generates an HTML file that uses NGL.js for interactive 3D visualization.
    It inlines the protein's PDB content (via a Blob URL) and loads the ligand from its SMILES.
    
    If 'pdb_id' is provided, it downloads the corresponding PDB file (using a sanitized file name).
    Otherwise, it uses ProteinStructurePredictor to generate a PDB from the protein sequence.
    """
    html_path = os.path.join(output_dir, f"ngl_visualization_{index}.html")
    try:
        # Determine which PDB file to use
        if pdb_id:
            pdb_path = download_pdb(pdb_id, output_dir, protein_name)
        else:
            predictor = ProteinStructurePredictor()
            pdb_path = predictor.predict_structure(protein_seq, protein_name)
        
        with open(pdb_path, 'r') as pdb_file:
            # Replace any backticks to avoid breaking the template literal in JS
            pdb_content = pdb_file.read().replace("`", "'")
        
        # Build HTML content using NGL.js (inlining the PDB content via a Blob)
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>3D Visualization - {protein_name}</title>
    <script src="https://cdn.jsdelivr.net/npm/ngl@0.10.4-1/dist/ngl.js"></script>
    <style>
        #viewport {{
            width: 800px;
            height: 600px;
            position: relative;
        }}
        .controls {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(255,255,255,0.7);
            padding: 10px;
            border-radius: 5px;
        }}
        button {{
            margin: 2px;
        }}
    </style>
</head>
<body>
    <div id="viewport"></div>
    <div class="controls">
        <button onclick="cartoon()">Cartoon</button>
        <button onclick="surface()">Surface</button>
        <button onclick="ballAndStick()">Ball+Stick</button>
        <button onclick="toggleLigand()">Toggle Ligand</button>
    </div>
    <script>
        var stage = new NGL.Stage("viewport", {{backgroundColor:"white"}});
        var structureComponent;
        var ligandComponent;
        
        // Create a Blob URL for the inlined PDB content
        var pdbContent = `{pdb_content}`;
        var blob = new Blob([pdbContent], {{ type: 'text/plain' }});
        var url = URL.createObjectURL(blob);
        
        // Load protein structure from the Blob URL
        stage.loadFile(url, {{ext:"pdb"}}).then(function(o) {{
            structureComponent = o;
            o.addRepresentation("cartoon", {{
                color: "chainid",
                opacity: 0.9,
                quality: "high"
            }});
            o.addRepresentation("surface", {{
                sele: "polymer",
                opacity: 0.1,
                quality: "high",
                visibility: false
            }});
            stage.autoView();
        }});
        
        // Load ligand from SMILES
        var smilesString = `{smiles}`;
        stage.loadFile("data:chemical/smiles," + smilesString).then(function(o) {{
            ligandComponent = o;
            o.addRepresentation("ball+stick", {{
                multipleBond: true,
                quality: "high"
            }});
            stage.autoView();
        }});
        
        // UI functions to change representations
        function cartoon() {{
            structureComponent.removeAllRepresentations();
            structureComponent.addRepresentation("cartoon", {{
                color: "chainid",
                opacity: 0.9,
                quality: "high"
            }});
        }}
        function surface() {{
            structureComponent.removeAllRepresentations();
            structureComponent.addRepresentation("surface", {{
                opacity: 0.5,
                quality: "high"
            }});
        }}
        function ballAndStick() {{
            structureComponent.removeAllRepresentations();
            structureComponent.addRepresentation("ball+stick", {{
                quality: "high"
            }});
        }}
        function toggleLigand() {{
            if (ligandComponent) {{
                ligandComponent.setVisibility(!ligandComponent.visible);
            }}
        }}
    </script>
</body>
</html>
"""
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        return html_path
        
    except Exception as e:
        print("Error generating NGL.js visualization:", e)
        return None

########################################
# 2D Visualization Function
########################################
def generate_2d_visualization(smiles, protein_name, binding_site, output_dir, index, attention_weights=None):
    """
    Generate a 2D visualization of the drug molecule with highlighted interaction sites.
    Returns the path to the saved image.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Process the drug molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None and attention_weights is not None:
        # Generate 2D coordinates for the molecule
        AllChem.Compute2DCoords(mol)
        
        # Process attention weights to get per-atom scores
        atom_attention = attention_weights.mean(axis=0) if len(attention_weights.shape) > 2 else attention_weights
        atom_attention = atom_attention.mean(axis=1)  # Average across protein positions
        
        # Normalize attention scores to [0, 1]
        atom_attention = (atom_attention - atom_attention.min()) / (atom_attention.max() - atom_attention.min())
        
        # Get atoms to highlight (those with attention score above threshold)
        threshold = 0.3  # Adjust this threshold as needed
        highlight_atoms = []
        highlight_colors = []
        highlight_radii = {}
        
        for i, score in enumerate(atom_attention):
            if i < mol.GetNumAtoms() and score > threshold:
                highlight_atoms.append(i)
                # Convert score to pink intensity
                pink_intensity = (255, int(192 * score), int(203 * score))
                highlight_colors.append(pink_intensity)
                highlight_radii[i] = score * 0.5  # Adjust radius based on attention score
        
        # Create custom drawer
        d2d = Draw.rdDepictor.Compute2DCoords(mol)
        drawer = Draw.MolDraw2DCairo(800, 800)  # Increased size for better visibility
        drawer.drawOptions().bondLineWidth = 2
        drawer.drawOptions().minFontSize = 12  # Increased font size
        
        # Draw the molecule with highlights
        Draw.MolToImage(mol, size=(800, 800), 
                       highlightAtoms=highlight_atoms if highlight_atoms else None,
                       highlightColor=(1, 0.75, 0.8) if highlight_atoms else None,  # Light pink
                       highlightRadius=0.4)
        
        # Convert to PIL Image and display in matplotlib
        img = Draw.MolToImage(mol, size=(800, 800),
                             highlightAtoms=highlight_atoms if highlight_atoms else None,
                             highlightColor=(1, 0.75, 0.8) if highlight_atoms else None,
                             highlightRadius=0.4)
        
        ax.imshow(img)
        ax.axis('off')
        
        # Add title
        plt.title(f"Drug Structure with Interaction Sites\n{protein_name}", pad=20)
        
        # Save the figure
        output_path = os.path.join(output_dir, f"drug_interaction_sites_{index}.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
        plt.close()
        
        return output_path
    
    return None

########################################
# Combined Visualization Wrapper
########################################
def generate_combined_visualizations(protein_info, smiles, output_dir="visualizations", index=1, attention_weights=None):
    """
    Generate three interactive visualizations:
      1. A 3D drug visualization (py3Dmol) via create_interactive_visualization.
      2. A protein–ligand interaction view (py3Dmol) via create_protein_ligand_interaction.
      3. A 3D view using NGL.js via generate_ngl_visualization.
    
    The protein_info dict should contain:
       - "name": Protein name.
       - "sequence": Protein sequence.
       - "pdb_id": Protein PDB ID.
    
    Returns a tuple of (interactive_drug_html, protein_ligand_html, ngl_html)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Interactive drug visualization (py3Dmol)
    interactive_drug = create_interactive_visualization(
         smiles=smiles,
         attention_weights=attention_weights,
         output_dir=output_dir,
         prefix=f"sample_{index}"
    )
    
    # 3. NGL.js visualization
    ngl_visualization = generate_ngl_visualization(
         protein_name=protein_info["name"],
         protein_seq=protein_info["sequence"],
         smiles=smiles,
         output_dir=output_dir,
         index=index,
         pdb_id=protein_info.get("pdb_id")
    )
    
    return interactive_drug , ngl_visualization

########################################
# Model Loading Function
########################################
def load_model():
    args = {
        'batch_size': 1,
        'lstm_hid_dim': 64,
        'd_a': 32,
        'r': 10,
        'n_chars_smi': 247,
        'n_chars_seq': 21,
        'dropout': 0.2,
        'in_channels': 8,
        'cnn_channels': 16,
        'cnn_layers': 5,
        'emb_dim': 30,
        'dense_hid': 100,
        'task_type': 0,
        'n_classes': 1
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = DrugVQA(args, ResidualBlock).to(device)
    model_path = os.path.join('model_pkl', 'DUDE', 'DUDE30Res-fold3-50.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    print(f"Loading model weights from: {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device

########################################
# Main Testing Function
########################################
def test_model():
    model, device = load_model()
    
    print("Loading sample data...")
    trainFoldPath = './data/DUDE/dataPre/DUDE-foldTrain3'
    contactPath = './data/DUDE/contactMap'
    contactDictPath = './data/DUDE/dataPre/DUDE-contactDict'
    smileLettersPath = './data/DUDE/voc/combinedVoc-wholeFour.voc'
    seqLettersPath = './data/DUDE/voc/sequence.voc'
    
    trainDataSet = getTrainDataSet(trainFoldPath)
    seqContactDict = getSeqContactDict(contactPath, contactDictPath)
    smiles_letters = getLetters(smileLettersPath)
    sequence_letters = getLetters(seqLettersPath)
    
    protein_seq_dict = {}
    for protein_name, contact_data in seqContactDict.items():
        if isinstance(contact_data, dict) and 'sequence' in contact_data:
            protein_seq_dict[protein_name] = contact_data['sequence']
    
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Select sample indices for demonstration (here we use one sample for brevity)
    total_samples = len(trainDataSet)
    random_indices = [1]  # Change or extend this list for more samples
    dataset = ProDataset(dataSet=[trainDataSet[i] for i in random_indices], seqContactDict=seqContactDict)
    
    print(f"\nRunning inference on {len(dataset)} sample(s) out of {total_samples} total:")
    with torch.no_grad():
        for i in range(len(dataset)):
            smiles, contact_map, label = dataset[i]
            protein_name = trainDataSet[random_indices[i]][1]
            protein_seq = protein_seq_dict.get(protein_name, "")
            
            smiles_tensor, smiles_lengths, _ = make_variables([smiles], torch.tensor([]), smiles_letters)
            contact_map = torch.FloatTensor(contact_map).unsqueeze(0).to(device)
            
            output, attention = model(smiles_tensor, contact_map)
            prediction = output.item()
            
            print(f"\nSample {i+1}:")
            print(f"Protein: {protein_name}")
            print(f"SMILES (first 50 chars): {smiles[:50]}...")
            print(f"True Label: {label}")
            print(f"Predicted Score: {prediction:.4f}")
            predicted_class = 1 if prediction >= 0.5 else 0
            print(f"Prediction: {'Binding' if predicted_class == 1 else 'Non-binding'}")
            
            # For visualization purposes, we set a PDB id.
            pdb_id_for_visuals = "6LU7"
            
            # Create a protein info dictionary for the combined visualizations
            protein_info = {
                "name": protein_name,
                "sequence": protein_seq,
                "pdb_id": pdb_id_for_visuals
            }
            
            # Generate all visualizations
            interactive_drug, ngl_html = generate_combined_visualizations(
                protein_info=protein_info,
                smiles=smiles,
                output_dir=output_dir,
                index=i+1,
                attention_weights=attention[0].cpu().numpy()
            )
            
            # Generate only the drug visualization with interaction sites
            vis_2d_path = generate_2d_visualization(
                smiles=smiles,
                protein_name=protein_name,
                binding_site=None,  # Not needed for this visualization
                output_dir=output_dir,
                index=i+1,
                attention_weights=attention[0].cpu().numpy()
            )
            
            if interactive_drug:
                print(f"Interactive drug visualization saved to: {interactive_drug}")
            if ngl_html:
                print(f"NGL.js visualization saved to: {ngl_html}")
            if vis_2d_path:
                print(f"Drug interaction visualization saved to: {vis_2d_path}")
            
            print("-" * 50)

if __name__ == "__main__":
    test_model()
