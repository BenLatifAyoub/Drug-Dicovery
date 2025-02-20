
import os
from rdkit import Chem
from rdkit.Chem import AllChem

def create_integrated_visualization(protein_pdb_path, smiles_str, output_path):
    """Create an integrated visualization of protein and drug molecule"""
    
    # Convert SMILES to 3D structure
    mol = Chem.MolFromSmiles(smiles_str)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    
    # Convert to PDB format
    drug_pdb = Chem.MolToPDBBlock(mol)
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Protein-Drug Interaction Visualization</title>
    <script src="https://cdn.jsdelivr.net/npm/ngl@0.10.4-1/dist/ngl.js"></script>
    <style>
        body {{ margin: 0; overflow: hidden; }}
        #viewport {{ width: 100vw; height: 100vh; }}
        .control-panel {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(255, 255, 255, 0.7);
            padding: 10px;
            border-radius: 5px;
            z-index: 100;
        }}
        button {{
            margin: 2px;
            padding: 5px 10px;
            cursor: pointer;
        }}
    </style>
</head>
<body>
    <div id="viewport"></div>
    <div class="control-panel">
        <button onclick="toggleProtein()">Toggle Protein</button>
        <button onclick="toggleDrug()">Toggle Drug</button>
        <button onclick="showCartoon()">Cartoon View</button>
        <button onclick="showSurface()">Surface View</button>
        <button onclick="focusBinding()">Focus Binding Site</button>
        <button onclick="resetView()">Reset View</button>
    </div>
    <script>
        // Initialize viewer
        var stage = new NGL.Stage("viewport", {{backgroundColor: "white"}});
        var proteinComponent, drugComponent;
        
        // Load protein structure
        stage.loadFile("{protein_pdb_path}").then(function (o) {{
            proteinComponent = o;
            // Add protein representations
            o.addRepresentation("cartoon", {{
                color: "chainid",
                opacity: 0.9
            }});
            o.addRepresentation("surface", {{
                opacity: 0.3,
                color: "resname",
                visible: false
            }});
            stage.autoView();
        }});

        // Load drug structure from PDB string
        var drugBlob = new Blob([`{drug_pdb}`], {{type: 'text/plain'}});
        stage.loadFile(drugBlob, {{ext: "pdb"}}).then(function (o) {{
            drugComponent = o;
            o.addRepresentation("ball+stick", {{
                multipleBond: true,
                colorScheme: "element"
            }});
            
            // Position drug near protein center
            var proteinCenter = new NGL.Vector3();
            proteinComponent.structure.getCenter(proteinCenter);
            o.setPosition(proteinCenter.add(new NGL.Vector3(10, 0, 0)));
            stage.autoView();
        }});

        // UI Functions
        function toggleProtein() {{
            proteinComponent.toggleVisibility();
        }}
        
        function toggleDrug() {{
            drugComponent.toggleVisibility();
        }}
        
        function showCartoon() {{
            proteinComponent.removeAllRepresentations();
            proteinComponent.addRepresentation("cartoon", {{
                color: "chainid",
                opacity: 0.9
            }});
        }}
        
        function showSurface() {{
            proteinComponent.removeAllRepresentations();
            proteinComponent.addRepresentation("surface", {{
                opacity: 0.5,
                color: "resname"
            }});
        }}
        
        function focusBinding() {{
            var center = new NGL.Vector3();
            drugComponent.structure.getCenter(center);
            stage.viewerControls.moveTo(center, 1000);
        }}
        
        function resetView() {{
            stage.autoView();
        }}
        
        // Handle window resizing
        window.addEventListener("resize", function() {{
            stage.handleResize();
        }});
    </script>
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html_content)
    
    return output_path

def main():
    # Example usage
    protein_pdb = "6LU7.pdb"  # Your protein PDB file path
    smiles = "CC1=C([C@H](NC(=S)N1)c2ccc(cc2)OC)C(=O)N/N=C/c3ccccc3F"  # Your drug SMILES string
    
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "integrated_view.html")
    viz_path = create_integrated_visualization(
        protein_pdb_path=protein_pdb,
        smiles_str=smiles,
        output_path=output_path
    )
    
    print(f"Visualization saved to: {viz_path}")

if __name__ == "__main__":
    main()
