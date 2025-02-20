import torch
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import esm
import os

class ProteinStructurePredictor:
    def __init__(self, cache_dir="protein_structures"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load ESMFold model
        self.model = esm.pretrained.esmfold_v1()
        self.model = self.model.eval()
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def predict_structure(self, sequence, protein_name):
        """Predict protein structure from sequence using ESMFold."""
        cache_file = os.path.join(self.cache_dir, f"{protein_name}.pdb")
        
        # Check if structure is already cached
        if os.path.exists(cache_file):
            return cache_file
            
        # Predict structure
        with torch.no_grad():
            output = self.model.infer_pdb(sequence)
            
        # Save structure to PDB file
        with open(cache_file, 'w') as f:
            f.write(output)
            
        return cache_file

