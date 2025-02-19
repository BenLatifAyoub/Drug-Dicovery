import torch
import random
from model import DrugVQA, ResidualBlock
from dataPre import ProDataset, getTrainDataSet, getSeqContactDict, getLetters
from utils import make_variables, make_variables_seq
from interactive_viz import create_interactive_visualization, create_protein_ligand_interaction
import os

def load_model():
    # Define model parameters to match the checkpoint architecture
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

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = DrugVQA(args, ResidualBlock)
    model = model.to(device)
    
    # Load model weights
    model_path = os.path.join('model_pkl', 'DUDE', 'DUDE30Res-fold3-50.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    
    print(f"Loading model weights from: {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, device

def test_model():
    # Load the model
    model, device = load_model()
    
    # Load sample data
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
    
    # Get protein sequences from contact map data
    protein_seq_dict = {}
    for protein_name, contact_data in seqContactDict.items():
        if isinstance(contact_data, dict) and 'sequence' in contact_data:
            protein_seq_dict[protein_name] = contact_data['sequence']
    
    # Create output directory for visualizations
    output_dir = "interactive_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Select 3 random samples
    total_samples = len(trainDataSet)
    random_indices = random.sample(range(total_samples), 3)
    
    # Create dataset with random samples
    dataset = ProDataset(dataSet=[trainDataSet[i] for i in random_indices], seqContactDict=seqContactDict)
    
    # Test inference
    print(f"\nRunning inference on 3 random samples from a total of {total_samples} samples:")
    with torch.no_grad():
        for i in range(3):
            smiles, contact_map, label = dataset[i]
            protein_name = trainDataSet[random_indices[i]][1]  # Get protein name
            protein_seq = protein_seq_dict.get(protein_name, "")  # Get protein sequence
            
            # Preprocess SMILES
            smiles_tensor, smiles_lengths, _ = make_variables([smiles], torch.tensor([]), smiles_letters)
            
            # Convert contact map to tensor
            contact_map = torch.FloatTensor(contact_map).unsqueeze(0).to(device)
            
            # Run inference
            output, attention = model(smiles_tensor, contact_map)
            prediction = output.item()
            
            print(f"\nSample {i+1}:")
            print(f"Protein: {protein_name}")
            print(f"SMILES: {smiles[:50]}...")
            print(f"True Label: {label}")
            print(f"Predicted Score: {prediction:.4f}")
            predicted_class = 1 if prediction >= 0.5 else 0
            print(f"Predicted Class: {predicted_class}")
            print(f"Prediction: {'Binding' if predicted_class == 1 else 'Non-binding'}")
            
            # Create interactive visualizations
            print("Generating interactive visualizations...")
            
            # 1. Basic drug visualization
            viz_path = create_interactive_visualization(
                smiles=smiles,
                attention_weights=attention[0].cpu().numpy(),
                output_dir=output_dir,
                prefix=f"sample_{i+1}"
            )
            if viz_path:
                print(f"Interactive drug visualization saved to: {viz_path}")
            
            # 2. Protein-ligand interaction visualization (if PDB ID is available)
            # Note: In a real application, you would need to map your protein names to PDB IDs
            # For this example, we'll use a sample PDB ID (6LU7) for demonstration
            sample_pdb = "6LU7"  # You would need to get the actual PDB ID for your protein
            complex_path = create_protein_ligand_interaction(
                protein_pdb_id=sample_pdb,
                smiles=smiles,
                attention_weights=attention[0].cpu().numpy(),
                output_dir=output_dir,
                prefix=f"sample_{i+1}_complex"
            )
            if complex_path:
                print(f"Interactive protein-ligand visualization saved to: {complex_path}")
            
            print("-" * 50)

if __name__ == "__main__":
    test_model()
