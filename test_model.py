import torch
from model import DrugVQA, ResidualBlock
import os

def analyze_checkpoint():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join('model_pkl', 'DUDE', 'DUDE30Res-fold3-50.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    
    # Load the state dict with device mapping
    state_dict = torch.load(model_path, map_location=device)
    
    # Print the keys and shapes of the state dict
    print("\nModel Architecture from checkpoint:")
    print("-" * 50)
    
    # Group parameters by layers
    layer_params = {}
    for key, value in state_dict.items():
        layer_name = key.split('.')[0]
        if layer_name not in layer_params:
            layer_params[layer_name] = []
        layer_params[layer_name].append((key, value.shape))
    
    # Print organized by layer
    for layer_name, params in sorted(layer_params.items()):
        print(f"\n{layer_name}:")
        for param_name, shape in sorted(params):
            print(f"  {param_name}: {shape}")

def load_model_and_test():
    # Define model parameters to match the checkpoint architecture
    args = {
        'batch_size': 1,
        'lstm_hid_dim': 64,  # From lstm.weight_hh_l0: [256, 64]
        'd_a': 32,  # From linear_first.weight: [32, 128]
        'r': 10,  # From linear_second.weight: [10, 32]
        'n_chars_smi': 247,  # From embeddings.weight: [247, 30]
        'n_chars_seq': 21,  # From seq_embed.weight: [21, 30]
        'dropout': 0.2,
        'in_channels': 8,  # From bn.weight: [8]
        'cnn_channels': 16,  # First layer uses 16 channels
        'cnn_layers': 5,  # Number of blocks in each layer
        'emb_dim': 30,  # From embeddings.weight: [247, 30]
        'dense_hid': 100,  # From linear_final.weight: [1, 100]
        'task_type': 0,
        'n_classes': 1  # From linear_final.weight: [1, 100]
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
    
    print("Model loaded successfully!")
    return model

if __name__ == "__main__":
    print("Analyzing checkpoint architecture...")
    analyze_checkpoint()
    try:
        model = load_model_and_test()
        print("Model architecture matches the checkpoint!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
