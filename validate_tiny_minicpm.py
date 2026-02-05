import torch
from transformers import AutoConfig, AutoModelForCausalLM

def validate_model():
    model_path = "./tiny-model-local"
    print(f"Loading the model from {model_path}...")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error occured while loading: {e}")
        return

    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    
    print("Forward pass in process...")
    try:
        with torch.no_grad():
            outputs = model(input_ids)
        print("Forward pass is done!")
        print(f"Output tensor shape: {outputs.logits.shape}")
    except Exception as e:
        print(f"Error occured while doing forward pass: {e}")

if __name__ == "__main__":
    validate_model()