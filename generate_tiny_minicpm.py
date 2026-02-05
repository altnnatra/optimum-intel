import torch
import os
from transformers import Qwen2Config, Qwen2ForCausalLM, AutoTokenizer

def generate_tiny_minicpm():
    # weight = vocab_size * hidden_size
    config = Qwen2Config(
        vocab_size=1000,         
        hidden_size=64,          
        intermediate_size=128,   
        num_hidden_layers=2,     
        num_attention_heads=4,   
        num_key_value_heads=2,   
        max_position_embeddings=512,
        model_type="minicpmo"    
    )

    print("Creating model Qwen2-based MiniCPM...")
    model = Qwen2ForCausalLM(config)

    save_path = "tiny-model-local"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model.save_pretrained(save_path)
    print("Adding tokenizer files...")
    try:
        # take the tokenizator from the original model
        tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM-o-2_6", trust_remote_code=True)
        tokenizer.save_pretrained(save_path)
        print("Tokenizer added!")
    except Exception as e:
        print(f"Could not add original tokenizer, creating a blank one: {e}")
        
    print(f"Done! Model is saved here: {save_path}")
    
    # finding size
    size_mb = os.path.getsize(os.path.join(save_path, "model.safetensors")) / (1024 * 1024)
    print(f"Size (approximately) {size_mb:.2f} MB")

if __name__ == "__main__":
    generate_tiny_minicpm()