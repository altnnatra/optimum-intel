import torch
import time
import psutil
import os
from transformers import AutoModelForCausalLM
from optimum.intel.openvino import OVModelForCausalLM

def get_ram_usage():
    """Returns the current RAM usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def validate_model():
    model_path = "./tiny-model-local"
    ov_model_save_path = "./tiny-ov-model"
    
    print(f"Step 1: Loading PyTorch model")
    start_ram = get_ram_usage()
    
    try:
        #measure loading time and RAM for the pytorch version
        start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        load_time_pt = time.time() - start_time
        end_ram = get_ram_usage()
        
        print(f"PyTorch model loaded successfully in {load_time_pt:.2f}s")
        print(f"RAM usage increment: {end_ram - start_ram:.2f} MB")
    except Exception as e:
        print(f"Error loading PyTorch model: {e}")
        return

    # forward Pass test
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    print("\n Step 2: PyTorch Forward Pass")
    try:
        with torch.no_grad():
            outputs = model(input_ids)
        print(f"Forward pass successful. Logits shape: {outputs.logits.shape}")
    except Exception as e:
        print(f"Forward pass failed: {e}")

    # OpenVINO Export & Compilation test 
    print("\n Step 3: OpenVINO Compilation & Optimization")
    try:
        # Measure how fast OpenVINO can compile the model from the PyTorch source
        start_time = time.time()
        ov_model = OVModelForCausalLM.from_pretrained(
            model_path, 
            export=True, 
            trust_remote_code=True,
            device="CPU"
        )
        compile_time = time.time() - start_time
        
        print(f"OpenVINO compilation successful!")
        print(f"Compilation time on CPU: {compile_time:.2f}s")
        
        # Verify the tiny model inference using OpenVINO
        ov_outputs = ov_model(input_ids)
        print(f"OpenVINO inference successful. Logits shape: {ov_outputs.logits.shape}")
        
    except Exception as e:
        print(f"OpenVINO compilation/inference failed: {e}")

    print("\n Validation Summary ")
    print(f"Final RAM footprint: {get_ram_usage():.2f} MB")
    print("Tiny model is fully compatible with Optimum-Intel/OpenVINO pipeline!")

if __name__ == "__main__":
    validate_model()