# MiniCPM-o-2.6 Integration Testing Support

This branch (`tiny-minicpm-integration`) introduces support for lightweight integration testing of the **MiniCPM-o-2.6** architecture within `optimum-intel`.

## Deliverables Summary
1. **Tiny Model:** [h0witended/tiny-random-MiniCPM-o-2_6](https://huggingface.co/h0witended/tiny-random-MiniCPM-o-2_6)
   - Size: 0.77 MB
   - Purpose: Accelerating CI/CD and OpenVINO export validation.
2. **Integration Scripts:**
   - `generate_tiny_minicpm.py`: Script used to prune the original architecture and generate random weights.
   - `validate_tiny_minicpm.py`: A benchmarking utility to verify RAM usage and OpenVINO compilation time.

## Key Changes
- **Tests Updated:** Modified `tests/openvino/utils_tests.py` to point the `minicpmo` model ID to the custom tiny version.
- **Validation:** Confirmed that the model passes the `optimum-cli export` process and is compatible with the `OVModelForVisualCausalLM` class.

## Benchmark Results (Local)
- **OpenVINO Compilation:** Successful in **4.80 seconds**.
- **RAM Overhead:** Minimal (under 2MB during initial load).
- **Export Task:** `text-generation` (Verified).
