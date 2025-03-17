#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prepare models for deployment with Triton Inference Server
"""

import os
import shutil
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare a model for Triton Inference Server")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to the model to prepare (ONNX model preferred)"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="deployment/triton/model_repository",
        help="Path to the Triton model repository"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default=None,
        help="Name for the model in the repository"
    )
    parser.add_argument(
        "--model_version", 
        type=int, 
        default=1,
        help="Version number for the model"
    )
    parser.add_argument(
        "--max_batch_size", 
        type=int, 
        default=8,
        help="Maximum batch size for the model"
    )
    parser.add_argument(
        "--max_sequence_length", 
        type=int, 
        default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--instance_count", 
        type=int, 
        default=1,
        help="Number of model instances to create"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="onnxruntime",
        choices=["onnxruntime", "pytorch", "tensorrt"],
        help="Backend to use for the model"
    )
    parser.add_argument(
        "--execution_accelerator",
        type=str,
        default=None,
        choices=[None, "tensorrt", "cuda", "cpu"],
        help="Execution accelerator to use"
    )
    return parser.parse_args()

def create_model_config(args, model_repository_path, model_format):
    """
    Create model configuration for Triton Inference Server.
    
    Args:
        args: Command line arguments
        model_repository_path: Path to the model repository
        model_format: Format of the model (onnx, pytorch, etc.)
    """
    # Create the model configuration
    config = {
        "name": args.model_name,
        "backend": args.backend,
        "max_batch_size": args.max_batch_size,
        "input": [
            {
                "name": "input_ids",
                "data_type": "TYPE_INT32",
                "dims": [-1]  # Variable sequence length
            },
            {
                "name": "attention_mask",
                "data_type": "TYPE_INT32",
                "dims": [-1]  # Variable sequence length
            }
        ],
        "output": [
            {
                "name": "logits",
                "data_type": "TYPE_FP32",
                "dims": [-1, -1]  # [sequence_length, vocab_size]
            }
        ],
        "instance_group": [
            {
                "count": args.instance_count,
                "kind": "KIND_GPU"
            }
        ],
        "dynamic_batching": {
            "preferred_batch_size": [1, 2, 4, 8],
            "max_queue_delay_microseconds": 50000
        },
        "parameters": {
            "max_sequence_length": {"string_value": str(args.max_sequence_length)}
        }
    }
    
    # Add execution accelerator if specified
    if args.execution_accelerator:
        config["optimization"] = {
            "execution_accelerators": {
                "gpu_execution_accelerator": [
                    {"name": args.execution_accelerator}
                ]
            }
        }
    
    # Write the configuration to a file
    config_path = os.path.join(model_repository_path, "config.pbtxt")
    with open(config_path, "w") as f:
        for key, value in config.items():
            if key == "name":
                f.write(f'name: "{value}"\n')
            elif key == "backend":
                f.write(f'backend: "{value}"\n')
            elif key == "max_batch_size":
                f.write(f'max_batch_size: {value}\n')
            elif key == "input":
                for input_tensor in value:
                    f.write('input {\n')
                    f.write(f'  name: "{input_tensor["name"]}"\n')
                    f.write(f'  data_type: {input_tensor["data_type"]}\n')
                    f.write(f'  dims: {input_tensor["dims"]}\n')
                    f.write('}\n')
            elif key == "output":
                for output_tensor in value:
                    f.write('output {\n')
                    f.write(f'  name: "{output_tensor["name"]}"\n')
                    f.write(f'  data_type: {output_tensor["data_type"]}\n')
                    f.write(f'  dims: {output_tensor["dims"]}\n')
                    f.write('}\n')
            elif key == "instance_group":
                for group in value:
                    f.write('instance_group {\n')
                    f.write(f'  count: {group["count"]}\n')
                    f.write(f'  kind: {group["kind"]}\n')
                    f.write('}\n')
            elif key == "dynamic_batching":
                f.write('dynamic_batching {\n')
                if "preferred_batch_size" in value:
                    for size in value["preferred_batch_size"]:
                        f.write(f'  preferred_batch_size: {size}\n')
                if "max_queue_delay_microseconds" in value:
                    f.write(f'  max_queue_delay_microseconds: {value["max_queue_delay_microseconds"]}\n')
                f.write('}\n')
            elif key == "parameters":
                for param_name, param_value in value.items():
                    f.write('parameters {\n')
                    f.write(f'  key: "{param_name}"\n')
                    if "string_value" in param_value:
                        f.write(f'  value: {{ string_value: "{param_value["string_value"]}" }}\n')
                    f.write('}\n')
            elif key == "optimization":
                f.write('optimization {\n')
                if "execution_accelerators" in value:
                    f.write('  execution_accelerators {\n')
                    if "gpu_execution_accelerator" in value["execution_accelerators"]:
                        for accel in value["execution_accelerators"]["gpu_execution_accelerator"]:
                            f.write('    gpu_execution_accelerator {\n')
                            f.write(f'      name: "{accel["name"]}"\n')
                            f.write('    }\n')
                    f.write('  }\n')
                f.write('}\n')
    
    logger.info(f"Created model configuration at {config_path}")

def prepare_model(args):
    """
    Prepare a model for deployment with Triton Inference Server.
    
    Args:
        args: Command line arguments
    """
    # Determine model name if not provided
    if args.model_name is None:
        args.model_name = os.path.basename(args.model_path)
    
    logger.info(f"Preparing model {args.model_name} for Triton Inference Server")
    
    # Create the model repository directory
    model_repository_path = os.path.join(args.output_path, args.model_name)
    version_path = os.path.join(model_repository_path, str(args.model_version))
    os.makedirs(version_path, exist_ok=True)
    
    logger.info(f"Model will be prepared at: {model_repository_path}")
    
    # Determine the model format
    model_format = "unknown"
    source_model_path = None
    
    # Check if model is in ONNX format
    if os.path.exists(os.path.join(args.model_path, "model.onnx")):
        model_format = "onnx"
        source_model_path = os.path.join(args.model_path, "model.onnx")
    elif os.path.exists(os.path.join(args.model_path, "model_quantized.onnx")):
        model_format = "onnx"
        source_model_path = os.path.join(args.model_path, "model_quantized.onnx")
    elif os.path.exists(os.path.join(args.model_path, "model_optimized.onnx")):
        model_format = "onnx"
        source_model_path = os.path.join(args.model_path, "model_optimized.onnx")
    elif os.path.exists(os.path.join(args.model_path, "pytorch_model.bin")):
        model_format = "pytorch"
        source_model_path = args.model_path  # For PyTorch, we copy the entire directory
    
    # Copy the model files to the version directory
    if model_format == "onnx":
        logger.info(f"Copying ONNX model: {source_model_path}")
        dest_model_path = os.path.join(version_path, "model.onnx")
        shutil.copy2(source_model_path, dest_model_path)
        
        # Copy tokenizer and config files
        if os.path.exists(os.path.join(args.model_path, "tokenizer.json")):
            shutil.copy2(os.path.join(args.model_path, "tokenizer.json"), os.path.join(model_repository_path, "tokenizer.json"))
        if os.path.exists(os.path.join(args.model_path, "config.json")):
            shutil.copy2(os.path.join(args.model_path, "config.json"), os.path.join(model_repository_path, "config.json"))
    
    elif model_format == "pytorch":
        logger.info(f"Copying PyTorch model directory: {source_model_path}")
        # For PyTorch models, copy the entire directory structure
        for item in os.listdir(source_model_path):
            s = os.path.join(source_model_path, item)
            d = os.path.join(version_path, item)
            if os.path.isdir(s):
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
    
    else:
        logger.warning(f"Could not determine model format. Copying entire directory as is.")
        # Copy all files in the model directory
        for item in os.listdir(args.model_path):
            s = os.path.join(args.model_path, item)
            d = os.path.join(version_path, item)
            if os.path.isdir(s):
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
    
    # Create the model configuration
    create_model_config(args, model_repository_path, model_format)
    
    logger.info(f"Model preparation completed successfully")
    logger.info(f"Model repository path: {args.output_path}")
    logger.info(f"To start Triton with this model: docker run --gpus=all -p 8000:8000 -p 8001:8001 -p 8002:8002 -v {os.path.abspath(args.output_path)}:/models nvcr.io/nvidia/tritonserver:24.04-py3 tritonserver --model-repository=/models")

def main():
    """Main function."""
    args = parse_args()
    prepare_model(args)

if __name__ == "__main__":
    main() 