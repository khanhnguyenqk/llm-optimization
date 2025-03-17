#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert Mistral 7B models to ONNX format
"""

import os
import logging
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert a model to ONNX format")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to the model to convert"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default=None,
        help="Path to save the ONNX model"
    )
    parser.add_argument(
        "--optimize", 
        action="store_true",
        help="Whether to optimize the ONNX model"
    )
    parser.add_argument(
        "--quantize", 
        action="store_true",
        help="Whether to quantize the ONNX model"
    )
    parser.add_argument(
        "--opset", 
        type=int, 
        default=15,
        help="ONNX opset version"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for conversion"
    )
    return parser.parse_args()

def convert_to_onnx(args):
    """
    Convert a model to ONNX format.
    
    Args:
        args: Command line arguments
    """
    logger.info(f"Converting model from {args.model_path} to ONNX format")
    
    # Determine output directory
    if args.output_path is None:
        model_name = os.path.basename(args.model_path)
        args.output_path = os.path.join("models", f"{model_name}-ONNX")
    
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"ONNX model will be saved to: {output_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Export the model to ONNX with optimum
    logger.info("Exporting model to ONNX...")
    
    # Check if model is already in ONNX format
    if os.path.exists(os.path.join(args.model_path, "model.onnx")):
        logger.info("Model is already in ONNX format. Copying files...")
        onnx_model = ORTModelForCausalLM.from_pretrained(
            args.model_path,
            file_name="model.onnx",
            provider="CUDAExecutionProvider" if args.device == "cuda" else "CPUExecutionProvider"
        )
        onnx_model.save_pretrained(args.output_path)
    else:
        # Export to ONNX
        try:
            # First attempt to use optimum's export functionality
            onnx_model = ORTModelForCausalLM.from_pretrained(
                args.model_path,
                export=True,
                opset=args.opset,
                device=args.device,
            )
            
            # Save the exported model
            onnx_model.save_pretrained(args.output_path)
            tokenizer.save_pretrained(args.output_path)
            
        except Exception as e:
            logger.warning(f"Error using optimum for export: {e}")
            logger.info("Falling back to manual export...")
            
            # Fallback to manual export with torch.onnx.export
            model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float32)
            model.eval()
            model.to(args.device)
            
            # Create example inputs
            dummy_input = tokenizer("Hello, I am a language model", return_tensors="pt").to(args.device)
            input_names = ["input_ids", "attention_mask"]
            output_names = ["logits"]
            
            # Export the model
            dynamic_axes = {
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"},
            }
            
            torch.onnx.export(
                model,
                (dummy_input.input_ids, dummy_input.attention_mask),
                os.path.join(args.output_path, "model.onnx"),
                opset_version=args.opset,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )
            
            tokenizer.save_pretrained(args.output_path)
            
            # Save config
            with open(os.path.join(args.output_path, "config.json"), "w") as f:
                f.write(model.config.to_json_string())
    
    # Optimize the model
    if args.optimize:
        logger.info("Optimizing ONNX model...")
        try:
            import onnxruntime as ort
            from onnxruntime.transformers.optimizer import optimize_model
            
            optimized_model_path = os.path.join(args.output_path, "model_optimized.onnx")
            
            # Optimize for transformers
            model_type = "bert"  # Default, specific optimization depends on the model
            if "gpt" in args.model_path.lower() or "mistral" in args.model_path.lower():
                model_type = "gpt2"
            
            onnx_model_path = os.path.join(args.output_path, "model.onnx")
            optimized_model = optimize_model(
                onnx_model_path,
                model_type=model_type,
                num_heads=32,  # This should be based on model architecture
                hidden_size=4096,  # This should be based on model architecture
            )
            
            optimized_model.save_model_to_file(optimized_model_path)
            logger.info(f"Optimized model saved to {optimized_model_path}")
            
        except Exception as e:
            logger.error(f"Error optimizing ONNX model: {e}")
    
    # Quantize the model
    if args.quantize:
        logger.info("Quantizing ONNX model...")
        try:
            import onnxruntime as ort
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            onnx_model_path = os.path.join(args.output_path, "model_optimized.onnx" if args.optimize else "model.onnx")
            quantized_model_path = os.path.join(args.output_path, "model_quantized.onnx")
            
            quantize_dynamic(
                onnx_model_path,
                quantized_model_path,
                weight_type=QuantType.QInt8
            )
            logger.info(f"Quantized model saved to {quantized_model_path}")
            
        except Exception as e:
            logger.error(f"Error quantizing ONNX model: {e}")
    
    logger.info("ONNX conversion completed successfully")

def main():
    """Main function."""
    args = parse_args()
    convert_to_onnx(args)

if __name__ == "__main__":
    main() 