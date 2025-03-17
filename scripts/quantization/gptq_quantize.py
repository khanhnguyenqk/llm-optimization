#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GPTQ Quantization for Mistral 7B models
"""

import argparse
import logging
import os
from pathlib import Path

from transformers import AutoTokenizer
from datasets import load_dataset
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GPTQ Quantization for LLMs")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to the model to quantize"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default=None,
        help="Path to save the quantized model"
    )
    parser.add_argument(
        "--bits", 
        type=int, 
        default=4,
        choices=[2, 3, 4, 8],
        help="Number of bits for quantization"
    )
    parser.add_argument(
        "--group_size", 
        type=int, 
        default=128,
        help="Group size for quantization"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="wikitext",
        choices=["wikitext", "c4", "ptb"],
        help="Dataset to use for calibration"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=128,
        help="Number of samples to use for calibration"
    )
    parser.add_argument(
        "--use_cuda", 
        action="store_true",
        help="Use CUDA for quantization"
    )
    return parser.parse_args()

def prepare_calibration_data(tokenizer, dataset_name, num_samples=128):
    """
    Prepare calibration data from a dataset.
    
    Args:
        tokenizer: The tokenizer to use
        dataset_name: The name of the dataset to use
        num_samples: Number of samples to use
        
    Returns:
        list: List of tokenized examples
    """
    logger.info(f"Loading calibration dataset: {dataset_name}")
    
    if dataset_name == "wikitext":
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        examples = [text for text in data["text"] if len(text) > 100]
    elif dataset_name == "c4":
        data = load_dataset("c4", "en", split="validation", streaming=True)
        examples = []
        for sample in data:
            examples.append(sample["text"])
            if len(examples) >= num_samples * 2:  # Get more than we need, for filtering
                break
    elif dataset_name == "ptb":
        data = load_dataset("ptb_text_only", "penn_treebank", split="validation")
        examples = [" ".join(sample["sentence"]) for sample in data]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Filter examples
    examples = [ex for ex in examples if len(ex.split()) > 50][:num_samples]
    
    logger.info(f"Tokenizing {len(examples)} examples")
    tokenized_examples = []
    for text in examples:
        tokenized = tokenizer(text, return_tensors="pt")
        if tokenized["input_ids"].shape[1] > 0:
            tokenized_examples.append(tokenized)
    
    return tokenized_examples

def quantize_model(args):
    """
    Quantize a model using GPTQ.
    
    Args:
        args: Command line arguments
        
    Returns:
        The quantized model
    """
    # Load the model and tokenizer
    logger.info(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Prepare calibration data
    examples = prepare_calibration_data(tokenizer, args.dataset, args.num_samples)
    
    # Determine the output directory
    if args.output_path is None:
        model_name = os.path.basename(args.model_path)
        args.output_path = os.path.join("models", f"{model_name}-GPTQ-{args.bits}bit")
    
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Quantized model will be saved to: {output_path}")
    
    # Configure quantization
    quantize_config = BaseQuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        desc_act=False,  # Set to True for better accuracy, but slower
    )
    
    # Load and quantize the model
    device_map = "auto" if args.use_cuda else "cpu"
    model = AutoGPTQForCausalLM.from_pretrained(
        args.model_path,
        quantize_config=quantize_config,
        use_triton=False,  # Set to True for faster inference if Triton is available
    )
    
    # Prepare inputs for calibration
    calibration_examples = []
    for example in examples:
        if len(calibration_examples) >= args.num_samples:
            break
        input_ids = example["input_ids"].to(model.device)
        if input_ids.shape[1] <= 2048:  # Limit to context length
            calibration_examples.append(input_ids)
    
    # Run quantization
    logger.info(f"Starting GPTQ quantization with {args.bits} bits and group size {args.group_size}")
    model.quantize(calibration_examples)
    
    # Save the quantized model
    logger.info(f"Saving quantized model to {output_path}")
    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)
    
    logger.info("Quantization completed successfully")
    return model

def main():
    """Main function."""
    args = parse_args()
    quantize_model(args)

if __name__ == "__main__":
    main() 