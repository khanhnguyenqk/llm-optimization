#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Download and prepare Mistral 7B models from Hugging Face
"""

import argparse
import logging
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download and prepare a model from Hugging Face")
    parser.add_argument(
        "--model", 
        required=True,
        type=str, 
        help="Model ID on Hugging Face Hub"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="Output directory for model files"
    )
    parser.add_argument(
        "--force_download", 
        action="store_true", 
        help="Force redownload even if files exist locally"
    )
    parser.add_argument(
        "--test", 
        action="store_true", 
        help="Run a quick test inference after download"
    )
    parser.add_argument(
        "--fp16", 
        action="store_true", 
        help="Download model in fp16 precision"
    )
    parser.add_argument(
        "--token",
        type=str,
        help="Hugging Face API token (overrides HF_TOKEN from .env)"
    )
    return parser.parse_args()

def download_model(model_id, output_dir, force_download=False, fp16=False, token=None):
    """
    Download model and tokenizer from Hugging Face
    
    Args:
        model_id: The model ID on the Hugging Face Hub
        output_dir: Directory to save the model to
        force_download: Whether to force redownload
        fp16: Whether to download in fp16 precision
        token: Hugging Face API token
    
    Returns:
        tuple: (model, tokenizer)
    """
    logger.info(f"Downloading model: {model_id}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Model will be saved to: {output_dir}")
    
    # Get token from environment variable if not provided
    if token is None:
        token = os.getenv("HF_TOKEN")
        if not token:
            logger.warning("No Hugging Face token provided. If the model is gated, download will fail.")
    
    # Download model
    try:
        # First, download the files from HF Hub
        snapshot_download(
            repo_id=model_id,
            local_dir=output_dir,
            force_download=force_download,
            token=token,
        )
        
        # Then load the model and tokenizer
        torch_dtype = torch.float16 if fp16 else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            output_dir,
            torch_dtype=torch_dtype,
            device_map="auto",
            token=token,
        )
        tokenizer = AutoTokenizer.from_pretrained(output_dir, token=token)
        
        logger.info(f"Successfully downloaded and loaded model: {model_id}")
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        raise

def test_model(model, tokenizer):
    """Run a simple test inference with the model."""
    try:
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=100,
        )
        
        test_prompt = "What are the key advantages of transformer models?"
        logger.info(f"Testing model with prompt: '{test_prompt}'")
        
        result = generator(test_prompt)
        logger.info(f"Generated text: {result[0]['generated_text']}")
        logger.info("Model test completed successfully")
    
    except Exception as e:
        logger.error(f"Error testing model: {e}")
        raise

def main():
    """Main function."""
    args = parse_args()
    
    model, tokenizer = download_model(
        args.model, 
        args.output_dir, 
        args.force_download,
        args.fp16,
        args.token
    )
    
    if args.test:
        test_model(model, tokenizer)
    
    logger.info("Model preparation completed")

if __name__ == "__main__":
    main() 