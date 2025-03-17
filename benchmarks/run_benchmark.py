#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Benchmark script for LLM inference performance
"""

import os
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Any, Tuple

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional imports for different backends
try:
    import onnxruntime as ort
    from optimum.onnxruntime import ORTModelForCausalLM
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import tritonclient.http as triton_http
    import tritonclient.grpc as triton_grpc
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark LLM inference performance")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to the model to benchmark"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="benchmarks/results",
        help="Directory to save benchmark results"
    )
    parser.add_argument(
        "--backend", 
        type=str, 
        default="pytorch",
        choices=["pytorch", "onnx", "triton"],
        help="Backend to use for inference"
    )
    parser.add_argument(
        "--batch_sizes", 
        type=str, 
        default="1",
        help="Comma-separated list of batch sizes to benchmark"
    )
    parser.add_argument(
        "--sequence_lengths", 
        type=str, 
        default="128",
        help="Comma-separated list of sequence lengths to benchmark"
    )
    parser.add_argument(
        "--num_iterations", 
        type=int, 
        default=10,
        help="Number of iterations for each benchmark"
    )
    parser.add_argument(
        "--warmup_iterations", 
        type=int, 
        default=3,
        help="Number of warmup iterations before benchmarking"
    )
    parser.add_argument(
        "--triton_url", 
        type=str, 
        default="localhost:8000",
        help="URL for Triton Inference Server"
    )
    parser.add_argument(
        "--triton_model_name", 
        type=str, 
        default=None,
        help="Model name in Triton Inference Server"
    )
    parser.add_argument(
        "--triton_protocol", 
        type=str, 
        default="http",
        choices=["http", "grpc"],
        help="Protocol to use for Triton Inference Server"
    )
    parser.add_argument(
        "--use_cuda", 
        action="store_true",
        help="Use CUDA for inference"
    )
    return parser.parse_args()

def load_pytorch_model(model_path: str, use_cuda: bool) -> Tuple[Any, Any]:
    """
    Load a PyTorch model for benchmarking.
    
    Args:
        model_path: Path to the model
        use_cuda: Whether to use CUDA
        
    Returns:
        tuple: (model, tokenizer)
    """
    logger.info(f"Loading PyTorch model from {model_path}")
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    
    if use_cuda and torch.cuda.is_available():
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Using CPU for inference")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    
    if device == "cpu":
        model = model.to(device)
    
    return model, tokenizer

def load_onnx_model(model_path: str, use_cuda: bool) -> Tuple[Any, Any]:
    """
    Load an ONNX model for benchmarking.
    
    Args:
        model_path: Path to the model
        use_cuda: Whether to use CUDA
        
    Returns:
        tuple: (model, tokenizer)
    """
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX Runtime is not installed. Please install it with: pip install onnxruntime-gpu")
    
    logger.info(f"Loading ONNX model from {model_path}")
    
    # Check if model is in ONNX format
    onnx_path = None
    if os.path.isfile(model_path) and model_path.endswith(".onnx"):
        onnx_path = model_path
    elif os.path.exists(os.path.join(model_path, "model.onnx")):
        onnx_path = os.path.join(model_path, "model.onnx")
    elif os.path.exists(os.path.join(model_path, "model_quantized.onnx")):
        onnx_path = os.path.join(model_path, "model_quantized.onnx")
    elif os.path.exists(os.path.join(model_path, "model_optimized.onnx")):
        onnx_path = os.path.join(model_path, "model_optimized.onnx")
    
    if onnx_path is None:
        raise ValueError(f"Could not find ONNX model in {model_path}")
    
    provider = "CUDAExecutionProvider" if use_cuda else "CPUExecutionProvider"
    
    # Load tokenizer
    if os.path.exists(os.path.join(os.path.dirname(onnx_path), "tokenizer.json")):
        tokenizer_path = os.path.dirname(onnx_path)
    else:
        tokenizer_path = model_path
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Try to use optimum-onnxruntime if model has right format
    try:
        model = ORTModelForCausalLM.from_pretrained(
            os.path.dirname(onnx_path),
            provider=provider
        )
    except Exception as e:
        logger.warning(f"Could not load model with optimum-onnxruntime: {e}")
        logger.info("Falling back to manual ONNX session creation")
        
        # Create ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            onnx_path, 
            sess_options=sess_options,
            providers=[provider]
        )
        
        model = session
    
    return model, tokenizer

def create_triton_client(args):
    """
    Create a client for Triton Inference Server.
    
    Args:
        args: Command line arguments
        
    Returns:
        The Triton client
    """
    if not TRITON_AVAILABLE:
        raise ImportError("Triton client is not installed. Please install it with: pip install tritonclient[all]")
    
    logger.info(f"Creating Triton client for {args.triton_url}")
    
    if args.triton_protocol == "http":
        client = triton_http.InferenceServerClient(args.triton_url)
    else:
        client = triton_grpc.InferenceServerClient(args.triton_url)
    
    return client

def benchmark_pytorch(model, tokenizer, prompt, batch_size, num_iterations, warmup_iterations):
    """
    Benchmark a PyTorch model.
    
    Args:
        model: The PyTorch model
        tokenizer: The tokenizer
        prompt: The input prompt
        batch_size: Batch size for inference
        num_iterations: Number of iterations
        warmup_iterations: Number of warmup iterations
        
    Returns:
        dict: Benchmark results
    """
    # Create input batch
    inputs = tokenizer([prompt] * batch_size, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Warm up
    logger.info(f"Warming up with {warmup_iterations} iterations")
    for _ in range(warmup_iterations):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False
            )
    
    # Benchmark
    logger.info(f"Running benchmark with {num_iterations} iterations")
    latencies = []
    start_time = time.time()
    
    for i in range(num_iterations):
        iter_start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False
            )
        iter_end = time.time()
        latencies.append((iter_end - iter_start) * 1000)  # ms
    
    end_time = time.time()
    
    # Calculate metrics
    total_time = end_time - start_time
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    throughput = (batch_size * num_iterations) / total_time
    
    # Get memory usage
    memory_used = 0
    if model.device.type == "cuda":
        memory_used = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
    
    # Clear cache
    if model.device.type == "cuda":
        torch.cuda.empty_cache()
    
    return {
        "backend": "pytorch",
        "batch_size": batch_size,
        "prompt_length": len(tokenizer.encode(prompt)),
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency,
        "p99_latency_ms": p99_latency,
        "throughput_samples_per_sec": throughput,
        "memory_mb": memory_used
    }

def benchmark_onnx(model, tokenizer, prompt, batch_size, num_iterations, warmup_iterations):
    """
    Benchmark an ONNX model.
    
    Args:
        model: The ONNX model
        tokenizer: The tokenizer
        prompt: The input prompt
        batch_size: Batch size for inference
        num_iterations: Number of iterations
        warmup_iterations: Number of warmup iterations
        
    Returns:
        dict: Benchmark results
    """
    # Handle different model types
    is_optimum_model = hasattr(model, "generate")
    
    if is_optimum_model:
        # For optimum-onnxruntime models, use the generate method
        inputs = tokenizer([prompt] * batch_size, return_tensors="pt", padding=True)
        
        # Warm up
        logger.info(f"Warming up with {warmup_iterations} iterations")
        for _ in range(warmup_iterations):
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False
            )
        
        # Benchmark
        logger.info(f"Running benchmark with {num_iterations} iterations")
        latencies = []
        start_time = time.time()
        
        for i in range(num_iterations):
            iter_start = time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False
            )
            iter_end = time.time()
            latencies.append((iter_end - iter_start) * 1000)  # ms
    
    else:
        # For raw ONNX session, prepare inputs and run inference directly
        inputs = tokenizer([prompt] * batch_size, return_tensors="pt", padding=True)
        input_dict = {
            "input_ids": inputs.input_ids.numpy(),
            "attention_mask": inputs.attention_mask.numpy()
        }
        
        # Get input and output names
        input_names = [input.name for input in model.get_inputs()]
        output_names = [output.name for output in model.get_outputs()]
        
        # Warm up
        logger.info(f"Warming up with {warmup_iterations} iterations")
        for _ in range(warmup_iterations):
            outputs = model.run(output_names, {name: input_dict[name] for name in input_names})
        
        # Benchmark
        logger.info(f"Running benchmark with {num_iterations} iterations")
        latencies = []
        start_time = time.time()
        
        for i in range(num_iterations):
            iter_start = time.time()
            outputs = model.run(output_names, {name: input_dict[name] for name in input_names})
            iter_end = time.time()
            latencies.append((iter_end - iter_start) * 1000)  # ms
    
    end_time = time.time()
    
    # Calculate metrics
    total_time = end_time - start_time
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    throughput = (batch_size * num_iterations) / total_time
    
    return {
        "backend": "onnx",
        "batch_size": batch_size,
        "prompt_length": len(tokenizer.encode(prompt)),
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency,
        "p99_latency_ms": p99_latency,
        "throughput_samples_per_sec": throughput,
        "memory_mb": 0  # Not available for ONNX Runtime
    }

def benchmark_triton(client, model_name, tokenizer, prompt, batch_size, num_iterations, warmup_iterations, protocol="http"):
    """
    Benchmark a model served by Triton Inference Server.
    
    Args:
        client: The Triton client
        model_name: The name of the model in Triton
        tokenizer: The tokenizer
        prompt: The input prompt
        batch_size: Batch size for inference
        num_iterations: Number of iterations
        warmup_iterations: Number of warmup iterations
        protocol: The protocol to use (http or grpc)
        
    Returns:
        dict: Benchmark results
    """
    # Encode input
    input_ids = tokenizer.encode(prompt, return_tensors="pt").repeat(batch_size, 1).numpy()
    attention_mask = np.ones_like(input_ids)
    
    # Prepare inputs for Triton
    if protocol == "http":
        inputs = [
            triton_http.InferInput("input_ids", input_ids.shape, "INT32"),
            triton_http.InferInput("attention_mask", attention_mask.shape, "INT32")
        ]
        inputs[0].set_data_from_numpy(input_ids)
        inputs[1].set_data_from_numpy(attention_mask)
        
        outputs = [
            triton_http.InferRequestedOutput("logits")
        ]
    else:
        inputs = [
            triton_grpc.InferInput("input_ids", input_ids.shape, "INT32"),
            triton_grpc.InferInput("attention_mask", attention_mask.shape, "INT32")
        ]
        inputs[0].set_data_from_numpy(input_ids)
        inputs[1].set_data_from_numpy(attention_mask)
        
        outputs = [
            triton_grpc.InferRequestedOutput("logits")
        ]
    
    # Warm up
    logger.info(f"Warming up with {warmup_iterations} iterations")
    for _ in range(warmup_iterations):
        response = client.infer(model_name, inputs, outputs=outputs)
    
    # Benchmark
    logger.info(f"Running benchmark with {num_iterations} iterations")
    latencies = []
    start_time = time.time()
    
    for i in range(num_iterations):
        iter_start = time.time()
        response = client.infer(model_name, inputs, outputs=outputs)
        iter_end = time.time()
        latencies.append((iter_end - iter_start) * 1000)  # ms
    
    end_time = time.time()
    
    # Calculate metrics
    total_time = end_time - start_time
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    throughput = (batch_size * num_iterations) / total_time
    
    return {
        "backend": "triton",
        "batch_size": batch_size,
        "prompt_length": len(tokenizer.encode(prompt)),
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency,
        "p99_latency_ms": p99_latency,
        "throughput_samples_per_sec": throughput,
        "memory_mb": 0  # Not available for Triton
    }

def run_benchmark(args):
    """
    Run benchmarks for the specified model and configuration.
    
    Args:
        args: Command line arguments
        
    Returns:
        list: Benchmark results
    """
    # Parse batch sizes and sequence lengths
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    sequence_lengths = [int(x) for x in args.sequence_lengths.split(",")]
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and tokenizer based on the backend
    if args.backend == "pytorch":
        model, tokenizer = load_pytorch_model(args.model_path, args.use_cuda)
    elif args.backend == "onnx":
        model, tokenizer = load_onnx_model(args.model_path, args.use_cuda)
    elif args.backend == "triton":
        client = create_triton_client(args)
        model_name = args.triton_model_name or os.path.basename(args.model_path)
        # Load tokenizer from model path
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Generate prompts of different lengths
    prompts = {
        128: "Explain the concept of transformer models in machine learning and their applications in natural language processing.",
        512: "Explain the concept of transformer models in machine learning and their applications in natural language processing. Transformers have become the dominant architecture for many NLP tasks, replacing recurrent neural networks like LSTMs. Discuss their advantages, limitations, and some recent advancements in transformer architecture. Also, explain how self-attention mechanism works and why it's important for handling sequential data. Provide examples of popular transformer models and their specific innovations.",
        1024: "Explain the concept of transformer models in machine learning and their applications in natural language processing. Transformers have become the dominant architecture for many NLP tasks, replacing recurrent neural networks like LSTMs. Discuss their advantages, limitations, and some recent advancements in transformer architecture. Also, explain how self-attention mechanism works and why it's important for handling sequential data. Provide examples of popular transformer models and their specific innovations.\n\nTransformers were introduced in the paper 'Attention Is All You Need' and have since revolutionized natural language processing. The key innovation was replacing recurrence and convolutions with self-attention mechanisms that allow the model to weigh the importance of different words in a sequence when making predictions. This approach allows for much more parallelization during training compared to recurrent models, resulting in faster training times on modern hardware like GPUs and TPUs.\n\nThe basic transformer architecture consists of an encoder and a decoder. The encoder processes the input sequence, while the decoder generates the output sequence. Each encoder layer has a multi-head self-attention mechanism and a feed-forward neural network, with layer normalization and residual connections. The decoder has a similar structure but with an additional attention mechanism that attends to the encoder's output.\n\nOne of the most significant advantages of transformers is their ability to capture long-range dependencies in sequential data, which was challenging for previous architectures. However, the standard transformer model has a quadratic computational complexity with respect to sequence length, which becomes problematic for very long sequences."
    }
    
    # Use default prompt for sequence lengths not in the dictionary
    for seq_len in sequence_lengths:
        if seq_len not in prompts:
            # Find the closest available prompt length
            available_lengths = list(prompts.keys())
            closest_length = min(available_lengths, key=lambda x: abs(x - seq_len))
            prompts[seq_len] = prompts[closest_length]
    
    # Run benchmarks
    results = []
    
    for batch_size in batch_sizes:
        for seq_len in sequence_lengths:
            prompt = prompts[seq_len]
            
            logger.info(f"Benchmarking with batch size {batch_size} and sequence length {seq_len}")
            
            if args.backend == "pytorch":
                benchmark_result = benchmark_pytorch(
                    model, tokenizer, prompt, batch_size, 
                    args.num_iterations, args.warmup_iterations
                )
            elif args.backend == "onnx":
                benchmark_result = benchmark_onnx(
                    model, tokenizer, prompt, batch_size, 
                    args.num_iterations, args.warmup_iterations
                )
            elif args.backend == "triton":
                benchmark_result = benchmark_triton(
                    client, model_name, tokenizer, prompt, batch_size,
                    args.num_iterations, args.warmup_iterations, args.triton_protocol
                )
            
            # Add details about the model and system
            benchmark_result["model"] = os.path.basename(args.model_path)
            benchmark_result["sequence_length"] = seq_len
            benchmark_result["device"] = "cuda" if args.use_cuda else "cpu"
            
            results.append(benchmark_result)
            print_result(benchmark_result)
    
    # Save results
    save_results(results, args)
    
    return results

def print_result(result):
    """Print a benchmark result in a formatted way."""
    print("\n" + "="*80)
    print(f"Model: {result['model']}")
    print(f"Backend: {result['backend']}")
    print(f"Device: {result['device']}")
    print(f"Batch Size: {result['batch_size']}")
    print(f"Sequence Length: {result['sequence_length']}")
    print(f"Prompt Length: {result['prompt_length']}")
    print(f"Average Latency: {result['avg_latency_ms']:.2f} ms")
    print(f"P95 Latency: {result['p95_latency_ms']:.2f} ms")
    print(f"P99 Latency: {result['p99_latency_ms']:.2f} ms")
    print(f"Throughput: {result['throughput_samples_per_sec']:.2f} samples/second")
    if result['memory_mb'] > 0:
        print(f"Memory Usage: {result['memory_mb']:.2f} MB")
    print("="*80 + "\n")

def save_results(results, args):
    """
    Save benchmark results to files and generate visualizations.
    
    Args:
        results: List of benchmark results
        args: Command line arguments
    """
    output_dir = Path(args.output_dir)
    
    # Save results as JSON
    model_name = os.path.basename(args.model_path)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"benchmark_{model_name}_{args.backend}_{timestamp}.json"
    
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved benchmark results to {json_path}")
    
    # Convert to DataFrame for plotting
    df = pd.DataFrame(results)
    
    # Generate plots
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    
    # Latency vs Batch Size for different sequence lengths
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=df, 
        x="batch_size", 
        y="avg_latency_ms", 
        hue="sequence_length",
        marker="o"
    )
    plt.title(f"{model_name} - {args.backend.upper()} - Latency vs Batch Size")
    plt.xlabel("Batch Size")
    plt.ylabel("Average Latency (ms)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / f"latency_vs_batch_{model_name}_{args.backend}_{timestamp}.png")
    
    # Throughput vs Batch Size for different sequence lengths
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=df, 
        x="batch_size", 
        y="throughput_samples_per_sec", 
        hue="sequence_length",
        marker="o"
    )
    plt.title(f"{model_name} - {args.backend.upper()} - Throughput vs Batch Size")
    plt.xlabel("Batch Size")
    plt.ylabel("Throughput (samples/second)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / f"throughput_vs_batch_{model_name}_{args.backend}_{timestamp}.png")
    
    # Latency vs Sequence Length for different batch sizes
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=df, 
        x="sequence_length", 
        y="avg_latency_ms", 
        hue="batch_size",
        marker="o"
    )
    plt.title(f"{model_name} - {args.backend.upper()} - Latency vs Sequence Length")
    plt.xlabel("Sequence Length")
    plt.ylabel("Average Latency (ms)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / f"latency_vs_seq_{model_name}_{args.backend}_{timestamp}.png")
    
    logger.info(f"Generated plots in {plot_dir}")

def main():
    """Main function."""
    args = parse_args()
    run_benchmark(args)

if __name__ == "__main__":
    main() 