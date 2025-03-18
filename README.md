# LLM Optimization Pipeline

A comprehensive pipeline for optimizing, quantizing, and deploying Mistral 7B models. This project includes tools for model optimization, ONNX conversion, Triton deployment, and benchmarking.

## Project Structure

```
llm-optimization/
├── notebooks/             # Jupyter notebooks for experimentation and visualization
├── models/                # Directory to store model weights and configurations
├── scripts/               # Python scripts for automation
│   ├── quantization/      # Scripts for model quantization (GPTQ, AWQ, etc.)
│   ├── onnx/              # Scripts for ONNX conversion
│   └── deployment/        # Utilities for deployment
├── deployment/            # Deployment configurations
│   ├── docker/            # Dockerfiles and docker-compose files
│   ├── triton/            # Triton Inference Server configurations
│   └── k8s/               # Kubernetes deployment manifests
└── benchmarks/            # Benchmarking scripts and results
```

## Getting Started

### Prerequisites

- Python 3.12.6+
- Poetry (for dependency management)
- CUDA compatible GPU (for training and optimization)
- Docker (for containerization)
- Kubernetes cluster (for deployment)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm-optimization.git
cd llm-optimization
```

2. Install dependencies with Poetry:
```bash
# Install Poetry if you don't have it
# curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install
```

3. Activate the Poetry virtual environment:
```bash
poetry shell
```

> **Note**: All commands in the following sections can be run either:
> - Within an activated Poetry shell (after running `poetry shell`), or
> - By prefixing commands with `poetry run` if you're not in the Poetry shell

## Workflow

### 1. Model Download and Preparation

```bash
python scripts/download_model.py --model mistralai/Mistral-7B-v0.1
```

### 2. Quantization

```bash
# For GPTQ quantization
python scripts/quantization/gptq_quantize.py --model_path models/Mistral-7B-v0.1 --bits 4 --group_size 128

# For AWQ quantization
python scripts/quantization/awq_quantize.py --model_path models/Mistral-7B-v0.1 --bits 4
```

### 3. ONNX Conversion

```bash
python scripts/onnx/convert_to_onnx.py --model_path models/Mistral-7B-v0.1-GPTQ --output_path models/Mistral-7B-v0.1-GPTQ-ONNX
```

### 4. Triton Deployment

```bash
python scripts/deployment/prepare_triton_model.py --model_path models/Mistral-7B-v0.1-GPTQ-ONNX --output_path deployment/triton/model_repository
```

### 5. Benchmarking

```bash
python benchmarks/run_benchmark.py --model_path models/Mistral-7B-v0.1-GPTQ --batch_sizes 1,2,4,8 --sequence_lengths 128,512,1024
```

For running benchmarks with specific precision:

```bash
# Benchmark with FP16 precision
python benchmarks/run_benchmark.py --model_path models/Mistral-7B-v0.1 --batch_sizes 1,2,4,8 --precision float16 

# Benchmark with BF16 precision (useful for some newer GPUs)
python benchmarks/run_benchmark.py --model_path models/Mistral-7B-v0.1 --batch_sizes 1,2,4,8 --precision bfloat16
```

## Deployment

### Docker

```bash
cd deployment/docker
docker-compose up -d
```

### Kubernetes

```bash
kubectl apply -f deployment/k8s/triton-deployment.yaml
```

## Benchmarks

Results of various optimizations and their impacts on:
- Inference latency
- Throughput
- Memory consumption
- Accuracy metrics

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Mistral AI](https://mistral.ai/) for the base model
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) for model implementation
- [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server) for deployment

# Model Optimization

This repository contains scripts for downloading and optimizing large language models.

## Setup

1. Install the required dependencies:
```bash
# Install Poetry if you don't have it
# curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install
```

2. Set up your Hugging Face token:
   - Copy `.env.example` to `.env`
   - Replace `your_token_here` with your actual Hugging Face token from https://huggingface.co/settings/tokens

## Downloading Models

You can download models using the `download_model.py` script:

```bash
# If already in a Poetry shell:
python ./scripts/download_model.py --model "MODEL_ID" --output_dir "OUTPUT_PATH"

# Or alternatively:
poetry run python ./scripts/download_model.py --model "MODEL_ID" --output_dir "OUTPUT_PATH"
```

### Options

- `--model`: The model ID on the Hugging Face Hub (required)
- `--output_dir`: Directory to save the model to (required)
- `--force_download`: Force redownload even if files exist locally
- `--test`: Run a quick test inference after download
- `--precision`: Precision to load the model in (choices: "float32", "float16", "bfloat16", default: "float32")
- `--token`: Directly provide a Hugging Face token (overrides the one in .env)

### Example

```bash
python ./scripts/download_model.py --model "mistralai/Mistral-7B-v0.1" --output_dir "./models/Mistral-7B-v0_1"

# Download in half precision
python ./scripts/download_model.py --model "mistralai/Mistral-7B-v0.1" --output_dir "./models/Mistral-7B-v0_1" --precision "float16"
```

## Accessing Gated Models

For gated models (like some versions of Mistral), you need to:

1. Make sure you have been granted access to the model on the Hugging Face Hub
2. Provide your Hugging Face token either:
   - In the `.env` file as `HF_TOKEN=your_token_here`
   - Or directly via the command line: `--token "your_token_here"`