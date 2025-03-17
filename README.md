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

- Python 3.10+
- CUDA compatible GPU (for training and optimization)
- Docker (for containerization)
- Kubernetes cluster (for deployment)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm-optimization.git
cd llm-optimization
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

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