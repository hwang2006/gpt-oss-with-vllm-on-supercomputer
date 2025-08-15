# Running GPT-OSS with vLLM on Supercomputers (SLURM + Singularity)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![vLLM](https://img.shields.io/badge/vLLM-0.10.2+-green.svg)](https://github.com/vllm-project/vllm)
[![HPC](https://img.shields.io/badge/HPC-SLURM%20%2B%20Singularity-orange.svg)]()

This repository provides a complete setup for running [GPT-OSS](https://openai.com/index/gpt-oss/) and other large language models using [vLLM](https://github.com/vllm-project/vllm) on high-performance computing (HPC) clusters with SLURM and Singularity. It includes a user-friendly Gradio web interface and comprehensive API access.

**Key Features:**
- **HPC Optimized**: Designed for SLURM job schedulers and Singularity containers
- **Multi-GPU Support**: Tensor parallelism for large models
- **Web Interface**: Beautiful Gradio chat interface
- **API Compatible**: OpenAI-compatible REST API
- **Easy Setup**: One-command deployment with automated monitoring
- **Model Flexibility**: Support for various GPT-OSS model sizes

## Why vLLM for HPC?

| Feature | Benefits |
|---------|----------|
| **Multi-GPU Support** | Excellent tensor parallelism for large models |
| **Memory Efficiency** | Optimized GPU memory utilization |
| **API Compatibility** | Native OpenAI-compatible endpoints |
| **Model Support** | Direct HuggingFace Hub integration |
| **Production Ready** | Battle-tested for high-throughput inference |

## Table of Contents

- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)
- [Contributing](#contributing)

## Requirements

### Hardware
- **HPC cluster** with SLURM scheduler
- **NVIDIA GPUs** (H200, A100, etc.)
- **CUDA 12.8+**

### Software
- **Singularity** 3.5+
- **Python** 3.10+
- **SLURM** job scheduler
- **SSH access** for tunneling

## Install Conda on KISTI Neuron GPU Cluster

This repository has been specifically tested for the [KISTI Neuron GPU Cluster](https://www.ksc.re.kr/eng/resources/neuron).

- **Conda Setup**: [Installation Guide](https://github.com/hwang2006/gpt-oss-with-ollama-on-supercomputing?tab=readme-ov-file#installing-conda)

## Clone the Repository
to set up this repository on your scratch directory.
```
[glogin01]$ cd /scratch/$USER
[glogin01]$ git clone https://github.com/hwang2006/gpt-oss-with-vllm-on-supercomputer.git
[glogin01]$ cd gpt-oss-with-vllm-on-supercomputer
```

## Build vLLM Singularity Image
```bash
[glogin01]$ singularity build --fakeroot vllm-gptoss.sif docker://vllm/vllm-openai:gptoss
```
```bash
[glogin01]$ singularity exec ./vllm-gptoss.sif pip list | grep vllm
vllm                              0.10.1+gptoss
```

## Create a Conda Virtual Environment
1. Create a conda virtual environment with a python version 3.11+
```
[glogin01]$ conda create -n vllm-hpc python=3.11
Retrieving notices: ...working... done
Collecting package metadata (current_repodata.json): done
Solving environment: done

## Package Plan ## 

  environment location: /scratch/qualis/miniconda3/envs/vllm-hpc

  added / updated specs:
    - python=3.11
.
.
.
Proceed ([y]/n)? y    <========== type yes

Downloading and Extracting Packages:

Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use
#
#     $ conda activate vllm-hpc
#
# To deactivate an active environment, use
#
#     $ conda deactivate
```

2. load modules
```
[glogin01]$ module load gcc/10.2.0 cuda/12.1
```

3. Install Gradio for UI
```
[glogin01]$ conda activate vllm-hpc
(vllm-hpc) [glogin01]$ pip install gradio
Looking in indexes: https://pypi.org/simple, https://pypi.ngc.nvidia.com
Collecting gradio
  Downloading gradio-5.42.0-py3-none-any.whl.metadata (16 kB)
Collecting aiofiles<25.0,>=22.0 (from gradio)
  Downloading aiofiles-24.1.0-py3-none-any.whl.metadata (10 kB)
.
.
.
Successfully installed aiofiles-24.1.0 annotated-types-0.7.0 anyio-4.10.0 brotli-1.1.0 certifi-2025.8.3 charset_normalizer-3.4.3 click-8.2.1 fastapi-0.116.1 ffmpy-0.6.1 filelock-3.18.0 fsspec-2025.7.0 gradio-5.42.0 gradio-client-1.11.1 groovy-0.1.2 h11-0.16.0 hf-xet-1.1.7 httpcore-1.0.9 httpx-0.28.1 huggingface-hub-0.34.4 idna-3.10 jinja2-3.1.6 markdown-it-py-3.0.0 markupsafe-3.0.2 mdurl-0.1.2 numpy-2.3.2 orjson-3.11.1 packaging-25.0 pandas-2.3.1 pillow-11.3.0 pydantic-2.11.7 pydantic-core-2.33.2 pydub-0.25.1 pygments-2.19.2 python-dateutil-2.9.0.post0 python-multipart-0.0.20 pytz-2025.2 pyyaml-6.0.2 requests-2.32.4 rich-14.1.0 ruff-0.12.8 safehttpx-0.1.6 semantic-version-2.10.0 shellingham-1.5.4 six-1.17.0 sniffio-1.3.1 starlette-0.47.2 tomlkit-0.13.3 tqdm-4.67.1 typer-0.16.0 typing-extensions-4.14.1 typing-inspection-0.4.1
```

### 4. Submit Job
```bash
# Edit vllm_gradio_run_singularity.sh to set your model
sbatch vllm_gradio_run_singularity.sh
```

### 5. Access Interface
```bash
# Get port forwarding command
cat port_forwarding_*.txt

# From your local machine
ssh -L localhost:7860:gpu##:7860 -L localhost:8000:gpu##:8000 $USER@your-cluster.edu

# Open in browser
# Gradio UI: http://localhost:7860
# API Docs: http://localhost:8000/docs
```

## 📦 Installation

### Prerequisites Setup
```bash
# Load required modules (adjust for your cluster)
module load gcc/10.2.0 cuda/12.1 singularity

# Create conda environment
conda create -n vllm-hpc python=3.11
conda activate vllm-hpc
```

### Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Build Container Image
```bash
# Option 1: Use pre-built image
singularity pull vllm-gptoss.sif docker://vllm/vllm-openai:latest

# Option 2: Build custom image (if needed)
singularity build --fakeroot vllm-custom.sif vllm.def
```

## 🎮 Usage

### Basic Usage
```bash
# 1. Configure your model in the SLURM script
export MODEL_NAME="openai/gpt-oss-20b"
export TENSOR_PARALLEL_SIZE=1

# 2. Submit job
sbatch vllm_gradio_run_singularity.sh

# 3. Monitor job
squeue -u $USER
tail -f vllm_server_*.log
```

### Advanced Configuration
```bash
# Multi-GPU setup
export TENSOR_PARALLEL_SIZE=4
#SBATCH --gres=gpu:4

# Memory optimization
export GPU_MEMORY_UTILIZATION=0.85
export MAX_MODEL_LEN=8192

# High throughput setup
export MAX_NUM_BATCHED_TOKENS=8192
export MAX_NUM_SEQS=256
```

## 🔌 API Reference

### List Models
```bash
curl http://localhost:8000/v1/models
```

### Chat Completion
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-20b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Python Client
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Streaming Responses
```python
stream = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## ⚡ Performance Tuning

### GPU Memory Optimization
```bash
# Adjust memory utilization (default: 0.9)
export GPU_MEMORY_UTILIZATION=0.85

# Enable CPU offloading for large models
export CPU_OFFLOAD_GB=16
```

### Multi-GPU Scaling
```bash
# Tensor parallelism (splits model across GPUs)
export TENSOR_PARALLEL_SIZE=4
#SBATCH --gres=gpu:4

# Pipeline parallelism (for very large models)
export PIPELINE_PARALLEL_SIZE=2
```

### Throughput Optimization
```bash
# Increase batch processing
export MAX_NUM_BATCHED_TOKENS=8192
export MAX_NUM_SEQS=256

# Disable logging for max performance
export DISABLE_LOG_REQUESTS=true
```

### Context Length Tuning
```bash
# Adjust based on your use case
export MAX_MODEL_LEN=4096   # Faster, less memory
export MAX_MODEL_LEN=32768  # Longer context, more memory
```

## 🐛 Troubleshooting

### Common Issues

#### Model Loading Fails
```bash
# Check model availability
python download_models.py --list-local

# Download model explicitly
python download_models.py openai/gpt-oss-20b
```

#### GPU Memory Errors
```bash
# Reduce memory utilization
export GPU_MEMORY_UTILIZATION=0.7

# Use smaller model
export MODEL_NAME="openai/gpt-oss-7b"
```

#### Connection Issues
```bash
# Check if server is running
curl http://localhost:8000/health

# Verify SSH tunnel
ssh -L 8000:gpu##:8000 $USER@cluster.edu
```

#### Job Fails to Start
```bash
# Check SLURM logs
cat slurm-*.out

# Verify Singularity image
singularity exec vllm-gptoss.sif python --version
```

### Performance Issues
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Check vLLM logs for bottlenecks
tail -f vllm_server_*.log

# Profile memory usage
cat /proc/meminfo
```

## 📊 Example Usage

Your vLLM server will be accessible through multiple interfaces:

- **Gradio Web UI**: Interactive chat interface at http://localhost:7860
- **OpenAI API**: Compatible endpoints at http://localhost:8000/v1
- **Direct CLI**: Use curl or Python clients for automation

The setup handles model loading, GPU memory management, and provides monitoring through SLURM logs.

## 🏗️ Project Structure

```
gpt-oss-vllm-hpc/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment
├── vllm_gradio_run_singularity.sh     # Main SLURM script
├── vllm_web.py                        # Gradio web interface
├── download_models.py                 # Model management
├── examples/                          # Usage examples
│   ├── api_examples.py                # API usage examples
│   ├── batch_inference.py             # Batch processing
│   └── performance_test.py            # Benchmarking
├── configs/                           # Configuration files
│   ├── models.yaml                    # Model configurations
│   └── cluster_configs/               # Cluster-specific configs
├── scripts/                           # Utility scripts
│   ├── setup_environment.sh           # Environment setup
│   └── health_check.py                # System diagnostics
├── docs/                              # Documentation
│   ├── installation.md                # Detailed installation
│   ├── performance.md                 # Performance tuning
│   └── troubleshooting.md             # Troubleshooting guide
└── .github/                           # GitHub workflows
    └── workflows/
        └── ci.yml                     # Continuous integration
```

## 🤝 Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md).

### Development Setup
```bash
git clone https://github.com/YOUR_USERNAME/gpt-oss-vllm-hpc.git
cd gpt-oss-vllm-hpc
conda create -n vllm-dev python=3.11
conda activate vllm-dev
pip install -r requirements-dev.txt
```

### Running Tests
```bash
python -m pytest tests/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [vLLM Team](https://github.com/vllm-project/vllm) for the excellent inference engine
- [OpenAI](https://openai.com/) for releasing GPT-OSS
- [KISTI](https://www.ksc.re.kr/) for providing Neuron GPU cluster access
- HPC communities for testing and feedback

## 📚 Citations

If you use this work in your research, please cite:

```bibtex
@software{gpt_oss_vllm_hpc,
  title={GPT-OSS with vLLM on Supercomputers},
  author={Your Name},
  year={2025},
  url={https://github.com/YOUR_USERNAME/gpt-oss-vllm-hpc}
}
```

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/YOUR_USERNAME/gpt-oss-vllm-hpc/issues)
- **Discussions**: [GitHub Discussions](https://github.com/YOUR_USERNAME/gpt-oss-vllm-hpc/discussions)
- **Email**: your.email@institution.edu

---

**⭐ Star this repository if it helps your research!**
quali@hwang-laptop MINGW64 ~/gpt-oss-with-ollama-on-supercomputing (main)
$
