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

### Submit the Slurm Script
- to launch both Ollama and Gradio server
```
(ollama-hpc) [glogin01]$ sbatch vllm_gradio_run_singularity.sh --model openai/gpt-oss-20b
Submitted batch job XXXXXX
```
- to check if the servers are up and running
```
(ollama-hpc) [glogin01]$ squeue -u $USER
             JOBID       PARTITION     NAME     USER    STATE       TIME TIME_LIMI  NODES NODELIST(REASON)
            XXXXXX    eme_h200nv_8    vllm_g    $USER  RUNNING       0:02   2-00:00:00      1 gpu##
```
- to check the SSH tunneling information generated by the vllm_gradio_run_singularity.sh script 
```
(ollama-hpc) [glogin01]$ cat port_forwarding_command_xxxxx.txt (xxxxx = SLURM jobid)
ssh -L localhost:7860:gpu50:7860 -L localhost:8000:gpu50:8000 qualis@neuron.ksc.re.kr
```

## Connect to the Gradio UI
- Once the job starts, open a new SSH client (e.g., Putty, MobaXterm, PowerShell, Command Prompt, etc) on your local machine and run the port forwarding command displayed in port_forwarding_command_xxxxx.txt:

<img width="863" height="380" alt="Image" src="https://github.com/user-attachments/assets/0f30fa76-1022-4853-858a-cfba52116184" />

- Then, open http://localhost:7860 in your browser to access the Gradio UI and pull a gpt-oss model (for example, 'gpt-oss:latest') to the ollama server models directory (e.g., OLLAMA_MODELS="/scratch/$USER/.ollama" in the slurm script) from the [Ollama models site](https://ollama.com/search) 

<img width="1134" height="707" alt="Image" src="https://github.com/user-attachments/assets/d26f62ce-99d5-479e-a7d4-79b1bb2eb009" />


- Once the gpt-oss model is successfully downloaded, it will be listed in the 'Select Model' dropdown menu on the top right of the Gradio UI. You can start chatting with the gpt-oss model. You could also pull and chat with other models (e.g., llama3, mistral, etc) by pulling them from the Ollama models list site. 

<img width="1141" height="657" alt="Image" src="https://github.com/user-attachments/assets/5991e328-7140-40b9-a5d0-cc4bebf08157" />

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
