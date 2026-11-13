# 🤖 MedRAX Reproduction on ChestAgentBench

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Uyhuynh23/MedRAX-ChestAgentBench/blob/main/notebooks/MedRAX_ChestAgentBench_Colab.ipynb)
[![Dataset on HuggingFace](https://img.shields.io/badge/HuggingFace-ChestAgentBench-yellow?logo=huggingface)](https://huggingface.co/datasets/wanglab/chest-agent-bench)
[![arXiv](https://img.shields.io/badge/arXiv-2502.02673-red)](https://arxiv.org/abs/2502.02673)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

Official reproduction and evaluation suite for **MedRAX** (*Medical Reasoning Agent for Chest X-ray*, ICML 2025) on the **ChestAgentBench** benchmark, tailored for seamless execution on **Google Colab** with **Google Drive persistent caching** and **GitHub version control**.

---

## 📑 Table of Contents
1. [Overview & Reproduction Goals](#-overview--reproduction-goals)
2. [Google Drive Persistent Architecture](#-google-drive-persistent-architecture)
3. [Repository Layout](#-repository-layout)
4. [Quickstart on Google Colab](#-quickstart-on-google-colab)
5. [Hardware & GPU Memory Profiles](#-hardware--gpu-memory-profiles)
6. [GitHub Setup & Version Control Guide](#-github-setup--version-control-guide)
7. [Running Evaluations Locally or Headless](#-running-evaluations-locally-or-headless)
8. [Benchmarking Axes & Metric Analysis](#-benchmarking-axes--metric-analysis)
9. [Citation](#-citation)

---

## 🎯 Overview & Reproduction Goals

**MedRAX** is the first versatile AI agent that dynamically orchestrates specialized chest radiograph analysis tools using multimodal LLMs (GPT-4o, Gemini, Claude) to solve complex multi-step clinical reasoning tasks.

**ChestAgentBench** is the accompanying clinical benchmark containing:
- **2,500 expert-curated multiple-choice questions**
- **609 unique clinical cases** from peer-reviewed European radiology case reports (Eurorad)
- **1,346 thoracic radiograph figures**
- **7 clinical competency categories**: Detection, Classification, Localization, Comparison, Relationship, Diagnosis, and Characterization.

### This Repository Delivers:
- **Zero Data-Loss Colab Architecture**: Pretrained weights (~30 GB) and unzipped image files are stored once in Google Drive and symlinked transparently.
- **Ready-to-Run Colab Notebook**: [MedRAX_ChestAgentBench_Colab.ipynb](notebooks/MedRAX_ChestAgentBench_Colab.ipynb) with 1-click execution.
- **Hardware Agnostic**: Automatic profile switching between free **T4 (15GB)** with 8-bit quantization and **A100 (40GB/80GB)** full tool suites.
- **Automated Logging & Resume**: Interrupted runs automatically pick up where they left off without duplicate API calls.

---

## 💾 Google Drive Persistent Architecture

In Google Colab, default disk storage (`/content/`) is wiped when the runtime disconnects. To prevent re-downloading multi-gigabyte models and datasets on every session, this setup structures your Google Drive as follows:

```
Google Drive: /content/drive/MyDrive/MedRAX_Thesis/
├── checkpoints/              # Pretrained model weights (CheXagent, Maira-2, LLaVA-Med, PSPNet)
├── cache/
│   ├── huggingface/          # HF_HOME: Transformers & Hub cache
│   └── torch/                # PyTorch hub model cache
├── data/
│   └── chestagentbench/      # metadata.jsonl + unzipped figures/<case_id>/
├── results/                  # predictions.jsonl, summary.json, plots
└── logs/                     # Verbose agent reasoning traces & tool calls
```

When you run `python scripts/setup_drive.py`, it automatically creates these directories and symlinks them into `./checkpoints`, `./cache`, and `./data` in your working directory.

---

## 📂 Repository Layout

```
MedRAX-ChestAgentBench/
├── .github/                  # CI/CD and issue templates
├── .gitignore                # Comprehensive ignore rules (ignoring large weights, data, cache, keys)
├── .env.example              # Template for API keys and paths
├── pyproject.toml            # Project packaging specification
├── requirements.txt          # Resolved dependencies for Colab (numpy<2, PyTorch, LangGraph, etc.)
├── LICENSE                   # Apache 2.0 license
├── README.md                 # This documentation
│
├── configs/                  # Hardware-specific execution configurations
│   ├── config_t4.yaml        # Profile for Colab T4 (15GB VRAM, 8-bit quantized)
│   ├── config_a100.yaml      # Profile for Colab A100 (40GB/80GB VRAM, full tools)
│   └── config_api_only.yaml  # Fast API-only multimodal baseline profile
│
├── notebooks/                # Turnkey interactive reproduction notebooks
│   └── MedRAX_ChestAgentBench_Colab.ipynb
│
├── scripts/                  # Standalone automation scripts
│   ├── setup_drive.py        # Initializes Google Drive layout and sets environment variables
│   ├── download_benchmark.py # Downloads & unzips ChestAgentBench with integrity checks
│   └── run_evaluation.py     # CLI benchmark runner with resume and metric logging
│
├── medrax/                   # Core MedRAX agent source code
│   ├── agent/                # LangGraph state machine & agent logic
│   ├── tools/                # Specialized radiology tools (classifier, segmentation, VQA, etc.)
│   ├── llava/                # LLaVA-Med model architectures and weights loader
│   ├── utils/                # Image decoding, prompt parsers, formatting helpers
│   └── docs/                 # System prompt templates
│
├── benchmark/                # Benchmark dataset schemas and question parsers
└── experiments/              # Paper reproduction scripts, ACE playbooks, and comparison tools
```

---

## 🚀 Quickstart on Google Colab

1. Click the **Open In Colab** badge at the top of this README.
2. Select **Runtime -> Change runtime type -> T4 GPU** (or **A100 GPU** if using Colab Pro).
3. Set your API keys in Colab Secrets (the 🔑 icon in the sidebar) or enter them in the key prompt cell:
   - `OPENAI_API_KEY`: Required for the agent reasoning backbone (GPT-4o).
   - `HF_TOKEN`: Required for downloading gated models like Maira-2.
4. Run the notebook cells sequentially:
   - Cell 1-2: Verifies GPU & mounts Google Drive.
   - Cell 3-4: Clones this repository and installs requirements.
   - Cell 5-6: Downloads ChestAgentBench once into your Drive.
   - Cell 7: Runs an interactive visual inspection of radiographs.
   - Cell 8: Runs the benchmark evaluation (start with `--max-cases 2` for a 1-minute smoke test).
   - Cell 9: Generates accuracy charts and exports summary reports.

---

## ⚡ Hardware & GPU Memory Profiles

| Hardware Tier | Memory | Recommended Configuration | Available Tools |
| :--- | :--- | :--- | :--- |
| **Colab Free (T4)** | 15 GB VRAM | `configs/config_t4.yaml` (`load_in_8bit=True`) | Classifier, PSPNet Segmentation, SwinV2 Report Generator, CheXagent VQA |
| **Colab Pro (L4)** | 24 GB VRAM | `configs/config_t4.yaml` | Classifier, Segmentation, Report Generator, CheXagent, Maira-2 Grounding (8-bit) |
| **Colab Pro (A100)** | 40/80 GB VRAM | `configs/config_a100.yaml` | **All Tools**: Classifier, Segmentation, Report Gen, CheXagent, Maira-2, LLaVA-Med |
| **CPU / API Only** | Any | `configs/config_api_only.yaml` | Zero local GPU: Direct Vision-Language Model baseline |

---

## 🐙 GitHub Setup & Version Control Guide

### 1. Linking your Local Workspace to GitHub

In your local terminal (inside `c:\Users\LENOVO\Desktop\THESIS\MedRAX-ChestAgentBench`):

```bash
# 1. Initialize git (if not already done)
git init
git branch -M main

# 2. Add all tracked files
git add .
git commit -m "feat: complete MedRAX reproduction suite for ChestAgentBench with Colab and Drive integration"

# 3. Create a new repository on your GitHub account (named MedRAX-ChestAgentBench)
# Then link and push:
git remote add origin https://github.com/Uyhuynh23/MedRAX-ChestAgentBench.git
git push -u origin main
```

### 2. Pushing Updates from Google Colab Back to GitHub

Whenever you edit the notebook or generate new experiment configs inside Colab:

```python
# In Colab:
!git config --global user.name "Uyhuynh23"
!git config --global user.email "htuy23@apcs.fitus.edu.vn"

# Commit updates
!git add configs/ scripts/ notebooks/
!git commit -m "Update reproduction configs and evaluation results"

# Push using your GitHub Personal Access Token (PAT)
!git push https://<YOUR_GITHUB_TOKEN>@github.com/Uyhuynh23/MedRAX-ChestAgentBench.git main
```

*(Create a GitHub Personal Access Token at [github.com/settings/tokens](https://github.com/settings/tokens) with `repo` scope).*

---

## 💻 Running Evaluations Locally or Headless

You can also run evaluations via CLI on any machine with Python 3.10+ and CUDA:

```bash
# 1. Set environment variables
export OPENAI_API_KEY="your-key"
export BENCHMARK_DATA_DIR="./data/chestagentbench"
export MEDRAX_RESULTS_DIR="./results"

# 2. Run Direct Baseline on 10 cases:
python scripts/run_evaluation.py \
    --mode baseline \
    --model chatgpt-4o-latest \
    --max-cases 10

# 3. Run Full MedRAX Agent on all cases with automatic resume:
python scripts/run_evaluation.py \
    --mode medrax \
    --model chatgpt-4o-latest \
    --device cuda
```

---

## 📊 Benchmarking Axes & Metric Analysis

ChestAgentBench evaluates clinical reasoning across 7 distinct axes:
- **Detection**: Finding presence or absence of specific lesions (pneumothorax, effusion, nodules).
- **Classification**: Categorizing pathology severity, opacity types, or cardiomegaly grade.
- **Localization**: Identifying specific anatomical lobes, quadrants, or mediastinal compartments.
- **Comparison**: Assessing temporal change across prior vs current radiographs.
- **Relationship**: Spatial relationships between devices/tubes and anatomical landmarks.
- **Diagnosis**: Formulating differential and definitive clinical diagnoses.
- **Characterization**: Describing margins, density, calcification, and contours.

Compare multiple runs using the built-in comparator:
```bash
python experiments/compare_runs.py results/baseline/predictions.jsonl results/medrax/predictions.jsonl
```

---

## 📚 Citation

If you use MedRAX or ChestAgentBench in your thesis, please cite the original authors:

```bibtex
@misc{fallahpour2025medraxmedicalreasoningagent,
      title={MedRAX: Medical Reasoning Agent for Chest X-ray}, 
      author={Adibvafa Fallahpour and Jun Ma and Alif Munim and Hongwei Lyu and Bo Wang},
      year={2025},
      eprint={2502.02673},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.02673}, 
}
```
