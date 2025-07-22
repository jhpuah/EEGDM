<div align="center">
<br>
<img src="title.png" width="166">
<h3>EEG Diffusion Model</h3></div>

<p align="center">
  <a href="https://arxiv.org/abs/2505.15809">
    <img
      src="https://img.shields.io/badge/MMaDA-Paper-red?logo=arxiv&logoColor=red"
      alt="MMaDA Paper on arXiv"
    />
  </a>
  <a href="https://huggingface.co/spaces/Gen-Verse/MMaDA">
    <img 
        src="https://img.shields.io/badge/MMaDA%20Demo-Hugging%20Face%20Space-blue?logo=huggingface&logoColor=blue" 
        alt="MMaDA on Hugging Face"
    />
  </a>
  <a href="https://huggingface.co/Gen-Verse/MMaDA-8B-Base">
    <img 
        src="https://img.shields.io/badge/MMaDA--8B--Base-Hugging%20Face%20Model-orange?logo=huggingface&logoColor=yellow" 
        alt="MMaDA on Hugging Face"
    />
  </a>
    <a href="https://huggingface.co/Gen-Verse/MMaDA-8B-MixCoT">
    <img 
        src="https://img.shields.io/badge/MMaDA--8B--MixCoT-Hugging%20Face%20Model-orange?logo=huggingface&logoColor=yellow" 
        alt="MMaDA on Hugging Face"
    />
  </a>
  <a href="https://github.com/Gen-Verse/MMaDA/blob/main/assets/wx-mmada-0613.jpeg">
    <img 
        src="https://img.shields.io/badge/Wechat-Join-green?logo=wechat&amp" 
        alt="Wechat Group Link"
    />
  </a>
  
</p>


## 🌌 Introduction

EEGDM is a novel EEG diffusion model designed for generating realistic EEG signals. This project aims to provide a framework for research and development in the field of EEG synthesis and analysis.

## 📰 Latest Updates

*   **[2025-07-16]** Initial setup and README update.

## ⚙️ Quick Start

First, set up the environment:

```bash
conda create -n proper python=3.11
conda activate proper
pip install numpy==1.26.4 hydra-core mne torch torchvision torchaudio lightning pyhealth ema-pytorch diffusers einops wandb scipy pyhealth
```
or
```bash
pip install -r requirements.txt
```

### Usage Examples:

**Preprocessing:**
```bash
python main.py preprocessing=faithful
```

**Pre-training:**
```bash
python main.py pretrain=base
```

**Fine-tuning:**
```bash
python main.py finetune=base finetune.rng_seeding.seed=0
```

**Reporting:**
```bash
python main.py report=base
```

## 📖 Citation

If you use this work, please cite:

```
@article{eegdm2025,
  title={EEGDM: A Novel EEG Diffusion Model},
  author={Your Name(s) Here},
  journal={Preprint},
  year={2025}
}
```

## 🤝 Acknowledgments

This work is inspired by and builds upon various open-source projects and research in diffusion models and EEG processing. We acknowledge the contributions of the communities behind PyTorch, Hugging Face Diffusers, MNE-Python, and other related libraries.

## 💬 Discussion and Collaboration

We welcome discussions and collaborations to improve EEGDM. Please feel free to open issues or pull requests on GitHub.


