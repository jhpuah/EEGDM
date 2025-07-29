<div align="center">
<br>
<img src="assets/title.png" width="166">
<h3>EEG Diffusion Model</h3></div>

<p align="center">
  <a href="https://arxiv.org/abs/2505.15809">
    <img
      src="https://img.shields.io/badge/MMaDA-Paper-red?logo=arxiv&logoColor=red"
      alt="MMaDA Paper on arXiv"
    />
  </a>
  <a href="https://huggingface.co/Gen-Verse/MMaDA-8B-Base">
    <img 
        src="https://img.shields.io/badge/MMaDA--8B--Base-Hugging%20Face%20Model-orange?logo=huggingface&logoColor=yellow" 
        alt="MMaDA on Hugging Face"
    />
  </a>


  
</p>

<div align="center">
<br>
<img src="assets/diff_v0.png" width="1066">
</div>

<div align="center">
<br>
<img src="assets/pool_v0.png" width="1066">
</div>

<div align="center">
<br>
<img src="assets/cls_v0.png" width="1066">
</div>


## 🌌 Introduction

EEG-Diff is a novel self-supervised diffusion model designed for superior EEG signal representation learning. Unlike traditional "tokenization-then-masking" approaches, EEG-Diff leverages the power of diffusion models to achieve robust and meaningful representations.

EEG-Diff is distinguished by three key innovations:

1. First Application of Diffusion Models for Direct EEG Representation Extraction: This work pioneers the direct use of diffusion models for extracting EEG signal representations, opening up a new avenue for research in neurological signal processing.

2. Time-Domain Gaussian Noise for Self-Supervision: EEG-Diff empirically demonstrates that adding Gaussian noise directly in the time domain can effectively achieve self-supervised representation learning, providing a promising alternative to existing methodologies.

3. Competitive Performance with Compact Design: The proposed method achieves performance exceeding previous domain-specific models and matches large EEG Foundation Models, despite using significantly less training data and a smaller model size.

## 😮 Hightlights

• This work is one of the first (if not the first) to use diffusion models directly in the extraction of EEG signal representations, thus opening a new research direction.  
• This work empirically shows that self-supervised representation learning can be achieved through adding Gaussian noise in the time domain directly, offering apromising alternative to the tokenization-then-masking method.  
• The proposed method achieved performance exceeding previous domain-specific models, and matches large EEG FMs despite the disadvantages in training data size and model size.  

## Main result

<div align="center">
<br>
<img src="assets/mainImage1.png" width="466">
</div>

<div align="center">
<br>
<img src="assets/mainImage2.png" width="566">
</div>

<div align="center">
<br>
<img src="assets/mainImage3.png" width="566">
</div>

<div align="center">
<br>
<img src="assets/mainImage4.png" width="566">
</div>

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
python main.py preprocessing=pretrain
```
Refering to preprocessing of LaBraM

**Pre-training:**
```bash
python main.py pretrain=base
```

**Caching:**
```bash
python main.py cache=base_t2
```

**Fine-tuning:**
```bash
python main.py finetune=base finetune.rng_seeding.seed=0
python main.py finetune=base_gatem finetune.rng_seeding.seed=0
python main.py finetune=base_filters finetune.rng_seeding.seed=0
python main.py finetune=base_filterm finetune.rng_seeding.seed=0
python main.py finetune=linear finetune.rng_seeding.seed=0
python main.py finetune=base_t2 finetune.rng_seeding.seed=0
python main.py finetune=nolaw finetune.rng_seeding.seed=0
python main.py finetune=noise finetune.rng_seeding.seed=0
```
All seeds need to be iterated from 0 to 4

**Reporting:**
```bash
python main.py report=base
```

**Other**
Scripts of certain ablation experiments are put in src/aux
```bash
python main.py aux=no_fusion aux.rng_seeding.seed=0
python main.py aux=mean_fusion aux.rng_seeding.seed=0
```
All seeds need to be iterated from 0 to 4

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


