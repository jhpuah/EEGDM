```
conda create -n proper python=3.11
conda activate proper
pip install numpy==1.26.4 hydra-core mne torch torchvision torchaudio lightning pyhealth ema-pytorch diffusers einops wandb scipy pyhealth
```
or
```
pip install -r requirements.txt
```

```
python main.py preprocessing=faithful

python main.py pretrain=base

python main.py finetune=base finetune.rng_seeding.seed=0

python main.py report=base
```
