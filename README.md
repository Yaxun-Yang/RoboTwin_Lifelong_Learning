# README

**This section of the codebase is maintained by *Siwen Wang*.  
It includes the baseline Diffusion Policy implementation, Full Fine-Tuning (FFT), and Experience Replay (ER) for continual learning, as well as the training and evaluation pipelines.**

---

## 📦 Environment Setup & Data Collection

### 1. Install RoboTwin Environment  
(Used for **data collection** & **evaluation simulation**)  
Official documentation:  
https://robotwin-platform.github.io/doc/usage/robotwin-install.html

### 2. Collect Data

```bash
conda activate RoboTwin

bash collect_data.sh adjust_bottle demo_clean 0
bash collect_data.sh click_bell demo_clean 0
bash collect_data.sh beat_block_hammer demo_clean 0
```

---

## 🚀 Training Pipeline

### 1. Install Training Environment

```bash
cd policy/DP
conda create -n DP python=3.10 -y
conda activate DP

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install zarr==2.12.0 wandb ipdb gpustat dm_control omegaconf hydra-core==1.2.0 dill==0.3.5.1 \
            einops==0.4.1 diffusers==0.11.1 numba==0.56.4 moviepy imageio av matplotlib termcolor sympy

pip install -e .
```

---

### 2. Train from Scratch  
(Tasks: **A = adjust_bottle**, **B = click_bell**, **C = beat_block_hammer**)

```bash
# Example: train click_bell from scratch
bash train.sh click_bell demo_clean 50 0 14 0
```

---

### 3. Sequential Full Fine-Tuning (FFT)

```bash
bash train.sh click_bell demo_clean 50 0 14 0 \
     checkpoints/adjust_bottle-demo_clean-50-0/100.ckpt True
```

---

### 4. Experience Replay (ER)

```bash
bash train.sh click_bell demo_clean 50 0 14 0 \
     checkpoints/adjust_bottle-demo_clean-50-0/100.ckpt True \
     "adjust_bottle:0.2"
```

---

## 🧪 Evaluation

```bash
conda activate RoboTwin
cd policy/DP

bash eval.sh click_bell demo_clean demo_clean 50 0 0 \
     /home/swwang/RoboTwin/policy/DP/checkpoints/click_bell-demo_clean-50-0/100.ckpt
```

---

## 📁 Dataset Links (Google Drive)

- **[data](https://drive.google.com/drive/folders/1bDUKtdn7YhcI2A_O925cyDWV-zKcbY_z?usp=drive_link)** — placed under: `policy/DP/`

- **[embodiments](https://drive.google.com/drive/folders/1K7QGRzTrpN4R6hoovEzSxaNTSPUVzSQa?usp=drive_link)** — placed under: `assets/`

- **[objects](https://drive.google.com/drive/folders/1_GfGB3apqn5yetQbEAj4j5EAKZpIw3HT?usp=drive_link)** — placed under: `assets/`


---

