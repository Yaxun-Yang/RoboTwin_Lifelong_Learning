# README




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

## 🚀 Training Pipeline for DP

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
## 🚀 Training Pipeline for TinyVLA 
The whole process is refer to ` https://robotwin-platform.github.io/doc/usage/TinyVLA.html`
### 1. Install Training Environment and Modify Evaluation Environment
for training:
```bash
cd policy/TinyVLA
conda env create -f Train_Tiny_DexVLA_train.yml
conda activate dexvla-robo
cd policy_heads
pip install -e .
```
for evaluation:

```bash
conda activate your_RoboTwin_env
pip install -r Eval_Tiny_DexVLA_requirements.txt
```
---
### 2. Convert Data
convert original RoboTwin 2.0 data into the format required for TinyVLA training.
```bash
cd policy/TinyVLA
# python process_data.py ${task_name} ${task_config} ${expert_data_num}
python process_data.py beat_block_hammer demo_clean 50
```
---
### 3. Download VLM model 

VLM model InternVL3-1B(https://huggingface.co/OpenGVLab/InternVL3-1B/tree/main) is used here.
The model should be downloaded into `.../policy/TinyVLA/model_param/InternVL3-1B`
Then modify the config.json file in the folder as follows:

```json
{
    "_name_or_path": ".../robotiwin/policy/TinyVLA/vla/models/internvl", # Modify this.
    "architectures": [
        "TinyVLA" # Change this.
    ],
    # "auto_map":{...} # Delete this.
    ...
    "llm_config": {}, # Don't Change.
    "min_dynamic_patch": 1,
    "model_type": "tinyvla", # Change this.
    ...
}
```
---
### 3. Full Parameter Training
(Tasks: **A = adjust_bottle**, **B = click_bell**, **C = beat_block_hammer**)

```bash
bash ./scripts/franks/train_robotwin_aloha.sh
```
Configure the training by modifying the following items in the train_robotwin_aloha.sh file.
```bash
TASK=your_task # Set the Task
ROOT=.../robotiwin/policy/TinyVLA # Set Root Path
mnop=.../robotiwin/policy/TinyVLA/model_param/InternVL3-1B/ # Set The Path of base VLM
```
---

### 4. LoRA Training
modify line 73 in `policy/TinyVLA/train_vla.py`
```python
lora_enable: bool = True
``` 
---

## 🧪 Evaluation
for dp:
```bash
conda activate RoboTwin
cd policy/DP

bash eval.sh click_bell demo_clean demo_clean 50 0 0 \
     /home/swwang/RoboTwin/policy/DP/checkpoints/click_bell-demo_clean-50-0/100.ckpt
```
for TinyVLA:
modify the corresponding path in the deploy_policy.yml file: 1. model_path : Path to the trained model, in the OUTPUT path. 2. state_path : Path to dataset_stats.pkl, in the OUTPUT path. 3. model_base : Path to InternVL3-1B.
```bash
# bash eval.sh ${task_name} ${task_config} ${ckpt_setting} ${expert_data_num} ${seed} ${gpu_id}
 bash eval.sh beat_block_hammer demo_clean 0 50 0 0
```
---

## 📁 Dataset Links (Google Drive)

- **[data](https://drive.google.com/drive/folders/1bDUKtdn7YhcI2A_O925cyDWV-zKcbY_z?usp=drive_link)** — placed under: `policy/DP/`

- **[embodiments](https://drive.google.com/drive/folders/1K7QGRzTrpN4R6hoovEzSxaNTSPUVzSQa?usp=drive_link)** — placed under: `assets/`

- **[objects](https://drive.google.com/drive/folders/1_GfGB3apqn5yetQbEAj4j5EAKZpIw3HT?usp=drive_link)** — placed under: `assets/`


---

