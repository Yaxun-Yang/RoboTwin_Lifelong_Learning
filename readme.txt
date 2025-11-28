# Our code is based on the RoboTwin codebase (https://github.com/robotwin-Platform/RoboTwin) and its documentation (https://robotwin-platform.github.io/doc/).
# We use this framework to collect data and perform training and testing, and we additionally implemented the continual learning module.



# ========== collect data ==========
# install RoboTwin environment (for data collection & evaluation environment)
https://robotwin-platform.github.io/doc/usage/robotwin-install.html

# collect data
conda activate RoboTwin
bash collect_data.sh adjust_bottle demo_clean 0
bash collect_data.sh click_bell demo_clean 0
bash collect_data.sh beat_block_hammer demo_clean 0



# ========== training ==========
# install training environment
cd policy/DP
conda create -n DP python=3.10 -y
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install zarr==2.12.0 wandb ipdb gpustat dm_control omegaconf hydra-core==1.2.0 dill==0.3.5.1 einops==0.4.1 diffusers==0.11.1 numba==0.56.4 moviepy imageio av matplotlib termcolor sympy
pip install -e .


# train from scratch (A=adjust_bottle, B=click_bell, C=beat_block_hammer)
# bash train.sh adjust_bottle demo_clean 50 0 14 0
bash train.sh click_bell demo_clean 50 0 14 0
# bash train.sh beat_block_hammer demo_clean 50 0 14 0


# train sequential Full Fine-Tuning (FFT)
bash train.sh click_bell demo_clean 50 0 14 0 checkpoints/adjust_bottle-demo_clean-50-0/100.ckpt True


# train Experience Replay (ER)
bash train.sh click_bell demo_clean 50 0 14 0 checkpoints/adjust_bottle-demo_clean-50-0/100.ckpt True "adjust_bottle:0.2"



# ========== evaluation ==========
conda activate RoboTwin
cd policy/DP
bash eval.sh click_bell demo_clean demo_clean 50 0 0 /home/swwang/RoboTwin/policy/DP/checkpoints/click_bell-demo_clean-50-0/100.ckpt

