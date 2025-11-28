#!/bin/bash

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
action_dim=${5}
gpu_id=${6}
# 7th parameter: Pretrained ckpt (can be empty)
pretrained_ckpt=${7:-""}
# 8th parameter: Whether to enable pretraining (True/False)
enable_pretrained=${8:-False}
# 9th parameter: Replay spec (task:ratio,task2:ratio2)
replay_spec=${9:-""}

head_camera_type=D435

DEBUG=False
save_ckpt=True

alg_name=robot_dp_$action_dim
config_name=${alg_name}
addition_info=train
exp_name=${task_name}-robot_dp-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

# Adjust experiment name based on pretraining/replay
if [ "$enable_pretrained" = "True" ]; then
    exp_name="${exp_name}-pretrained"
else
    exp_name="${exp_name}-scratch"
fi
if [ -n "$replay_spec" ]; then
    # Use '-' instead of '=' to avoid inserting '=' into hydra overrides (causes parse errors)
    replay_tag=$(echo "$replay_spec" | sed -E 's/:/-/g; s/,/-/g')
    exp_name="${exp_name}-replay"
    run_dir="data/outputs/${exp_name}_${replay_tag}_seed${seed}"
fi

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


if [ $DEBUG = True ]; then
    wandb_mode=offline
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

if [ ! -d "./data/${task_name}-${task_config}-${expert_data_num}.zarr" ]; then
    bash process_data.sh ${task_name} ${task_config} ${expert_data_num}
fi

cmd=(python train.py --config-name=${config_name}.yaml \
    task.name=${task_name} \
    task.dataset.zarr_path="data/${task_name}-${task_config}-${expert_data_num}.zarr" \
    training.debug=$DEBUG \
    training.seed=${seed} \
    training.device="cuda:0" \
    exp_name=${exp_name} \
    logging.mode=${wandb_mode} \
    setting=${task_config} \
    expert_data_num=${expert_data_num} \
    head_camera_type=$head_camera_type \
    # Wrap pretrained path in single quotes so Hydra treats any '=' in the path as part of the value
    training.pretrained_ckpt="'${pretrained_ckpt}'" \
    training.enable_pretrained=${enable_pretrained} \
    training.replay.enable=$( [ -n "$replay_spec" ] && echo True || echo False ) \
    training.replay.spec="'${replay_spec}'" \
    hydra.run.dir="${run_dir}")

echo "[train.sh] CMD: ${cmd[*]}"
"${cmd[@]}"
echo "[train.sh] replay_spec='${replay_spec}' enable_pretrained=${enable_pretrained} pretrained_ckpt='${pretrained_ckpt}'"