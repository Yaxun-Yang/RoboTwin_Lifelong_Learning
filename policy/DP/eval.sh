#!/bin/bash

# == keep unchanged ==
policy_name=DP
task_name=${1}
task_config=${2}
ckpt_setting=${3}
expert_data_num=${4}
seed=${5}
gpu_id=${6}
# (optional) explicit checkpoint path inside ./policy/DP/checkpoints or an absolute path
# Usage (with custom ckpt): bash eval.sh beat_block_hammer demo_clean demo_clean 50 0 0 ./policy/DP/checkpoints/beat_block_hammer-0-50-0/600.ckpt
eval_ckpt_path=${7:-""}
DEBUG=False

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../..

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --expert_data_num ${expert_data_num} \
    --seed ${seed} \
    $( [ -n "${eval_ckpt_path}" ] && echo --eval_ckpt_path ${eval_ckpt_path} )