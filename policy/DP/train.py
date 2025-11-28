"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import hydra, pdb
from omegaconf import OmegaConf
import pathlib, yaml
from diffusion_policy.workspace.base_workspace import BaseWorkspace

import os

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("diffusion_policy", "config")),
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    head_camera_type = cfg.head_camera_type
    head_camera_cfg = get_camera_config(head_camera_type)
    cfg.task.image_shape = [3, head_camera_cfg["h"], head_camera_cfg["w"]]
    cfg.task.shape_meta.obs.head_cam.shape = [
        3,
        head_camera_cfg["h"],
        head_camera_cfg["w"],
    ]
    OmegaConf.resolve(cfg)
    cfg.task.image_shape = [3, head_camera_cfg["h"], head_camera_cfg["w"]]
    cfg.task.shape_meta.obs.head_cam.shape = [
        3,
        head_camera_cfg["h"],
        head_camera_cfg["w"],
    ]

    # ========== Data Replay Parsing ========== MODIFIED_BY_SW
    try:
        if hasattr(cfg.training, 'replay') and cfg.training.replay.enable and cfg.training.replay.spec:
            spec_items = [s.strip() for s in str(cfg.training.replay.spec).split(',') if s.strip()]
            extras = []
            for it in spec_items:
                if ':' not in it:
                    continue
                task, ratio_str = it.split(':', 1)
                task = task.strip(); ratio = float(ratio_str)
                if ratio <= 0.0:
                    continue
                # Construct extra task zarr path (same as main task setting & expert_data_num)
                zarr_path = f"data/{task}-{cfg.setting}-{cfg.expert_data_num}.zarr"
                extras.append({'task': task, 'ratio': ratio, 'zarr_path': zarr_path})
            if extras:
                from omegaconf import OmegaConf as _OC
                base_ds = cfg.task.dataset
                new_ds = {
                    '_target_': 'diffusion_policy.dataset.mixed_robot_image_dataset.MixedRobotImageDataset',
                    'main_zarr_path': base_ds.zarr_path,
                    'extras': extras,
                    'horizon': base_ds.horizon,
                    'pad_before': base_ds.pad_before,
                    'pad_after': base_ds.pad_after,
                    'seed': base_ds.seed,
                    'val_ratio': base_ds.val_ratio,
                    'batch_size': base_ds.batch_size,
                    'max_train_episodes': base_ds.max_train_episodes,
                }
                cfg.task.dataset = _OC.create(new_ds)
                print(f"[Replay] Enabled. Extras={extras}")
            else:
                print("[Replay] Spec provided but no valid extras parsed.")
    except Exception as e:
        print(f"[Replay] Failed enabling mixed dataset: {e}")

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)

    # ========== Loading pretrained checkpoint ========== MODIFIED_BY_SW
    try:
        if (
            hasattr(cfg.training, 'enable_pretrained') and cfg.training.enable_pretrained and
            hasattr(cfg.training, 'pretrained_ckpt') and cfg.training.pretrained_ckpt not in [None, 'null', '', '""']
        ):
            ckpt_path = cfg.training.pretrained_ckpt
            if os.path.isfile(ckpt_path):
                import torch, dill
                print(f"[Finetune] Loading pretrained checkpoint: {ckpt_path}")
                payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
                # drop optimizer & lr_scheduler states
                for drop_key in ['optimizer', 'lr_scheduler']:
                    if drop_key in payload['state_dicts']:
                        del payload['state_dicts'][drop_key]
                workspace.load_payload(payload, exclude_keys=None, include_keys=None)
                
                if hasattr(workspace, 'epoch'): workspace.epoch = 0
                if hasattr(workspace, 'global_step'): workspace.global_step = 0
                # lr
                if hasattr(cfg.training, 'lr_finetune_scale') and cfg.training.lr_finetune_scale not in [None, 1.0]:
                    scale = cfg.training.lr_finetune_scale
                    if hasattr(workspace, 'optimizer'):
                        for pg in workspace.optimizer.param_groups:
                            if 'lr' in pg:
                                old_lr = pg['lr']; pg['lr'] = old_lr * scale
                        print(f"[Finetune] LR scaled by {scale}")
            else:
                print(f"[Finetune] Pretrained ckpt not found: {ckpt_path}, skip.")
        else:
            print("[Finetune] Disabled or no ckpt provided; training from scratch.")
    except Exception as e:
        print(f"[Finetune] Failed loading pretrained: {e}")

    # Print dataset source for sanity, handling both single and mixed datasets
    try:
        ds_cfg = cfg.task.dataset
        src = None
        if hasattr(ds_cfg, 'zarr_path'):
            src = ds_cfg.zarr_path
        elif hasattr(ds_cfg, 'main_zarr_path'):
            src = ds_cfg.main_zarr_path
        tname = cfg.task.name if hasattr(cfg, 'task') and hasattr(cfg.task, 'name') else str(getattr(cfg, 'task_name', 'unknown'))
        print(f"[Dataset] source={src} task={tname}")
    except Exception as _e:
        print(f"[Dataset] print skipped: {_e}")
    workspace.run()


if __name__ == "__main__":
    main()
