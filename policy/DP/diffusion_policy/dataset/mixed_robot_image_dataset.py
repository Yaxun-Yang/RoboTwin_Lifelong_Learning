"""MODIFIED_BY_SW: MixedRobotImageDataset

Mixes in data from multiple additional tasks according to specified ratios.
Each element in extras should be a dictionary of the form:
{'task': task_name, 'ratio': float_ratio, 'zarr_path': path}

Ratio interpretation:
The number of appended episodes ≈ (number of episodes in the main task) * ratio
(with a minimum of 1 episode and not exceeding the total number of episodes available in that additional task).
"""
from typing import Dict, List, Optional
import torch, numpy as np, copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer


class MixedRobotImageDataset(BaseImageDataset):
    def __init__(
        self,
        main_zarr_path: str,
        extras: Optional[List[Dict]] = None,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        seed: int = 42,
        val_ratio: float = 0.0,
        batch_size: int = 128,
        max_train_episodes: Optional[int] = None,
    ):
        super().__init__()
        rng = np.random.default_rng(seed)

        main_rb = ReplayBuffer.copy_from_path(main_zarr_path, keys=["head_camera", "state", "action"])
        n_main = main_rb.n_episodes

        merged = ReplayBuffer.create_empty_numpy()
        for epi in range(n_main):
            merged.add_episode(main_rb.get_episode(epi, copy=True))

        if extras:
            for e in extras:
                task = e.get('task'); ratio = float(e.get('ratio', 0.0) or 0.0); zpath = e.get('zarr_path')
                if not task or ratio <= 0.0 or not zpath:
                    continue
                try:
                    extra_rb = ReplayBuffer.copy_from_path(zpath, keys=["head_camera", "state", "action"])
                    n_extra = extra_rb.n_episodes
                    if n_extra == 0:
                        print(f"[MixedDataset] {task} empty, skip.")
                        continue
                    take = int(round(n_main * ratio))
                    take = max(1, min(take, n_extra))
                    idxs = rng.permutation(n_extra)[:take]
                    for epi in idxs:
                        merged.add_episode(extra_rb.get_episode(int(epi), copy=True))
                    print(f"[MixedDataset] Added {take} episodes from {task} (ratio={ratio})")
                except Exception as ex:
                    print(f"[MixedDataset] Failed {task}: {ex}")

        self.replay_buffer = merged
        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.batch_size = batch_size

        seq_len = self.sampler.sequence_length
        self.buffers = {k: np.zeros((batch_size, seq_len, *v.shape[1:]), dtype=v.dtype) for k, v in self.sampler.replay_buffer.items()}
        self.buffers_torch = {k: torch.from_numpy(v) for k, v in self.buffers.items()}
        for v in self.buffers_torch.values():
            v.pin_memory()

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {"action": self.replay_buffer["action"], "agent_pos": self.replay_buffer["state"]}
        norm = LinearNormalizer(); norm.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        norm["head_cam"] = get_image_range_normalizer()
        norm["front_cam"] = get_image_range_normalizer()
        norm["left_cam"] = get_image_range_normalizer()
        norm["right_cam"] = get_image_range_normalizer()
        return norm

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            sample = self.sampler.sample_sequence(idx)
            sample = dict_apply(sample, torch.from_numpy)
            return sample
        elif isinstance(idx, np.ndarray):
            assert len(idx) == self.batch_size
            from diffusion_policy.dataset.robot_image_dataset import batch_sample_sequence
            for k, v in self.sampler.replay_buffer.items():
                batch_sample_sequence(self.buffers[k], v, self.sampler.indices, idx, self.sampler.sequence_length)
            return self.buffers_torch
        raise ValueError(idx)

    def postprocess(self, samples, device):
        agent_pos = samples["state"].to(device, non_blocking=True)
        head_cam = samples["head_camera"].to(device, non_blocking=True) / 255.0
        action = samples["action"].to(device, non_blocking=True)
        return {"obs": {"head_cam": head_cam, "agent_pos": agent_pos}, "action": action}
