from __future__ import annotations

import logging
import torchcodec
import os
from typing import Any, List
import numpy as np
import pandas as pd
import torch
import torch.utils.data.distributed
import tqdm
import sha3
import yaml


def make_fullvideodata(
    data_paths: List[str],
    batch_size: int,
    world_size: int,
    frames_per_clip: int,
    frame_step: int,
    rank: int,
    transform,
    *,
    num_workers: int = 10,
    force_indexing: bool = True,
    disable_indexing_tqdm: bool = True,
    logger: logging.Logger = logging.getLogger(__name__)
):

    dataset = FullVideoDataset(
        data_paths=data_paths,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        transform=transform,
        logger=logger,
        min_pad=True,
        force_indexing=force_indexing,
        disable_tqdm=disable_indexing_tqdm,
    )

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    return dataset, data_loader, sampler


class FullVideoDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_paths: List[str],
        frames_per_clip: int = 16,
        frame_step: int = 4,
        *,
        transform=None,
        logger: logging.Logger = logging.getLogger(__name__),
        min_pad: bool = True,
        force_indexing: bool = False,
        disable_tqdm: bool = True
    ) -> None:
        super(FullVideoDataset, self).__init__()

        video_paths, labels = [], []
        for data_path in data_paths:
            data = pd.read_csv(data_path, header=None, delimiter=' ')
            video_paths += list(data.values[:, 0])
            labels += list(data.values[:, 1])

        self.clip_len = int(frames_per_clip * frame_step)
        self.logger = logger
        self.transform = transform

        # Creating indexing file path if it exists.
        id_string = ''.join(map(str, video_paths + labels))
        id_string = id_string + str(min_pad)
        id_string = id_string + str(frames_per_clip)
        id_string = id_string + str(frame_step)
        k = sha3.keccak_512()
        k.update(id_string.encode('utf-8'))
        index_file_id = k.hexdigest()
        index_file_name = index_file_id + '.yaml'
        # TODO: This cannot be the final location for tmp files.
        self.index_file_path = index_file_name

        if not os.path.exists(self.index_file_path) or force_indexing:
            self.samples = []

            num_dropped_videos = 0
            total_num_videos = 0
            p_bar = tqdm.tqdm(zip(video_paths, labels), total=len(
                video_paths), disable=disable_tqdm)
            for video_path, label in p_bar:

                video_decoder = torchcodec.decoders.VideoDecoder(
                    source=video_path,
                )

                if video_decoder.metadata.num_frames is None:
                    raise ValueError(
                        f"Video at {video_path} does not contain any video. Aborting...")

                num_frames = video_decoder.metadata.num_frames

                num_clips = num_frames - self.clip_len

                if num_clips < 1:
                    if not min_pad:
                        logger.info(

                            f'Dropping video at {video_path}, because it does not fit a single clip')
                        num_dropped_videos += 1
                        continue

                    # If min_pad is enabled we pad the clip index with the last frame of the video.

                    if min_pad:

                        indices = np.linspace(
                            0, num_frames, num=num_frames // frame_step)
                        indices = np.concatenate(
                            (indices, np.ones(frames_per_clip - num_frames // frame_step) * num_frames,))
                        indices = np.clip(
                            indices, 0, num_frames - 1).astype(np.int64)

                        self.samples.append({
                            'path': video_path,
                            'indices': indices,
                            'label': label,
                            'position': 0,
                            'video_len': num_clips
                        })

                        total_num_videos += 1
                else:
                    total_num_videos += 1

                p_bar.set_description(
                    f'Indexing videos. Number of indexed videos: {total_num_videos}; Number of dropped videos: {num_dropped_videos}.')

                for start_idx in range(num_clips):

                    end_idx = start_idx + self.clip_len

                    indices = np.linspace(
                        start_idx,
                        end_idx,
                        num=frames_per_clip
                    )
                    indices = np.clip(
                        indices,
                        start_idx,
                        end_idx - 1
                    ).astype(np.int64)

                    self.samples.append({
                        'path': video_path,
                        'indices': indices,
                        'label': label,
                        'position': start_idx,
                        'video_len': num_clips
                    })

            if not force_indexing:

                logger.info(
                    f'Creating clip index file at {self.index_file_path}.')

                if os.path.exists(self.index_file_path):
                    os.remove(self.index_file_path)
                with open(self.index_file_path, 'w') as tmp_index_file:
                    writable = [entry.copy() for entry in self.samples]
                    for entry in writable:
                        entry['indices'] = entry['indices'].tolist()
                    yaml.dump(writable, tmp_index_file,
                              default_flow_style=False)

            logger.info(
                f'Indexed a total number of {len(self.unique_videos)} videos containing {len(self.samples)} samples.')

        else:
            logger.info(
                f'Loading clip index from {self.index_file_path}. If you want to avoid this use the \"force_update\" flag or delete the index file.')
            with open(self.index_file_path, 'r') as tmp_index_file:
                read = yaml.load(tmp_index_file, Loader=yaml.FullLoader)
                for entry in read:
                    entry['indices'] = np.array(entry['indices'])
                self.samples = [entry.copy() for entry in read]
            logger.info(
                f'Read index file from {self.index_file_path} with a total of {len(self.samples)} samples from {len(self.unique_videos)} videos.')

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index) -> Any:
        sample = self.samples[index]

        video_decoder = torchcodec.decoders.VideoDecoder(
            source=sample["path"]
        )

        clip = video_decoder.get_frames_at(sample["indices"])
        clip = clip.data

        if self.transform is not None:
            clip = self.transform(clip)

        return (
            [clip],
            sample['label'],
            {
                'path': sample['path'],
                'indices': [sample['indices']],
                'position': sample['position'],
                'video_len': sample['video_len']
            }
        )

    @property
    def unique_videos(self):
        return list(set(map(lambda sample: sample['path'], self.samples)))

    def save_index_to_tmp_file(self):
        if os.path.exists(self.index_file_path):
            self.logger.info(
                f'Not saving index file to {self.index_file_path} since it already exists.')
            return
        with open(self.index_file_path, 'w+') as f:
            yaml.dump(self.samples, f)

    def load_index_from_tmp_file(self):
        with open(self.index_file_path, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    dataset = FullVideoDataset(
        ['/mnt/datasets/CREMA-D/additional/splits/split_0.csv'],
        min_pad=True,
        logger=logger
    )

    # This way I can check that actually all videos were indexed and are loaded from.
    logger.info(
        f'Number of unique indexed videos: {len(dataset.unique_videos)}.')

    unique_loaded_videos = set()
    for clips, lables, meta in tqdm.tqdm(iter(dataset)):
        unique_loaded_videos.add(meta['path'])
    logger.info(f'Number of unique loaded videos: {len(unique_loaded_videos)}')
