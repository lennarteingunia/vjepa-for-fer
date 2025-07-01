# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

import torch.distributed


from datasets.full_video_dataset import make_fullvideodata
from evals.video_classification_frozen.utils import ClipAggregation, FrameAggregation, make_transforms
from utils.logging import ClassificationMeter, FullVideoClassificationMeter

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import logging
import pprint

import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel

import src.models.vision_transformer as vit
from src.models.attentive_pooler import AttentiveClassifier
from src.utils.distributed import (
    init_distributed
)

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)


def main(args_eval, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- PRETRAIN
    args_pretrain = args_eval.get('pretrain')
    checkpoint_key = args_pretrain.get('checkpoint_key', 'target_encoder')
    model_name = args_pretrain.get('model_name', None)
    patch_size = args_pretrain.get('patch_size', None)
    pretrain_folder = args_pretrain.get('folder', None)
    ckp_fname = args_pretrain.get('checkpoint', None)
    tag = args_pretrain.get('write_tag', None)
    use_sdpa = args_pretrain.get('use_sdpa', True)
    use_SiLU = args_pretrain.get('use_silu', False)
    tight_SiLU = args_pretrain.get('tight_silu', True)
    uniform_power = args_pretrain.get('uniform_power', False)
    pretrained_path = os.path.join(pretrain_folder, ckp_fname)
    # Optional [for Video model]:
    tubelet_size = args_pretrain.get('tubelet_size', 2)
    pretrain_frames_per_clip = args_pretrain.get('frames_per_clip', 1)

    # -- DATA
    args_data = args_eval.get('data')
    val_data_path = args_data.get('dataset_val')
    if not (type(val_data_path) is list):
        val_data_path = [val_data_path]
    num_classes = args_data.get('num_classes')
    eval_frames_per_clip = args_data.get('frames_per_clip', 16)
    eval_frame_step = args_pretrain.get('frame_step', 4)

    # -- OPTIMIZATION
    args_opt = args_eval.get('optimization')
    resolution = args_opt.get('resolution', 224)
    batch_size = args_opt.get('batch_size')
    attend_across_segments = args_opt.get('attend_across_segments', False)

    eval_tag = args_eval.get('tag', None)

    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')

    # -- log/checkpointing paths
    folder = os.path.join(pretrain_folder, 'video_classification_frozen/')
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')

    # -- pretrained encoder (frozen)
    encoder = init_model(
        crop_size=resolution,
        device=device,
        pretrained=pretrained_path,
        model_name=model_name,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        frames_per_clip=pretrain_frames_per_clip,
        uniform_power=uniform_power,
        checkpoint_key=checkpoint_key,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
        use_sdpa=use_sdpa)

    if pretrain_frames_per_clip == 1:
        # Process each frame independently and aggregate
        encoder = FrameAggregation(encoder).to(device)
    else:
        # Process each video clip independently and aggregate
        encoder = ClipAggregation(
            encoder,
            tubelet_size=tubelet_size,
            attend_across_segments=attend_across_segments
        ).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # -- init classifier
    classifier = AttentiveClassifier(
        embed_dim=encoder.embed_dim,
        num_heads=encoder.num_heads,
        depth=1,
        num_classes=num_classes,
    ).to(device)

    val_loader = make_dataloader(
        root_path=val_data_path,
        resolution=resolution,
        frames_per_clip=eval_frames_per_clip,
        frame_step=eval_frame_step,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
    )

    ipe = len(val_loader)

    logger.info(f'Dataloader created... iterations per epoch: {ipe}')

    classifier = DistributedDataParallel(classifier, static_graph=True)

    classifier = load_checkpoint(
        r_path=latest_path,
        classifier=classifier,
    )

    run_validation(
        device=device,
        attend_across_segments=attend_across_segments,
        encoder=encoder,
        classifier=classifier,
        data_loader=val_loader,
        num_classes=num_classes,
        write_tag=tag,
    )


def run_validation(
    device,
    encoder,
    classifier,
    data_loader,
    attend_across_segments,
    num_classes,
    *,
    write_tag: str = ''
):

    classifier.train(mode=False)

    if not 'SLURM_JOB_ID' in os.environ:
        raise ValueError(f'$SLURM_JOB_ID not set! Can\'t log like this.')
    log_dir = f'/mnt/slurm/lennart/jepaslt/logs/additional/{os.environ["SLURM_JOB_ID"]}'

    meter = FullVideoClassificationMeter(
        num_classes=num_classes)
    clip_meter = ClassificationMeter(num_classes=num_classes)

    def log(msg, it) -> None:
        logger.info(
            f'rank {torch.distributed.get_rank()}/iteration {it}: {msg}')

    # Data is of form (buffer, label, indices)
    for iteration, data in enumerate(data_loader):

        def _log(msg): return log(msg=msg, it=iteration)

        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=True):

            # Load data and put on GPU
            clips = [
                [dij.to(device, non_blocking=True)
                 for dij in di]  # iterate over spatial views of clip
                for di in data[0]  # iterate over temporal index of clip
            ]
            clip_indices = [d.to(device, non_blocking=True)
                            for d in data[2]['indices']]
            labels = data[1].to(device)

            video_paths = data[2]['path']

            # Forward and prediction
            with torch.no_grad():
                outputs = encoder(clips, clip_indices)
                if attend_across_segments:
                    outputs = [classifier(o) for o in outputs]
                else:
                    outputs = [[classifier(ost) for ost in os]
                               for os in outputs]

        with torch.no_grad():

            if attend_across_segments:

                outputs = sum(
                    [F.softmax(o, dim=1) for o in outputs]
                ) / len(outputs)

            else:
                outputs = sum(
                    [
                        sum([F.softmax(ost, dim=1) for ost in os])
                        for os in outputs
                    ]
                ) / len(outputs) / len(outputs[0])

            meter.update(confidences=outputs, labels=labels, paths=video_paths)
            clip_meter.update(predictions=outputs.argmax(dim=1), labels=labels)

            if iteration % 20 == 0:

                _log(
                    f'Current Top 1 Accuracy on Full Video: {meter.top1_acc}\nCurrent Top 1 PC Accuracy on Full Video: {meter.top1_pc_acc}\nCurrent Top 1 Accuracy on Clips: {clip_meter.top1_acc}\nCurrent Top 1 PC Accuracy on Clips: {clip_meter.top1_pc_acc}')
                
                meter.tofile(
                    log_dir, 
                    file_prefix=f'{write_tag}_r{torch.distributed.get_rank()}_')
                
                clip_meter.tofile(
                    log_dir, 
                    file_prefix=f'clip_{write_tag}_r{torch.distributed.get_rank()}_'
                )
                
    _log(
        f'Current Top 1 Accuracy on Full Video: {meter.top1_acc}\nCurrent Top 1 PC Accuracy on Full Video: {meter.top1_pc_acc}\nCurrent Top 1 Accuracy on Clips: {clip_meter.top1_acc}\nCurrent Top 1 PC Accuracy on Clips: {clip_meter.top1_pc_acc}')

    meter.tofile(
        log_dir, 
        file_prefix=f'{write_tag}_r{torch.distributed.get_rank()}_'
    )
                
    clip_meter.tofile(
        log_dir, 
        file_prefix=f'clip_{write_tag}_r{torch.distributed.get_rank()}_'
    )


def load_checkpoint(
    r_path,
    classifier,
):
    checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
    epoch = checkpoint['epoch']

    # -- loading encoder
    pretrained_dict = checkpoint['classifier']
    msg = classifier.load_state_dict(pretrained_dict)
    logger.info(
        f'loaded pretrained classifier from epoch {epoch} with msg: {msg}')

    del checkpoint

    return classifier


def load_pretrained(
    encoder,
    pretrained,
    checkpoint_key='target_encoder'
):
    logger.info(f'Loading pretrained model from {pretrained}')
    checkpoint = torch.load(pretrained, map_location='cpu')
    try:
        pretrained_dict = checkpoint[checkpoint_key]
    except Exception:
        pretrained_dict = checkpoint['encoder']

    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace('backbone.', ''): v for k, v in pretrained_dict.items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(
                f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info(f'loaded pretrained model with msg: {msg}')
    logger.info(
        f'loaded pretrained encoder from epoch: {checkpoint["epoch"]}\n path: {pretrained}')
    del checkpoint
    return encoder


def make_dataloader(
    root_path,
    batch_size,
    world_size,
    rank,
    resolution=224,
    frames_per_clip=16,
    frame_step=4,
    num_views_per_segment=1,
    training=False,
    num_workers=12,
):
    # Make Video Transforms
    transform = make_transforms(
        training=training,
        num_views_per_clip=num_views_per_segment,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(0.75, 4/3),
        random_resize_scale=(0.08, 1.0),
        reprob=0.25,
        auto_augment=True,
        motion_shift=False,
        crop_size=resolution,
    )

    _dataset, data_loader, _data_sampler = make_fullvideodata(
        data_paths=root_path,
        batch_size=batch_size,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        transform=transform,
        world_size=world_size,
        rank=rank,
        num_workers=num_workers,
        logger=logger
    )

    return data_loader


def init_model(
    device,
    pretrained,
    model_name,
    patch_size=16,
    crop_size=224,
    # Video specific parameters
    frames_per_clip=16,
    tubelet_size=2,
    use_sdpa=False,
    use_SiLU=False,
    tight_SiLU=True,
    uniform_power=False,
    checkpoint_key='target_encoder'
):
    encoder = vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=frames_per_clip,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
    )

    encoder.to(device)
    encoder = load_pretrained(
        encoder=encoder, pretrained=pretrained, checkpoint_key=checkpoint_key)
    return encoder
