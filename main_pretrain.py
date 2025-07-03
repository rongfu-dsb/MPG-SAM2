import argparse
import datetime
import json
import random
import time
from pathlib import Path
from collections import namedtuple
from functools import partial

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils
import datasets.samplers as samplers
from datasets.coco_eval import CocoEvaluator
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from models.postprocessors import build_postprocessors
import deepspeed
import opts


def main(args):
    # set environ
    os.environ["MDETR_CPU_REDUCE"] = "1"

    args.masks = True
    assert args.dataset_file in ["refcoco", "refcoco+", "refcocog", "all"]

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_bifit_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if match_name_keywords(n, args.lr_bifit_names) and p.requires_grad],
            "lr": args.lr_bifit,
        },

    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    # build train  dataset
    if args.dataset_file != "all":
        dataset_train = build_dataset(args.dataset_file, image_set='train', args=args)
    else:
        dataset_names = ["refcoco", "refcoco+", "refcocog"]
        dataset_train = torch.utils.data.ConcatDataset(
            [build_dataset(name, image_set="train", args=args) for name in dataset_names]
        )

    print("\nTrain dataset sample number: ", len(dataset_train))
    print("\n")

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        }
    }
    model, optimizer, data_loader_train, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        model_parameters=model.parameters(),
        training_data=dataset_train,
        collate_fn=utils.collate_fn,
        config=ds_config,
    )
    model.to(device)

    output_dir = Path(args.output_dir)

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            model_paths = [os.path.join(output_dir, 'ckpt_model')]
            for i in range(1):
                checkpoint_path = checkpoint_paths[i]
                model_path = model_paths[i]
                save_checkpoint(model, checkpoint_path, model_path, optimizer, lr_scheduler, args, epoch)


        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     # **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def save_checkpoint(model_engine, checkpoint_path, model_path, optimizer, lr_scheduler, args, epoch):
    """ Saves the model checkpoint. """
    # If the checkpoint is the best, save it in ckpt_model_best, else in ckpt_model_last_epoch
    # save_dir_name = "ckpt_model_best" if is_best else "ckpt_model_last_epoch"
    #
    # save_dir = os.path.join(args.log_dir, save_dir_name)
    # Ensure the directory exists
    utils.save_on_master({
        'optimizer': optimizer.state_dict(),
        # 'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch,
        'args': args,
    }, checkpoint_path)
    torch.distributed.barrier()
    model_engine.save_checkpoint(model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('RVOSNet pretrain training and evaluation script',
                                     parents=[opts.get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

