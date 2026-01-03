# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

import multiprocessing as mp

import logging

import pprint
import yaml

from src.utils.distributed import init_distributed

parser = argparse.ArgumentParser()
parser.add_argument(
    '--mode', type=str,
    help='which training to run',
    default='pretrain',)
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs.yaml')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine')


def process_main(rank, mode, fname, world_size, devices):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])

    logging.basicConfig()
    logger = logging.getLogger()

    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f'called-params {fname}')

    # -- load script params
    params = None
    with open(fname, 'r') as y_file:
        params = yaml.safe_load(y_file)
        logger.info('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f'Running... (rank: {rank}/{world_size})')

    try:
        match mode:
            case 'pretrain':
                from src.train import main as pretrain_main
                pretrain_main(args=params)
            case 'finetune':
                from src.train_finetune import main as finetune_main
                finetune_main(args=params)
            case 'decoder':
                from src.train_decoder import main as decoder_main
                decoder_main(args=params)
            case _:
                raise ValueError(f'Unknown mode: {mode}')
    except Exception as ex:
        print(ex)


if __name__ == '__main__':
    args = parser.parse_args()

    num_gpus = len(args.devices)

    try:
        mp.freeze_support()
        mp.set_start_method('spawn')
    except Exception:
        pass

    processes = []
    for rank in range(num_gpus):
        processes.append(mp.Process(
            target=process_main,
            args=(rank, args.mode, args.fname, num_gpus, args.devices)
        ))
        processes[-1].start()

    for process in processes:
        process.join()

