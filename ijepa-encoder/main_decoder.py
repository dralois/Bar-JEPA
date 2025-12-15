import argparse
import multiprocessing as mp
import logging
import pprint
import yaml
import os

from src.utils.distributed import init_distributed
from src.train_decoder import main as app_main

def process_main(rank, fname, world_size, devices):
    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])

    # Configure logging
    logging.basicConfig()
    logger = logging.getLogger()

    # Add a log handler
    log_handler = logging.StreamHandler()
    logger.addHandler(log_handler)

    logger.info(f'Starting up {devices[rank]} -> {os.environ["CUDA_VISIBLE_DEVICES"]} (rank: {rank}/{world_size})')

    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f'Called with params {fname}')

    # Load config file
    params = None
    with open(fname, 'r') as y_file:
        params = yaml.safe_load(y_file)
        logger.info('Loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    # Initialize distributed training
    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f'Running... (rank: {rank}/{world_size})')

    try:
        app_main(args=params)
    except Exception as ex:
        print(ex)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--fname', type=str,
        help='Name of config file to load',
        default='configs/ijepa_ppn.yaml')
    parser.add_argument(
        '--devices', type=str, nargs='+', default=['cuda:0'],
        help='Which devices to use on local machine')

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
            args=(rank, args.fname, num_gpus, args.devices)
        ))
        processes[-1].start()

    for process in processes:
        process.join()
