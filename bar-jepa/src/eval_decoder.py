import os

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
import sys

import numpy as np

import torch

from tqdm.auto import tqdm

from src.utils.heatmap import (
    gt_maps_to_cls_lists,
    p_maps_to_cls_lists,
    evaluate_gt_p_match,
    nms
)
from src.utils.ocr import NumericOCR

from src.datasets.charts import make_charts
from src.datasets.ubpmc import make_ubpmc

from src.masks.charts import ChartsCollator, UBPMCCollator

from src.utils.logging import CSVLogger

from src.helper import (
    load_decoder_checkpoint,
    init_decoder_model
)

from src.transforms import make_transforms

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

# Initialize logging and distributed training
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def main(args):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- MODEL
    model_name = args['meta']['model_name']
    decoder_type = args['meta']['decoder_type']
    r_file = args['meta']['read_checkpoint']

    # -- DATA
    crop_size = args['data']['crop_size']
    patch_size = args['data']['patch_size']
    num_workers = args['data']['num_workers']
    pin_mem = args['data']['pin_mem']
    root_path = args['data']['root_path']
    is_ubpmc = args['data']['is_ubpmc']
    preserve_aspect_ratio = args['data']['preserve_aspect_ratio']
    patch_count = int((crop_size // patch_size) ** 2.)

    # -- KEYPOINT DETECTION
    use_aux_heads = args['keypoint']['use_aux_heads']
    num_hm_slots = args['keypoint']['hm_slots']
    pnt_thresh = args['keypoint']['pnt_detect_thresh']
    cls_thresh = args['keypoint']['cls_conf_thresh']
    eval_thresh = args['keypoint']['eval_thresh']
    score_norm = args['keypoint']['score_norm']

    # -- LOGGING
    folder =  args['logging']['folder']
    tag =  args['logging']['write_tag']
    os.makedirs(folder, exist_ok=True)

    # -- OCR
    ocr_enabled = args['ocr']['enabled']
    ocr_language = args['ocr']['language']
    ocr_rec_model = args['ocr']['rec_model']

    logger.info(f'Python version: {sys.version}, PyTorch version: {torch.__version__}')
    # ----------------------------------------------------------------------- #

    # -- create device
    if not torch.cuda.is_available():
        try:
            device = torch.device('cpu') if not torch.mps.is_available() else torch.device('mps')
        except AttributeError:
            device = torch.device('cpu')
        logger.warning(f'Falling back to {device.type}.')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- init model
    encoder, decoder = init_decoder_model(
        device=device,
        model_name=model_name,
        patch_size=patch_size,
        crop_size=crop_size,
        num_hm_slots=num_hm_slots,
        decoder_type=decoder_type,
        use_aux_heads=use_aux_heads)

    # -- make data transforms
    transform = make_transforms(
        crop_size=crop_size,
        random_resize_crop=False,
        horizontal_flip=False,
        color_distortion=False,
        gaussian_blur=False,
        preserve_aspect_ratio=preserve_aspect_ratio,
        max_patches=patch_count,
        patch_size=patch_size)

    # -- init test data-loaders/samplers
    if is_ubpmc:
        collator = UBPMCCollator()
        test_loader, test_sampler = make_ubpmc( # type: ignore
            transform=transform,
            batch_size=1,
            patch_size=patch_size,
            collator=collator,
            pin_mem=pin_mem,
            num_workers=num_workers,
            root_path=root_path,
            split=None,
            training=False,
            return_full_img=True,
            val_train_split=False,
            drop_last=False,
            shuffle=False)
    else:
        collator = ChartsCollator()
        test_loader, test_sampler = make_charts( # type: ignore
            transform=transform,
            batch_size=1,
            patch_size=patch_size,
            collator=collator,
            pin_mem=pin_mem,
            num_workers=num_workers,
            root_path=root_path,
            val_train_split=False,
            decoder_training=True,
            return_full_img=True,
            training=False,
            drop_last=False,
            shuffle=False)

    # -- load OCR engine if enabled
    ocr_engine = None
    if ocr_enabled:
        ocr_engine = NumericOCR(language=ocr_language, rec_model=ocr_rec_model)
        logger.info(
            f'Using PaddleOCR (language={ocr_language}, rec_model={ocr_rec_model}) '
            f'for numeric text extraction.'
        )
    else:
        logger.info('OCR is disabled by config.')

    # -- load training checkpoint
    encoder, decoder, _, _, _ = load_decoder_checkpoint(
        world_size=1,
        do_finetune=False,
        r_path=r_file,
        encoder=encoder,
        decoder=decoder,
        opt=None,
        scaler=None)

    # -- make csv_logger
    log_file = os.path.join(folder, f'{tag}.csv')
    csv_logger = CSVLogger(
        log_file,
        ('%s', 'dataset'),
        ('%s', 'checkpoint'),
        ('%.5f', 'pnt_detect_thresh'),
        ('%.5f', 'cls_conf_thresh'),
        ('%.5f', 'eval_thresh'),
        ('%d', 'samples'),
        ('%d', 'ocr_numeric'),
        ('%d', 'tick_text_matches'),
        ('%d', 'bars_with_value'),
        ('%.5f', 'mean_y_dist_per_value'),
        ('%.5f', 'bar_precision'),
        ('%.5f', 'bar_recall'),
        ('%.5f', 'bar_f1'),
        ('%.5f', 'tick_precision'),
        ('%.5f', 'tick_recall'),
        ('%.5f', 'tick_f1')
    )

    def load_chart(data, full_data, targets):
        img = data[0].to(device, non_blocking=True)
        full_img = full_data[0].cpu()
        grid = (img.shape[1] // patch_size, img.shape[2] // patch_size)
        gt_org = targets[0][0].to(device, non_blocking=True)
        gt_cls = targets[1][0].to(device, non_blocking=True)
        gt_reg = targets[2][0].to(device, non_blocking=True)
        return [img], full_img, [grid], gt_org, gt_cls, gt_reg

    def log_stats(split_name, stats):
        samples = stats['samples']

        if samples == 0:
            logger.warning(f'[{split_name}] No samples available for evaluation.')
            return

        logger.info((
            f'[{split_name}] samples={samples} | '
            f'bars P/R/F1={stats["bar_p"] / samples:.4f}/'
            f'{stats["bar_r"] / samples:.4f}/'
            f'{stats["bar_f1"] / samples:.4f} | '
            f'ticks P/R/F1={stats["tick_p"] / samples:.4f}/'
            f'{stats["tick_r"] / samples:.4f}/'
            f'{stats["tick_f1"] / samples:.4f} | '
            f'ocr_numeric={stats["ocr_numeric"]} | '
            f'tick_text_matches={stats["tick_text_matches"]} | '
            f'bars_with_value={stats["bars_with_value"]} | '
            f'mean_y_dist_per_value={stats["y_dist_per_value_sum"] / max(stats["value_mapping_count"], 1):.5f}'
        ))

    def avg_stats(stats):
        samples = stats['samples']
        if samples == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        return (
            stats['bar_p'] / samples,
            stats['bar_r'] / samples,
            stats['bar_f1'] / samples,
            stats['tick_p'] / samples,
            stats['tick_r'] / samples,
            stats['tick_f1'] / samples
        )

    encoder.eval()
    decoder.eval()

    split_stats = {
        'samples': 0,
        'ocr_numeric': 0,
        'tick_text_matches': 0,
        'bars_with_value': 0,
        'y_dist_per_value_sum': 0.0,
        'value_mapping_count': 0,
        'bar_p': 0.0,
        'bar_r': 0.0,
        'bar_f1': 0.0,
        'tick_p': 0.0,
        'tick_r': 0.0,
        'tick_f1': 0.0
    }

    with torch.no_grad():
        test_sampler.set_epoch(0)

        eval_iter = tqdm(
            test_loader,
            total=len(test_loader),
            desc='Evaluating',
            dynamic_ncols=True
        )

        for data, full_data, targets in eval_iter:
            img, full_img, grid, gt_org, gt_cls, gt_reg = load_chart(data, full_data, targets)
            size = torch.tensor(gt_cls.shape, device=device)

            # Inference
            h = encoder(img, grid)
            p_cls, p_reg, p_hm = [x[0] for x in decoder(h, grid)]

            # Extract numeric OCR labels from full-resolution images.
            if ocr_engine is not None:
                numeric_text = ocr_engine.extract_numeric_text(full_img)

            # Extract ground truth labels
            _, gt_bars, gt_ticks = gt_maps_to_cls_lists(
                gt_org, gt_cls, gt_reg, size
            )

            # Radius for nms is based on max(H, W)
            radius_thresh = eval_thresh / size.max()

            # Convert maps to bars & ticks + origin
            p_bars, p_ticks, _ = p_maps_to_cls_lists(
                p_cls, p_reg, size, pnt_thresh, cls_thresh, score_norm
            )

            # Filter predictions using nms
            p_bars, p_ticks = nms(p_bars, p_ticks, radius_thresh)

            # Match each predicted tick to closest numeric OCR text
            tick_text_matches = []
            bar_value_pairs = []
            y_dist_per_value = None
            if ocr_engine is not None:
                tick_text_matches = ocr_engine.match_ticks_to_numeric_text(
                    p_ticks, numeric_text
                )
                bar_value_pairs, y_dist_per_value = ocr_engine.infer_bar_values(
                    p_bars, tick_text_matches
                )

            # Evaluate predictions
            b_p, b_r, b_f1, t_p, t_r, t_f1 = evaluate_gt_p_match(
                gt_bars, gt_ticks, p_bars, p_ticks, radius_thresh
            )

            split_stats['samples'] += 1
            split_stats['ocr_numeric'] += len(numeric_text)
            split_stats['tick_text_matches'] += len(tick_text_matches)
            split_stats['bars_with_value'] += len(bar_value_pairs)
            if y_dist_per_value is not None:
                split_stats['y_dist_per_value_sum'] += y_dist_per_value
                split_stats['value_mapping_count'] += 1
            split_stats['bar_p'] += b_p
            split_stats['bar_r'] += b_r
            split_stats['bar_f1'] += b_f1
            split_stats['tick_p'] += t_p
            split_stats['tick_r'] += t_r
            split_stats['tick_f1'] += t_f1

        eval_iter.close()

    dataset_name = 'UB PMC' if is_ubpmc else 'Charts'

    log_stats(dataset_name, split_stats)

    b_p, b_r, b_f1, t_p, t_r, t_f1 = avg_stats(split_stats)
    mean_y_dist_per_value = (
        split_stats['y_dist_per_value_sum'] / split_stats['value_mapping_count']
        if split_stats['value_mapping_count'] > 0 else 0.0
    )

    csv_logger.log(
        dataset_name,
        r_file,
        pnt_thresh,
        cls_thresh,
        eval_thresh,
        split_stats['samples'],
        split_stats['ocr_numeric'],
        split_stats['tick_text_matches'],
        split_stats['bars_with_value'],
        mean_y_dist_per_value,
        b_p,
        b_r,
        b_f1,
        t_p,
        t_r,
        t_f1
    )

    logger.info(f'Wrote eval metrics to {log_file}')

if __name__ == '__main__':
    main({})
