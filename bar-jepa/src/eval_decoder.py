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

from src.utils.ocr import NumericOCR
from src.datasets.charts import make_charts
from src.datasets.ubpmc import make_ubpmc
from src.transforms import make_transforms
from src.masks.charts import EvalCollator
from src.utils.logging import CSVLogger

from src.helper import (
    load_decoder_checkpoint,
    init_decoder_model
)

from src.utils.heatmap import (
    p_maps_to_cls_lists
)

from src.utils.postprocessing import (
    evaluate_gt_p_match,
    evaluate_value_accuracy,
    nms
)


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
        collator = EvalCollator()
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
            eval_mode=True,
            val_train_split=False,
            drop_last=False,
            shuffle=False)
    else:
        collator = EvalCollator()
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
            eval_mode=True,
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
        ('%d', 'samples'),
        ('%d', 'value_chart_count'),
        ('%.5f', 'value_acc_hard'),
        ('%.5f', 'value_acc_relaxed'),
        ('%.5f', 'bar_precision'),
        ('%.5f', 'bar_recall'),
        ('%.5f', 'bar_f1'),
        ('%.5f', 'tick_precision'),
        ('%.5f', 'tick_recall'),
        ('%.5f', 'tick_f1')
    )

    def load_chart(data, full_data, targets):
        img = data[0].to(device, non_blocking=True)
        full_img = full_data[0]
        grid = (img.shape[1] // patch_size, img.shape[2] // patch_size)
        gt_org = targets[0][0].to(device, non_blocking=True)
        gt_bars = targets[1][0].to(device, non_blocking=True)
        gt_ticks = targets[2][0].to(device, non_blocking=True)
        return [img], full_img, [grid], gt_org, gt_ticks, gt_bars

    def report_stats(split_name, stats):
        samples = stats['samples']
        value_count = stats['value_chart_count']

        if samples == 0:
            logger.warning(f'[{split_name}] No samples available for evaluation.')

        b_p = stats['bar_p'] / samples if samples > 0 else 0.0
        b_r = stats['bar_r'] / samples if samples > 0 else 0.0
        b_f1 = stats['bar_f1'] / samples if samples > 0 else 0.0
        t_p = stats['tick_p'] / samples if samples > 0 else 0.0
        t_r = stats['tick_r'] / samples if samples > 0 else 0.0
        t_f1 = stats['tick_f1'] / samples if samples > 0 else 0.0

        value_acc_hard = (
            stats['value_hard_total'] / value_count
            if value_count > 0 else 0.0
        )
        value_acc_relaxed = (
            stats['value_relaxed_total'] / value_count
            if value_count > 0 else 0.0
        )

        value_msg = (
            f'value acc hard/relaxed={value_acc_hard:.4f}/{value_acc_relaxed:.4f} (n={value_count})'
            if value_count > 0 else
            'value acc hard/relaxed=n/a (n=0)'
        )
        logger.info((
            f'[{split_name}] samples={samples} | '
            f'{value_msg} | '
            f'bars P/R/F1={b_p:.4f}/{b_r:.4f}/{b_f1:.4f} | '
            f'ticks P/R/F1={t_p:.4f}/{t_r:.4f}/{t_f1:.4f}'
        ))

        csv_logger.log(
            split_name,
            r_file,
            samples,
            value_count,
            value_acc_hard,
            value_acc_relaxed,
            b_p,
            b_r,
            b_f1,
            t_p,
            t_r,
            t_f1
        )

    encoder.eval()
    decoder.eval()

    split_stats = {
        'samples': 0,
        'value_hard_total': 0.0,
        'value_relaxed_total': 0.0,
        'value_chart_count': 0,
        'bar_p': 0.0,
        'bar_r': 0.0,
        'bar_f1': 0.0,
        'tick_p': 0.0,
        'tick_r': 0.0,
        'tick_f1': 0.0
    }

    with torch.inference_mode():
        test_sampler.set_epoch(0)

        eval_iter = tqdm(
            test_loader,
            total=len(test_loader),
            desc='Evaluating',
            dynamic_ncols=True
        )

        for data, full_data, targets in eval_iter:
            img, full_img, grid, gt_org, gt_ticks, gt_bars = load_chart(data, full_data, targets)

            # Inference
            h = encoder(img, grid)
            p_cls, p_reg, p_hm = [x[0] for x in decoder(h, grid)]
            size = torch.tensor(p_cls.shape[1:], device=device)

            # Radius for all thresholds = thresh / (sqrt(H*W) / 4.0)
            radius_thresh = eval_thresh / ((crop_size // patch_size) * 4.0)

            # Convert maps to bars & ticks + origin
            p_bars, p_ticks, _ = p_maps_to_cls_lists(
                p_cls, p_reg, size, pnt_thresh, cls_thresh
            )

            # Filter predictions using nms
            p_bars, p_ticks = nms(p_bars, p_ticks, radius_thresh)

            # Evaluate predictions
            b_p, b_r, b_f1, t_p, t_r, t_f1 = evaluate_gt_p_match(
                gt_bars, gt_ticks, p_bars, p_ticks, radius_thresh
            )

            split_stats['samples'] += 1
            split_stats['bar_p'] += b_p
            split_stats['bar_r'] += b_r
            split_stats['bar_f1'] += b_f1
            split_stats['tick_p'] += t_p
            split_stats['tick_r'] += t_r
            split_stats['tick_f1'] += t_f1

            # Extract OCR text, match ticks, and infer bar values
            if ocr_engine is not None:
                p_bars_yxv = ocr_engine.infer_bar_values(
                    p_bars, p_ticks, full_img, radius_thresh * 5.
                )
                hard_acc, relaxed_acc = evaluate_value_accuracy(
                    gt_bars, p_bars_yxv, radius_thresh
                )
                split_stats['value_hard_total'] += hard_acc
                split_stats['value_relaxed_total'] += relaxed_acc
                split_stats['value_chart_count'] += 1

        eval_iter.close()

    dataset_name = 'UB PMC' if is_ubpmc else 'Charts'
    report_stats(dataset_name, split_stats)

    logger.info(f'Wrote eval metrics to {log_file}')

if __name__ == '__main__':
    main({})
