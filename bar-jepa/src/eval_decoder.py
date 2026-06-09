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

import matplotlib.pyplot as plt
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
    evaluate_confusion,
    nms
)


_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

# Initialize logging and distributed training
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

# Bucket thresholds for per-category ablation
# Values are the p95 of the decoder training distribution
# The generator produces a discrete set of values, so p95 equals the maximum
_BAR_THRESH   = 15    # p95 of training bar counts  (max=15, mean=7.0)
_TICK_THRESH  = 10    # p95 of training tick counts  (max=10, mean=7.7)
_AR_NORMAL_LO = 0.75  # p5  of training aspect ratio (min=0.75)
_AR_NORMAL_HI = 1.34  # p95 of training aspect ratio (min=8/6=1.33..)

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
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']
    ds_name = 'UB PMC' if is_ubpmc else 'Synthetic Charts'
    ds_path = 'ubpmc' if is_ubpmc else 'charts'
    os.makedirs(folder, exist_ok=True)

    # -- OCR
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

    # -- init OCR engine
    ocr_engine = NumericOCR(language=ocr_language, rec_model=ocr_rec_model)
    logger.info(
        f'Using PaddleOCR (language={ocr_language}, rec_model={ocr_rec_model}) '
        f'for numeric text extraction.'
    )

    # -- load training checkpoint
    encoder, decoder, _, _, _ = load_decoder_checkpoint(
        world_size=1,
        do_finetune=False,
        r_path=r_file,
        encoder=encoder,
        decoder=decoder,
        opt=None,
        scaler=None)

    # -- make csv loggers
    metrics_log_file = os.path.join(folder, f'{tag}_{ds_path}.csv')
    metrics_csv_logger = CSVLogger(
        metrics_log_file,
        ('%d', 'samples'),
        ('%.5f', 'acc_hard'),
        ('%.5f', 'acc_relaxed'),
        ('%.5f', 'oracle_acc_hard'),
        ('%.5f', 'oracle_acc_relaxed'),
        ('%.5f', 'bar_precision'),
        ('%.5f', 'bar_recall'),
        ('%.5f', 'bar_f1'),
        ('%.5f', 'tick_precision'),
        ('%.5f', 'tick_recall'),
        ('%.5f', 'tick_f1'),
    )

    cat_log_file = os.path.join(folder, f'{tag}_{ds_path}_categories.csv')
    cat_csv_logger = CSVLogger(
        cat_log_file,
        ('%s', 'category'),
        ('%s', 'bucket'),
        ('%d', 'samples'),
        ('%.5f', 'bar_f1'),
        ('%.5f', 'tick_f1'),
    )

    cm_plot_file = os.path.join(folder, f'{tag}_{ds_path}_confusion.png')

    def load_chart(data, full_data, targets):
        img = data[0].to(device, non_blocking=True)
        full_img = full_data[0]
        grid = (img.shape[1] // patch_size, img.shape[2] // patch_size)
        gt_org = targets[0][0].to(device, non_blocking=True)
        gt_bars = targets[1][0].to(device, non_blocking=True)
        gt_ticks = targets[2][0].to(device, non_blocking=True)
        return [img], full_img, [grid], gt_org, gt_ticks, gt_bars

    def report_stats(stats, cat_stats):
        samples = stats['samples']

        if samples == 0:
            logger.warning(f'No samples available for evaluation.')

        b_p = stats['bar_p'] / samples if samples > 0 else 0.0
        b_r = stats['bar_r'] / samples if samples > 0 else 0.0
        b_f1 = stats['bar_f1'] / samples if samples > 0 else 0.0
        t_p = stats['tick_p'] / samples if samples > 0 else 0.0
        t_r = stats['tick_r'] / samples if samples > 0 else 0.0
        t_f1 = stats['tick_f1'] / samples if samples > 0 else 0.0

        acc_hard = stats['acc_hard_total'] / samples if samples > 0 else 0.0
        acc_relaxed = stats['acc_relaxed_total'] / samples if samples > 0 else 0.0
        oracle_acc_hard = stats['oracle_acc_hard_total'] / samples if samples > 0 else 0.0
        oracle_acc_relaxed = stats['oracle_acc_relaxed_total'] / samples if samples > 0 else 0.0

        cm = stats['confusion_matrix']
        logger.info((
            f'samples={samples} | '
            f'acc hard/relaxed={acc_hard:.4f}/{acc_relaxed:.4f} | '
            f'oracle acc hard/relaxed={oracle_acc_hard:.4f}/{oracle_acc_relaxed:.4f} | '
            f'bars P/R/F1={b_p:.4f}/{b_r:.4f}/{b_f1:.4f} | '
            f'ticks P/R/F1={t_p:.4f}/{t_r:.4f}/{t_f1:.4f}'
        ))
        logger.info(
            f'Confusion matrix (rows=GT, cols=Pred):\n'
            f'              p_bar     p_tick     p_none\n'
            f'  gt_bar      {cm["tp_bar"]:>8d}  {cm["bar_as_tick"]:>9d}  {cm["fn_bar"]:>9d}\n'
            f'  gt_tick     {cm["tick_as_bar"]:>8d}  {cm["tp_tick"]:>9d}  {cm["fn_tick"]:>9d}\n'
            f'  gt_none     {cm["fp_bar"]:>8d}  {cm["fp_tick"]:>9d}          –'
        )

        metrics_csv_logger.log(
            samples,
            acc_hard,
            acc_relaxed,
            oracle_acc_hard,
            oracle_acc_relaxed,
            b_p,
            b_r,
            b_f1,
            t_p,
            t_r,
            t_f1
        )

        matrix = np.array([
            [cm['tp_bar'],    cm['bar_as_tick'], cm['fn_bar']],
            [cm['tick_as_bar'], cm['tp_tick'],   cm['fn_tick']],
            [cm['fp_bar'],    cm['fp_tick'],     np.nan],
        ], dtype=float)

        fig, ax = plt.subplots(figsize=(6, 5))
        masked = np.ma.masked_invalid(matrix)
        im = ax.imshow(masked, cmap='Blues')
        plt.colorbar(im, ax=ax)

        row_labels = ['GT Bar', 'GT Tick', 'GT None (FP)']
        col_labels = ['Pred Bar', 'Pred Tick', 'Pred None (FN)']
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)
        ax.set_title(f'{ds_name} ({tag.replace("eval_", "")})')

        vmax = float(masked.max()) if masked.count() > 0 else 1.0
        for i in range(3):
            for j in range(3):
                val = matrix[i, j]
                if np.isnan(val):
                    ax.text(j, i, '–', ha='center', va='center', color='#aaaaaa', fontsize=14)
                else:
                    color = 'white' if val > 0.6 * vmax else 'black'
                    ax.text(j, i, str(int(val)), ha='center', va='center', color=color)

        fig.tight_layout()
        fig.savefig(cm_plot_file, dpi=150)
        plt.close(fig)

        for (dim, bucket), cs in cat_stats.items():
            n = cs['n']
            cbf1 = cs['bar_f1'] / n if n > 0 else 0.0
            ctf1 = cs['tick_f1'] / n if n > 0 else 0.0
            cat_csv_logger.log(dim, bucket, n, cbf1, ctf1)

    encoder.eval()
    decoder.eval()

    split_stats = {
        'samples': 0,
        'acc_hard_total': 0.0,
        'acc_relaxed_total': 0.0,
        'oracle_acc_hard_total': 0.0,
        'oracle_acc_relaxed_total': 0.0,
        'bar_p': 0.0,
        'bar_r': 0.0,
        'bar_f1': 0.0,
        'tick_p': 0.0,
        'tick_r': 0.0,
        'tick_f1': 0.0,
        'confusion_matrix': {
            'tp_bar': 0,
            'bar_as_tick': 0,
            'fn_bar': 0,
            'fp_bar': 0,
            'tp_tick': 0,
            'tick_as_bar': 0,
            'fn_tick': 0,
            'fp_tick': 0,
        },
    }

    category_stats = {
        ('bars',  'in_dist'): {'n': 0, 'bar_f1': 0.0, 'tick_f1': 0.0},
        ('bars',  'ood'):     {'n': 0, 'bar_f1': 0.0, 'tick_f1': 0.0},
        ('ticks', 'in_dist'): {'n': 0, 'bar_f1': 0.0, 'tick_f1': 0.0},
        ('ticks', 'ood'):     {'n': 0, 'bar_f1': 0.0, 'tick_f1': 0.0},
        ('ar',    'in_dist'): {'n': 0, 'bar_f1': 0.0, 'tick_f1': 0.0},
        ('ar',    'ood'):     {'n': 0, 'bar_f1': 0.0, 'tick_f1': 0.0},
    }

    act_folder = os.path.join(folder, 'activations', tag)
    os.makedirs(act_folder, exist_ok=True)
    _CLS_CHANNEL_NAMES = ['Background', 'Bar', 'Tick']

    with torch.inference_mode():
        test_sampler.set_epoch(0)

        eval_iter = tqdm(
            test_loader,
            total=len(test_loader),
            desc='Evaluating',
            dynamic_ncols=True
        )

        for sample_idx, (data, full_data, targets) in enumerate(eval_iter):
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

            # Save sigmoid-activated classification maps + chart image
            act_maps = torch.sigmoid(p_cls).cpu().numpy()
            fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
            for ch, (ax, name) in enumerate(zip(axes, _CLS_CHANNEL_NAMES)):
                im = ax.imshow(act_maps[ch], cmap='viridis', vmin=0.0, vmax=1.0)
                ax.set_title(name, fontsize=9)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            h_img, w_img = full_img.shape[:2]
            axes[3].imshow(full_img)

            # Scatter predicted and GT bars/ticks
            for pts, color, marker, label in [
                (gt_bars,  '#2ecc71', 'o', 'GT bar'),
                (gt_ticks, '#e67e22', 'o', 'GT tick'),
                (p_bars,   '#27ae60', 'x', 'Pred bar'),
                (p_ticks,  '#d35400', 'x', 'Pred tick'),
            ]:
                if pts is not None and len(pts) > 0:
                    pts_np = pts.cpu().numpy()
                    xs = pts_np[:, 1] * (w_img - 1)
                    ys = pts_np[:, 0] * (h_img - 1)
                    axes[3].scatter(xs, ys, s=20, c=color, marker=marker,
                                    linewidths=1.2, label=label, zorder=3)
            axes[3].legend(fontsize=6, loc='upper right', framealpha=0.7)
            axes[3].set_title('Chart', fontsize=9)
            axes[3].axis('off')

            # Save in corresponding tags' folder
            arp_label = 'ARP' if preserve_aspect_ratio else 'No ARP'
            act_file = os.path.join(act_folder, f'{tag}_{ds_path}_{sample_idx:04d}_cls.png')
            fig.suptitle(f'{ds_name} — {decoder_type.capitalize()} {arp_label} — sample {sample_idx:04d}', fontsize=10)
            fig.tight_layout()
            fig.savefig(act_file, dpi=120)
            plt.close(fig)

            # Evaluate predictions
            b_p, b_r, b_f1, t_p, t_r, t_f1 = evaluate_gt_p_match(
                gt_bars, gt_ticks, p_bars, p_ticks, radius_thresh
            )

            # F1 scores
            split_stats['samples'] += 1
            split_stats['bar_p'] += b_p
            split_stats['bar_r'] += b_r
            split_stats['bar_f1'] += b_f1
            split_stats['tick_p'] += t_p
            split_stats['tick_r'] += t_r
            split_stats['tick_f1'] += t_f1

            # Per-category bucketing (bars, ticks, aspect ratio)
            bar_count = int(gt_bars.shape[0])
            tick_count = int(gt_ticks.shape[0])
            h_img, w_img = full_img.shape[:2]
            ar = w_img / h_img if h_img > 0 else 1.0
            bar_bucket  = 'in_dist' if bar_count  <= _BAR_THRESH else 'ood'
            tick_bucket = 'in_dist' if tick_count <= _TICK_THRESH else 'ood'
            ar_bucket   = 'in_dist' if _AR_NORMAL_LO <= ar <= _AR_NORMAL_HI else 'ood'
            for dim, bucket in [('bars', bar_bucket), ('ticks', tick_bucket), ('ar', ar_bucket)]:
                cs = category_stats[(dim, bucket)]
                cs['n'] += 1
                cs['bar_f1'] += b_f1
                cs['tick_f1'] += t_f1

            # Confusion matrix
            tp_bar, bar_as_tick, fn_bar, fp_bar, tp_tick, tick_as_bar, fn_tick, fp_tick = evaluate_confusion(
                gt_bars, gt_ticks, p_bars, p_ticks, radius_thresh
            )
            cm = split_stats['confusion_matrix']
            cm['tp_bar'] += tp_bar
            cm['bar_as_tick'] += bar_as_tick
            cm['fn_bar'] += fn_bar
            cm['fp_bar'] += fp_bar
            cm['tp_tick'] += tp_tick
            cm['tick_as_bar'] += tick_as_bar
            cm['fn_tick'] += fn_tick
            cm['fp_tick'] += fp_tick

            # OCR + oracle: extract numeric text and use GT tick labels
            del ocr_engine
            ocr_engine = NumericOCR(language=ocr_language, rec_model=ocr_rec_model)
            p_bars_yxv, p_bars_oracle = ocr_engine.infer_bar_values(
                p_bars, p_ticks, full_img, radius_thresh * 5., gt_ticks
            )
            hard_acc, relaxed_acc = evaluate_value_accuracy(
                gt_bars, p_bars_yxv, radius_thresh
            )
            split_stats['acc_hard_total'] += hard_acc
            split_stats['acc_relaxed_total'] += relaxed_acc

            oracle_hard, oracle_relaxed = evaluate_value_accuracy(
                gt_bars, p_bars_oracle, radius_thresh
            )
            split_stats['oracle_acc_hard_total'] += oracle_hard
            split_stats['oracle_acc_relaxed_total'] += oracle_relaxed

        eval_iter.close()

    report_stats(split_stats, category_stats)

    logger.info(f'Wrote eval metrics to {metrics_log_file}')
    logger.info(f'Wrote category breakdown to {cat_log_file}')
    logger.info(f'Saved confusion matrix plot to {cm_plot_file}')
    logger.info(f'Saved cls activation maps to {act_folder}')

if __name__ == '__main__':
    main({})
