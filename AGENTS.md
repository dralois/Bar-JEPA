# BAR-IJEPA Project Guide (Concise)

BAR-IJEPA is a four-stage system for bar chart understanding:
1) Data generation (synthetic charts + annotations)
2) I-JEPA fine-tuning (self-supervised)
3) Decoder training (keypoint detection)
4) Evaluation

Notes:
- Chart pipeline is the only active pipeline; legacy pretraining code path was removed.
- Fine-tuning ignores annotations (self-supervised).
- Evaluation supports both UBPMC and generated charts.
- Variable-resolution inputs are supported; batching happens after mask packing.
- Decoder forward groups same-size grids for batched deconv (no padding/chunking).
- Decoder NMS uses equal L1 distance for both bars and ticks (no anisotropic bar radius).
- Evaluation utilities are split into `src/utils/postprocessing.py` (`nms`, Hungarian matching, P/R/F1, value accuracy).

## Key scripts
- Data generation: `bar-gen/generator.py`
- Fine-tuning: `bar-jepa/src/train_finetune.py`
- Decoder training: `bar-jepa/src/train_decoder.py`
- Evaluation: `bar-jepa/src/eval_decoder.py`
- Entry point / modes: `bar-jepa/main.py` (`finetune`, `decoder`, `eval`)

## Variable resolution + masking (current behavior)
- Images remain as lists until patchification.
- Patch grids are per-image: `(H // patch_size, W // patch_size)`.
- Mask collator outputs boolean masks as `[B x nmasks]`, each mask length = `max_patches` (padding is masked).
- `pack_by_masks` compacts tokens and returns an attention mask (True = valid tokens).
- Encoder/predictor use attention masks to ignore padding + masked tokens.
- Keypoint decoder groups indices by `(H, W)` and runs deconvs batched per group; outputs are scattered back to per-image lists.

## Config notes
- Warmup is in epochs: `warmup_steps = warmup * iterations_per_epoch`.
- Decoder fine-tuning configs live in `bar-jepa/configs/keypoint/finetune/` and target UBPMC (`root_path: ./UBPMC`, `is_ubpmc: true`, `do_finetune: true`, `finetune_epoch: 50`).
- Decoder fine-tuning presets available: `classic_arp`, `classic_noarp`, `classic_vanilla`, `simple_arp`.
- ARP configs should only differ by `preserve_aspect_ratio`.
- Eval configs use `pnt_detect_thresh`, `cls_conf_thresh`, `eval_thresh`, and `score_norm` plus optional OCR settings under `ocr`.

## Evaluation (current behavior)
- Loads decoder checkpoints from `meta.read_checkpoint`.
- Runs sample-wise eval (`batch_size=1`) and computes bars/ticks P/R/F1 via Hungarian matching.
- Uses a normalized match/NMS radius: `eval_thresh / max(H, W)`.
- Supports optional OCR (`ocr.enabled`) via PaddleOCR:
  extracts numeric text -> matches predicted ticks -> infers bar values.
- OCR functions now operate with tensor outputs (no dict/list payloads in eval path).
- Value accuracy uses relative error on matched bars:
  `abs(gt - pred) / abs(gt) < eps` with hard `eps=0.02`, relaxed `eps=0.05`.
- Writes one summary row per run to `logging.folder/<write_tag>.csv` with:
  `samples`, `value_eval_count`, `value_hard_correct`, `value_relaxed_correct`,
  `value_acc_hard`, `value_acc_relaxed`, and bar/tick P/R/F1.

## Transforms / dataset
- `ResizeToFixedPatches` preserves aspect ratio; PIL returns `(width, height)`.
- `Charts` dataset expects matching image + JSON pairs unless `annotation_folder` == `image_folder`.
- Datasets use a unified `eval_mode` flag for evaluation output shape.
- In `eval_mode`, dataset targets are `(gt_org, gt_bars, gt_ticks, gt_bar_values)` and collated via `EvalCollator`.
- UBPMC specifics:
  origin is resolved from y-axis ticks using task labels (`0` tick if present, else minimum numeric tick),
  and GT bar values are taken from task6 `data series` (aligned with task6 visual bars).

## Coding style
- Use Sphinx notypes docstrings (see `bar-jepa/src/utils/heatmap.py` for style).

## Pixi quick usage
```
pixi run python bar-jepa/main.py --mode finetune --fname bar-jepa/configs/...
pixi run python bar-jepa/main.py --mode decoder --fname bar-jepa/configs/...
pixi run python bar-jepa/main.py --mode eval --fname bar-jepa/configs/...
```

## Submit helper
- Use `scripts/submit_viscom.sh` for cluster submits.
- Default GPU spec is `6000:1`.
- Supported modes: `decoder`, `decoder-finetune`, `finetune`.
- `decoder-finetune` uses configs from `bar-jepa/configs/keypoint/finetune/` and runs `main.py --mode decoder`.

## Common issues (short)
- OOM: lower `batch_size`, enable `use_bfloat16`.
- NaNs: lower LR, verify normalization and annotations.
