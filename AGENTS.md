# BAR-IJEPA Project Guide (Concise)

BAR-IJEPA is a four-stage system for bar chart understanding:
1) Data generation (synthetic charts + annotations)
2) I-JEPA fine-tuning (self-supervised)
3) Decoder training (keypoint detection)
4) Evaluation (WIP)

Notes:
- Chart pipeline is the focus; pretraining is legacy.
- Fine-tuning ignores annotations (self-supervised).
- Evaluation is WIP; supported datasets are UBPMC + generated charts.
- Variable-resolution inputs are supported; batching happens after mask packing.
- Decoder forward groups same-size grids for batched deconv (no padding/chunking).

## Key scripts
- Data generation: `bar-gen/generator.py`
- Fine-tuning: `bar-jepa/src/train_finetune.py`
- Decoder training: `bar-jepa/src/train_decoder.py`
- Evaluation: `bar-jepa/src/eval_decoder.py`
- Legacy pretraining (not used): `bar-jepa/src/train.py`

## Variable resolution + masking (current behavior)
- Images remain as lists until patchification.
- Patch grids are per-image: `(H // patch_size, W // patch_size)`.
- Mask collator outputs boolean masks as `[B x nmasks]`, each mask length = `max_patches` (padding is masked).
- `pack_by_masks` compacts tokens and returns an attention mask (True = valid tokens).
- Encoder/predictor use attention masks to ignore padding + masked tokens.
- Keypoint decoder groups indices by `(H, W)` and runs deconvs batched per group; outputs are scattered back to per-image lists.

## Config notes
- Warmup is in epochs: `warmup_steps = warmup * iterations_per_epoch`.
- Finetuning configs currently target moderate adaptation (LR ~1e-5, WD ~0.05) from epoch-300+ checkpoints.
- ARP configs should only differ by `preserve_aspect_ratio`.

## Transforms / dataset
- `ResizeToFixedPatches` preserves aspect ratio; PIL returns `(width, height)`.
- `Charts` dataset expects matching image + JSON pairs unless `annotation_folder` == `image_folder`.

## Coding style
- Use Sphinx notypes docstrings (see `bar-jepa/src/utils/heatmap.py` for style).

## Pixi quick usage
```
pixi run python bar-jepa/main.py --mode finetune --fname bar-jepa/configs/...
pixi run python bar-jepa/main.py --mode decoder --fname bar-jepa/configs/...
pixi run python bar-jepa/main.py --mode eval --fname bar-jepa/configs/...
```

## Common issues (short)
- OOM: lower `batch_size`, enable `use_bfloat16`.
- NaNs: lower LR, verify normalization and annotations.
