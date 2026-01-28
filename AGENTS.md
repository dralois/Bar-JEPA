# BAR-IJEPA Project Documentation

This document provides a comprehensive guide to the BAR-IJEPA project for understanding and extracting data from bar charts using self-supervised learning and keypoint detection.

## Project Overview

BAR-IJEPA is a three-stage system for bar chart understanding:

1. **Data Generation**: Synthetic bar chart generation with annotations
2. **Fine-tuning**: Adapt I-JEPA model to chart domain  
3. **Decoder Training**: Train keypoint detector on finetuned features
4. **Evaluation**: Test performance on real-world charts

```
Data Generation → Fine-tuning → Decoder Training → Evaluation
```

## Training Pipeline

### 1. Data Generation

**Script**: `bar-gen/generator.py`

Generates synthetic bar charts with precise JSON annotations including:
- Bar positions and values
- Tick marks and labels  
- Coordinate system origin
- Chart metadata (title, legend, axes)

**Key Features**:
- Parallel generation using multiprocessing
- Schema validation for annotations
- Configurable chart styles and layouts
- Random distributions for realistic data

**Usage**:
```bash
python bar-gen/generator.py \
  --output_dir data/charts \
  --train_total 10000 \
  --test_total 2000 \
  --num_processes 8 \
  --clear_output \
  --validate
```

**Output Structure**:
```
data/charts/
├── train/
│   ├── images/
│   │   ├── train_0.png
│   │   ├── train_1.png
│   │   └── ...
│   └── annotations/
│       ├── train_0.json
│       ├── train_1.json
│       └── ...
└── test/
    ├── images/
    └── annotations/
```

### 2. Fine-tuning

**Script**: `bar-jepa/src/train_finetune.py`

Adapts pretrained I-JEPA model to bar chart domain using masked region prediction.

**Key Components**:
- **Model Architecture**: Encoder + Predictor + Target Encoder (EMA)
- **Training Objective**: Masked region prediction (Smooth L1 Loss)
- **Data Augmentation**: Random resized crops, color distortion, horizontal flip
- **Optimization**: AdamW with cosine learning rate scheduling
- **Regularization**: Weight decay with cosine schedule

**Training Process**:
```python
# 1. Load pretrained I-JEPA model
encoder, predictor = init_ijepa_model(device, model_name, ...)
target_encoder = copy.deepcopy(encoder)

# 2. Initialize data loader with masking
mask_collator = MBMaskCollator(...)
train_loader = make_charts(transform, ..., collator=mask_collator)

# 3. Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        # Generate context/prediction masks
        masks_enc, masks_pred = mask_collator(batch)
        
        # Target encoding (no gradients)
        with torch.no_grad():
            h = target_encoder(imgs, grids)
            h = apply_masks(h, masks_pred)
        
        # Context encoding + prediction
        z = encoder(imgs, grids, masks_enc)
        z = predictor(z, grids, masks_enc, masks_pred)
        
        # Loss calculation
        loss = F.smooth_l1_loss(z, h)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Momentum update
        m = next(momentum_scheduler)
        for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
            param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)
```

**Usage**:
```bash
python bar-jepa/main.py \
  --mode finetune \
  --fname bar-jepa/configs/jepa/in1k_vith14_ep300.yaml
```

**Key Parameters**:
- `batch_size`: 32-256 (depends on GPU memory)
- `crop_size`: 224 (standard for ViT models)
- `patch_size`: 16 (16x16 patches)
- `num_enc_masks`: 1 (context blocks per image)
- `num_pred_masks`: 2 (target blocks per image)
- `enc_mask_scale`: [0.2, 0.8] (context block size range)
- `pred_mask_scale`: [0.2, 0.8] (target block size range)

### 3. Decoder Training

**Script**: `bar-jepa/src/train_decoder.py`

Trains keypoint detector on finetuned encoder features.

**Key Components**:
- **Model Architecture**: Frozen Encoder + Trainable Decoder
- **Training Objective**: Multi-component loss for keypoint detection
- **Data Format**: Chart images with ground truth heatmaps
- **Loss Components**: Origin, classification, regression, keypoint matching, alignment

**Training Process**:
```python
# 1. Load finetuned encoder (frozen)
encoder, decoder = init_decoder_model(device, model_name, ...)
for p in encoder.parameters():
    p.requires_grad = False

# 2. Initialize data loader
train_loader, val_loader = make_charts(transform, ...)

# 3. Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        # Forward pass
        h = encoder(imgs, grids)              # Feature extraction
        p_cls, p_reg, p_kps = decoder(h, grids)  # Keypoint prediction
        
        # Loss calculation (multi-component)
        loss = origin_loss + classification_loss + regression_loss
               + keypoint_loss + alignment_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Validation
        # Compute precision/recall/F1 metrics
```

**Loss Components**:

1. **Origin Loss** (`l_org`):
   - Binary cross-entropy for origin probability
   - L1 loss for origin coordinate prediction
   - Entropy regularization for one-hot selection

2. **Classification Loss** (`l_cls`):
   - Weighted cross-entropy for bar/tick classification
   - Class weights: [background, bar, tick] = [1.0, 2.0, 2.0]

3. **Regression Loss** (`l_reg`):
   - MSE loss for coordinate offsets
   - Applied only to non-background pixels

4. **Keypoint Loss** (`l_pts`):
   - Distance-based matching between predictions and ground truth
   - Components: distance, missing, claim, background
   - Soft assignment using temperature parameter (τ=0.04)

5. **Alignment Loss** (`l_align`):
   - Encourages tick alignment along value axis
   - Penalizes variance in tick x-coordinates

**Usage**:
```bash
python bar-jepa/main.py \
  --mode decoder \
  --fname bar-jepa/configs/keypoint/classic_arp.yaml
```

### 4. Evaluation

**Script**: `bar-jepa/src/eval_decoder.py`

Evaluates trained decoder on real-world chart datasets.

**Key Metrics**:
- **Precision/Recall**: For bar and tick detection
- **F1 Score**: Harmonic mean of precision and recall
- **Coordinate Accuracy**: Distance-based metrics

**Evaluation Process**:
```python
# 1. Load trained model
encoder, decoder = init_decoder_model(device, model_name, ...)
load_decoder_checkpoint(..., encoder, decoder)

# 2. Load evaluation dataset
eval_loader = make_charts(transform, ...)

# 3. Evaluation loop
decoder.eval()
for batch in eval_loader:
    # Forward pass
    h = encoder(imgs, grids)
    p_cls, p_reg, p_kps = decoder(h, grids)
    
    # Extract predictions
    p_bars, p_ticks, p_org = p_maps_to_cls_lists(...)
    p_bars, p_ticks = nms(p_bars, p_ticks, radius_thresh)
    
    # Compute metrics
    bar_precision, bar_recall = evaluate_gt_p_match(gt_bars, p_bars)
    tick_precision, tick_recall = evaluate_gt_p_match(gt_ticks, p_ticks)
    
    # Log results
    wandb.log({
        'bar_precision': bar_precision,
        'bar_recall': bar_recall,
        'tick_precision': tick_precision,
        'tick_recall': tick_recall,
        'bar_f1': f1(bar_precision, bar_recall),
        'tick_f1': f1(tick_precision, tick_recall)
    })
```

**Usage**:
```bash
python bar-jepa/main.py \
  --mode eval \
  --fname bar-jepa/configs/eval/ubpmc.yaml
```

## Key Components

### 1. Charts Dataset

**File**: `bar-jepa/src/datasets/charts.py`

**`Charts` Class**: Custom dataset loader for chart images

**Features**:
- Loads images and JSON annotations
- Converts annotations to normalized coordinates
- Generates ground truth heatmaps (origin, class, regression)
- Supports train/validation splits
- Handles both synthetic and real-world data

**Data Processing**:
```python
def __getitem__(self, idx):
    # Load image
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    
    # Load annotations
    ann = json.load(open(ann_path))
    
    # Extract and normalize coordinates
    size = torch.tensor(ann['chart_metadata']['size']['bbox'][2:])
    org = (torch.tensor(ann['chart_metadata']['origin']['bbox'][:2]) / size).flip(-1)
    
    # Extract bars and ticks
    bars = []
    for feature in ann['data']['features']:
        for bar in feature['data']:
            bars.append((torch.tensor([bar['bbox'][2], bar['bbox'][1]]) / size).flip(-1))
    
    ticks = []
    for tick in ann['data']['value_axis']['ticks']:
        ticks.append((torch.tensor(tick['bbox'][:2]) / size).flip(-1))
    
    # Generate heatmaps
    mapsize = (torch.tensor(img.shape[1:3]) // patch_size) * 4
    gt_org, gt_cls, gt_reg = cls_pts_to_maps([bars, ticks], org, mapsize)
    
    return img, (gt_org, gt_cls, gt_reg)
```

### 2. Data Transforms

**File**: `bar-jepa/src/transforms.py`

**Key Transformations**:
- **`ResizeToFixedPatches`**: Resizes images to fixed patch grid while preserving aspect ratio
- **`GaussianBlur`**: Optional augmentation for fine-tuning
- **Normalization**: Standard ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Aspect Ratio Preservation**: Critical for chart understanding

**Transform Pipeline**:
```python
transform = transforms.Compose([
    ResizeToFixedPatches(max_patches=patch_count, patch_size=patch_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### 3. Heatmap Utilities

**File**: `bar-jepa/src/utils/heatmap.py`

**Key Functions**:
- **`cls_pts_to_maps`**: Converts annotated points to heatmaps
- **`gt_maps_to_cls_lists`**: Converts heatmaps back to point lists
- **`keypoint_sets`**: Matches predictions to ground truth using soft assignment
- **`p_maps_to_cls_lists`**: Extracts predictions from heatmaps
- **`nms`**: Non-maximum suppression for filtering predictions
- **`evaluate_gt_p_match`**: Computes precision/recall metrics
- **`f1`**: Computes F1 score from precision/recall

**Heatmap Generation**:
```python
def cls_pts_to_maps(cls_pts_lists, origin, mapsize):
    # Create empty maps
    cls_map = torch.zeros(size=(mapsize[0], mapsize[1]), dtype=torch.long)
    org_map = torch.zeros(size=(mapsize[0], mapsize[1]))
    reg_map = torch.zeros(size=(2, mapsize[0], mapsize[1]))
    
    # Fill class map (1=bar, 2=tick)
    for cls_id, cls_list in enumerate(cls_pts_lists):
        for point in cls_list:
            pos = torch.floor(point * mapsize).type(torch.int32)
            reg = (point * mapsize) - pos - 0.5
            cls_map[pos[0], pos[1]] = cls_id + 1
            reg_map[:, pos[0], pos[1]] = reg
    
    # Fill origin map
    org_pos = torch.floor(origin * mapsize).type(torch.int32)
    org_reg = (origin * mapsize) - org_pos - 0.5
    org_map[org_pos[0], org_pos[1]] = 1.0
    reg_map[:, org_pos[0], org_pos[1]] = org_reg
    
    return org_map, cls_map, reg_map
```

### 4. Keypoint Decoder

**File**: `bar-jepa/src/models/decoders.py`

**`KeypointDetector` Class**: Combined detection and classification head

**Architecture**:
```python
class KeypointDetector(nn.Module):
    def __init__(self, max_patches, in_channels, num_keypoints, num_classes):
        # Feature decoder (Simple or Classic)
        self.decoder = SimpleDecoder(in_channels, num_keypoints)
        
        # Prediction heads
        self.fc_cls = nn.Conv2d(num_keypoints, num_classes, 1, 1, 0)  # Class probabilities
        self.fc_org = nn.Conv2d(num_keypoints, 1, 1, 1, 0)           # Origin probability
        self.fc_reg = nn.Conv2d(num_keypoints, 2, 1, 1, 0)           # (dx, dy) offsets
        
        # Dropout for regularization
        self.drop_layer = nn.Dropout(p=0.5)

    def forward(self, x, grids):
        # Feature upsampling
        x = self.decoder(x)  # [1, C_in, H, W] -> [C_out, H*4, W*4]
        x = self.drop_layer(x)
        
        # Prediction heads
        cls_logits = self.fc_cls(x)  # Class probabilities
        org_logits = self.fc_org(x)  # Origin probability
        reg = self.fc_reg(x)         # Coordinate offsets
        
        # Keypoint extraction
        kp_preds = self._predict_keypoint(x, cls_logits, org_logits)
        
        return cls_logits, reg, kp_preds
```

**Keypoint Extraction**:
```python
def _predict_keypoint(self, kp_logits, cls_logits, org_logits):
    # Spatial softmax
    _, H, W = kp_logits.shape
    ys, xs = torch.meshgrid(
        (torch.arange(H) + 0.5) / H,
        (torch.arange(W) + 0.5) / W,
        indexing="ij"
    )
    coord_grid = torch.stack([ys, xs], dim=-1).view(-1, 2)
    kp_logits = kp_logits.view(self.num_keypoints, -1)
    
    # Softmax over spatial dimensions
    weights = torch.softmax(kp_logits, dim=1)
    
    # Predict coordinates
    kp_coords = weights @ coord_grid
    
    # Predict class probabilities
    cls_logits = cls_logits.view(self.num_classes, -1).T
    kp_cls_logits = weights @ cls_logits
    
    # Predict origin probabilities
    org_logits = org_logits.view(1, -1).T
    kp_org_logits = weights @ org_logits
    
    return torch.cat([kp_coords, kp_cls_logits, kp_org_logits], dim=1)
```

## Configuration System

### Configuration Files

- `bar-jepa/configs/jepa/`: Fine-tuning configurations
- `bar-jepa/configs/keypoint/`: Decoder training configurations  
- `bar-jepa/configs/eval/`: Evaluation configurations

### Configuration Structure

**Fine-tuning Configuration** (`bar-jepa/configs/jepa/in1k_vith14_ep300.yaml`):
```yaml
meta:
  use_bfloat16: true
  model_name: "vit_base"
  load_checkpoint: true
  do_finetune: true
  checkpoint_epoch: 100
  pred_depth: 6
  pred_emb_dim: 384

data:
  batch_size: 64
  crop_size: 224
  patch_size: 16
  preserve_aspect_ratio: true
  use_gaussian_blur: true
  use_horizontal_flip: true
  use_color_distortion: true
  color_jitter_strength: 1.0
  root_path: "data/"
  image_folder: "images/"
  annotation_folder: "annotations/"

mask:
  allow_overlap: false
  num_enc_masks: 1
  num_pred_masks: 2
  enc_mask_scale: [0.2, 0.8]
  pred_mask_scale: [0.2, 0.8]
  aspect_ratio: [0.3, 3.0]
  min_keep: 4

optimization:
  epochs: 100
  lr: 1.5e-4
  weight_decay: 0.05
  final_weight_decay: 0.05
  warmup: 0.1
  start_lr: 1e-6
  final_lr: 0.0
  ema: [0.996, 1.0]
  ipe_scale: 1.25

logging:
  folder: "output/"
  write_tag: "bar-finetune"
```

**Decoder Training Configuration** (`bar-jepa/configs/keypoint/classic_arp.yaml`):
```yaml
meta:
  use_bfloat16: true
  model_name: "vit_base"
  decoder_type: "simple"
  load_checkpoint: true
  do_finetune: false
  finetune_epoch: 100

data:
  batch_size: 32
  crop_size: 224
  patch_size: 16
  preserve_aspect_ratio: true
  root_path: "data/"
  image_folder: "images/"
  annotation_folder: "annotations/"

keypoint:
  max_keypoints: 64
  pnt_detect_thresh: 0.5
  cls_conf_thresh: 0.5
  eval_thresh: 0.1
  class_weights: [1.0, 2.0, 2.0]  # [background, bar, tick]
  use_pts_loss: true
  use_align_loss: true

optimization:
  epochs: 50
  lr: 1e-4
  weight_decay: 0.05
  final_weight_decay: 0.05
  warmup: 0.1
  start_lr: 1e-6
  final_lr: 0.0
  ipe_scale: 1.0

logging:
  folder: "output/"
  write_tag: "bar-detector"
```

## Development Guidelines

### Coding Style

Follow existing patterns in `bar-jepa/src/utils/heatmap.py`:

```python
from typing import List, Tuple, Optional, Dict, Any
import torch

def function_name(
    param1: torch.Tensor,
    param2: List[torch.Tensor],
    threshold: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Clear description of function purpose.
    
    :param param1: Input tensor of shape [B, C, H, W]
    :param param2: List of tensors
    :param threshold: Confidence threshold (default: 0.5)
    :return: Tuple containing (result1, result2)
    """
    # Implementation with proper error handling
    if param1.dim() != 3:
        raise ValueError(f"Expected 3D tensor, got {param1.dim()}D")
    
    # Main logic
    result1 = process_tensor(param1)
    result2 = process_list(param2, threshold)
    
    return result1, result2
```

### Pixi Environment Management

The project uses **pixi** for dependency management and environment setup:

**Key Commands**:
```bash
# Add new dependencies
pixi add package_name
pixi add package_name@version
pixi add -D dev_package  # Development dependency

# Run commands in pixi environment
pixi run python script.py
pixi run python bar-jepa/main.py --mode decoder --fname bar-jepa/configs/config.yaml

# Activate pixi shell
pixi shell

# Update dependencies
pixi update

# List installed packages
pixi list
```

**Running Training Scripts**:
```bash
# Run fine-tuning
pixi run python bar-jepa/main.py --mode finetune --fname bar-jepa/configs/jepa/config.yaml

# Run decoder training
pixi run python bar-jepa/main.py --mode decoder --fname bar-jepa/configs/keypoint/config.yaml

# Run evaluation
pixi run python bar-jepa/main.py --mode eval --fname bar-jepa/configs/eval/config.yaml
```

### Modification Workflow

**Recommended workflow for making changes**:

1. **Create feature branch**:
```bash
git checkout -b feature/your-feature-name
```

2. **Add dependencies** (if needed):
```bash
pixi add new_package
```

3. **Implement changes**: Follow coding conventions and add appropriate documentation

4. **Test changes**:
```bash
pixi run python bar-jepa/main.py --mode decoder --fname bar-jepa/configs/test_config.yaml
```

5. **Commit changes** with clear messages:
```bash
git add .
git commit -m "feat: Add new feature"
git commit -m "fix: Fix bug in module"
git commit -m "docs: Update documentation"
```

6. **Create pull request**: Push to remote and create PR for review

## Key Innovations

### Multi-Component Loss Function

Decoder training combines five loss components:

1. **Origin Loss** (`l_org`):
   - Binary cross-entropy for origin probability
   - L1 loss for origin coordinate prediction
   - Entropy regularization for one-hot selection
   - Encourages single, confident origin prediction

2. **Classification Loss** (`l_cls`):
   - Weighted cross-entropy for bar/tick classification
   - Class weights: [background, bar, tick] = [1.0, 2.0, 2.0]
   - Higher weights for bars/ticks to handle class imbalance

3. **Regression Loss** (`l_reg`):
   - MSE loss for coordinate offsets
   - Applied only to non-background pixels
   - Predicts sub-pixel coordinate refinements

4. **Keypoint Loss** (`l_pts`):
   - Distance-based matching between predictions and ground truth
   - Components: distance, missing, claim, background
   - Soft assignment using temperature parameter (τ=0.04)
   - Differentiable matching process

5. **Alignment Loss** (`l_align`):
   - Encourages tick alignment along value axis
   - Penalizes variance in tick x-coordinates
   - Helps maintain proper chart structure

### Aspect Ratio Preservation

Critical for chart understanding:

- **`ResizeToFixedPatches`** transform maintains chart proportions
- Handles variable input sizes gracefully
- Ensures spatial relationships are preserved
- Critical for accurate keypoint detection

### Specialized Data Generation

Synthetic data generator creates realistic bar charts:

- **Random Styles**: Various colors, fonts, layouts
- **Precise Annotations**: Bounding boxes for all elements
- **Configurable Distributions**: Uniform, gamma, gaussian, exponential
- **Schema Validation**: Ensures annotation consistency
- **Parallel Generation**: Efficient large-scale data creation

## Architecture Summary

BAR-IJEPA implements a sophisticated three-stage system for bar chart understanding:

1. **Synthetic Data Generation**: Creates realistic bar charts with precise annotations
   - Generates diverse chart styles and layouts
   - Produces accurate JSON annotations
   - Supports parallel generation for efficiency

2. **Domain-Specific Fine-tuning**: Adapts I-JEPA to chart images using self-supervised learning
   - Uses masked region prediction objective
   - Preserves spatial relationships critical for charts
   - Adapts to chart-specific features and patterns

3. **Keypoint Detection**: Trains specialized decoder for extracting chart data
   - Detects bars, ticks, and coordinate system origin
   - Uses multi-component loss for robust training
   - Achieves high precision in element localization

The system achieves state-of-the-art performance in chart understanding by combining self-supervised representation learning with specialized keypoint detection, enabling accurate extraction of chart elements and their numerical values.

## Quick Reference

### File Structure
```
bar-jepa/
├── configs/
│   ├── jepa/          # Fine-tuning configs
│   ├── keypoint/      # Decoder training configs
│   └── eval/          # Evaluation configs
├── src/
│   ├── datasets/      # Dataset loaders
│   ├── models/        # Model architectures
│   ├── utils/         # Utility functions
│   ├── train_finetune.py  # Fine-tuning script
│   ├── train_decoder.py   # Decoder training script
│   └── eval_decoder.py    # Evaluation script
├── main.py           # Main entry point
└── README.md         # Project overview

bar-gen/
├── generator.py     # Data generation script
└── format.json       # Annotation schema
```

### Common Issues and Solutions

**CUDA Out of Memory**:
- Reduce `batch_size` in configuration
- Enable mixed precision: `use_bfloat16: true`
- Use smaller model: `model_name: "vit_small"`

**NaN Loss Values**:
- Reduce learning rate (try `lr: 1e-5`)
- Verify data normalization
- Check for invalid annotations

**Slow Training**:
- Use `pixi run` to ensure proper environment
- Set `num_workers` > 0 for data loading
- Reduce logging frequency

**Dependency Issues**:
- Run `pixi update` to resolve conflicts
- Check `pixi lock` file for version conflicts
- Create clean environment: `pixi shell --clean`

### Performance Optimization

**Memory Optimization**:
- Mixed precision training (bfloat16/float16)
- Gradient checkpointing
- Efficient attention masking
- Smaller batch sizes

**Training Efficiency**:
- Dynamic masking avoids redundant computations
- Position embedding interpolation handles variable sizes
- EMA target encoder provides stable training targets
- Cosine learning rate schedules for optimal convergence
