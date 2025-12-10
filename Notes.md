# Parameters
- [x] Exactly 5 bars
- [x] No legend
- [x] Grid / No grid
- [x] Bar colors / background
- [x] Bar width
- [x] Color consistancy within image
- [x] Axis information / Title
- [x] Various sizes / aspect ratio

# ARP notes
mask_keep = [8, 42, 1280] (B, min_keep, repeat index D times)
x = [8, 256, 1280] (B, N, D)
all_x = [4, 8, 42, 1280] (npred, B, min_keep, D)
returns [32, 42, 1280] (B * npred, min_keep, D) -> out[(i * 8) + j][v] == x[j][masks[i][j][v]], i < npred, j < B, v < min_keep

fwd_ctx:
	z = [8, 79, 1280] (B, min_keep_enc, D)
	z = [32, 42, 1280] (B * npred, min_keep, D)

--- 

x = [8, 119, 1280]
masks_x = [[8, 119]] (nenc * [B, min_keep])
masks = [[8, 42]] (npred * [B, min_keep])

x -> [8, 119, 384] (predictor encode)
x_pos_embed = [8, 256, 384] -> masked: [8, 119, 384]
! -> will be [8, 256, 384], masking instead of reduced

pos_embs = [8, 256, 384] -> masked: [32, 42, 384]
! -> will be [32, 256, 384], masking instead of reducing
! -> mask should be covering all but the patches from the encoder
! -> contain position embeddings for prediction tokens

pred_tokens = [32, 42, 384]
! -> will be [32, 256, 384], masking instead of reducing
! -> mask should be covering all but the prediction target patches
! -> preditiction token where target patches are, else 0

x -> [32, 119, 384] (repeat)
! -> will be [32, 256, 384] (repeated npred times)
x -> [32, 161, 384] (add pred tokens)
! -> will be [32, 512, 384] (x + tokens)

x -> [32, 42, 384] (only predictions)
! -> will be [32, 256, 384] (only tokens, 0 where masked?)

x -> [32, 42, 1280] (predictor decode)
! -> will be [32, 256, 1280]

[mem: 1.52e+04] (3141.7 ms)
[mem: 1.52e+04] (2952.1 ms)