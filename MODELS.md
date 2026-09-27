# LunaMatch Neural Models Specification & Provenance

This document details the neural weights, architectures, licenses, mathematical coordinate conventions, and preprocessing pipelines utilized by the correspondence matching engine.

---

## 1. SuperPoint + LightGlue Pipeline (Real Inference)

### 1.1 Overview & Architectures
- **SuperPoint**: Self-Supervised Interest Point Detection and Description
  - *Paper*: DeTone, Malisiewicz, Rabinovich (CVPRW 2018), *"SuperPoint: Self-Supervised Interest Point Detection and Description"*
  - *Architecture*: Fully convolutional encoder with shared representation yielding dense keypoint detector heatmap ($H/8 \times W/8 \times 65$) and L2-normalized 256-D descriptor maps.
- **LightGlue**: Local Feature Matching at Light Speed
  - *Paper*: Lindenberger, Sarlin, Pollefeys (ICCV 2023), *"LightGlue: Local Feature Matching at Light Speed"*
  - *Architecture*: Deep transformer graph neural network with positional relative encoding, cross-attention layers, and early-stopping matchability prediction.

### 1.2 Weights Source & Provenance
- **Export Repository**: [`fabio-sim/LightGlue-ONNX`](https://github.com/fabio-sim/LightGlue-ONNX) (Release `v0.1.3` / `v1.0.0`)
- **Direct Model Artifacts**:
  - `superpoint.onnx` (~5.1 MB): [https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v0.1.0/superpoint.onnx](https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v0.1.0/superpoint.onnx)
  - `superpoint_lightglue.onnx` (~25 MB): [https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v0.1.3/superpoint_lightglue.onnx](https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v0.1.3/superpoint_lightglue.onnx)
- **Licenses**:
  - LightGlue Architecture & Weights: **Apache License 2.0** (cvg/LightGlue, ETH Zurich)
  - SuperPoint Architecture / Weights Export: **Apache-2.0 / MIT** derivative export (`glue-factory` / `LightGlue-ONNX`)

### 1.3 Preprocessing & Coordinate Specifications
- **Input Tensor**:
  - Grayscale intensity $\in [0.0, 1.0]$, Float32.
  - Dimensions: `[1, 1, Height, Width]` (1 batch, 1 channel).
- **SuperPoint Output**:
  - `keypoints`: `[1, N, 2]` Int64 `(x, y)` coordinate pairs where $x = \text{column}, y = \text{row}$.
  - `scores`: `[1, N]` Float32 detection confidences.
  - `descriptors`: `[1, N, 256]` Float32 L2-normalized visual embeddings.
- **LightGlue Input Keypoint Normalization**:
  $$\text{shift} = \left[\frac{W}{2}, \frac{H}{2}\right], \quad \text{scale} = \frac{\max(W, H)}{2}$$
  $$\mathbf{k}_{\text{norm}} = \frac{\mathbf{k}_{\text{pixel}} - \text{shift}}{\text{scale}} \in [-1.0, 1.0]$$
- **LightGlue Output**:
  - `matches0`: `[1, N]` Int64 indexing matched target keypoints ($-1$ if unmatched).
  - `mscores0`: `[1, N]` Float32 match assignment likelihoods.

---

## 2. LoFTR: Detector-Free Local Feature Matching (Real Inference)

### 2.1 Overview & Architecture
- **LoFTR**: Detector-Free Local Feature Matching with Transformers
  - *Paper*: Sun, Shen, Yuan, Zhou, Bao, Zhou (CVPR 2021), *"LoFTR: Detector-Free Local Feature Matching with Transformers"*
  - *Architecture*: Dense FPN feature extractor with linear Transformer (coarse $1/8$ cross-attention) + dual-softmax matching + fine-level correlation refinement ($1/2$ resolution).

### 2.2 Current Status: Real ONNX Inference (`LoFTRMatcher`)
- **Status**: **Real Neural Inference** via `onnxruntime-web` WASM backend.
- **Model File**: `public/models/loftr_outdoor.onnx` — exported locally from Kornia's LoFTR with "outdoor" pretrained weights using `scripts/export_loftr_onnx.py`.
- **License**: Apache License 2.0 (original LoFTR weights and Kornia framework).

### 2.3 OOM Mitigation Strategy
- LoFTR's $O(N^2)$ quadratic attention memory is mitigated by **strict downsampling** of all input images to a fixed $480 \times 480$ grid before inference.
- At $480 \times 480$, the coarse attention grid is $60 \times 60 = 3600$ tokens, requiring ~$3600^2 \times 4 \approx 50$ MB for the attention matrix — well within the browser's WASM heap limit.
- Output keypoint coordinates are **scaled back** to the original image dimensions after inference.

### 2.4 Preprocessing & Coordinate Specifications
- **Input Tensor**:
  - Grayscale intensity $\in [0.0, 1.0]$, Float32.
  - Dimensions: `[1, 1, 480, 480]` (fixed resolution, bilinear-interpolated from original).
- **Output**:
  - `keypoints0`: `[N, 2]` Float32 matched coordinates in image0 (480×480 grid space).
  - `keypoints1`: `[N, 2]` Float32 matched coordinates in image1 (480×480 grid space).
  - `confidence`: `[N]` Float32 match confidence scores.
- **Coordinate Rescaling**: Output coordinates are multiplied by `(originalWidth / 480, originalHeight / 480)` to map back to pixel space.

### 2.5 Model Generation
```bash
pip install torch kornia onnx onnxruntime
python3 scripts/export_loftr_onnx.py
# → Creates public/models/loftr_outdoor.onnx (~45 MB)
```

