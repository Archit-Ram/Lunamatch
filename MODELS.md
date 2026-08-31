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

## 2. LoFTR: Detector-Free Local Feature Matching (Assessment & Status)

### 2.1 Overview & Architecture
- **LoFTR**: Detector-Free Local Feature Matching with Transformers
  - *Paper*: Sun, Shen, Yuan, Zhou, Bao, Zhou (CVPR 2021), *"LoFTR: Detector-Free Local Feature Matching with Transformers"*
  - *Architecture*: Dense FPN feature extractor with linear Transformer (coarse $1/8$ cross-attention) + dual-softmax matching + fine-level correlation refinement ($1/2$ resolution).

### 2.2 Technical Assessment: In-Browser Feasibility & Status
- **Current Status**: **Simulated Profile Matcher** (`SimulatedLoFTRProfileMatcher`), clearly labeled in UI.
- **Reason Real In-Browser Inference is Deferred**:
  1. **Weight Distribution & Licensing**: Official pretrained checkpoints (`outdoor_ds.ckpt`, ~128MB to 250MB) are released as raw PyTorch `.ckpt` files under Apache 2.0, requiring custom PyTorch-dependent conversion scripts (`loftr2onnx`). Unofficial ONNX hosts on Hugging Face require authentication tokens (HTTP 401) and lack audited provenance.
  2. **Memory & Compute Footprint**: LoFTR's coarse-to-fine full self-attention and cross-attention matrices require $O(N^2)$ quadratic memory for image tokens ($\approx 2400 \times 2400$ attention grids at $480 \times 480$ input), requiring over 1.2 GB WASM heap per forward pass. In browser WebAssembly, this frequently causes Out-Of-Memory (OOM) allocation crashes on client machines.
  3. **Fixed Resolution Grid Constraint**: Standard LoFTR ONNX exports hardcode a fixed spatial grid (typically $640 \times 480$ or $480 \times 480$) and cannot handle arbitrary crop sizes from ISRO lunar swaths (OHRC / TMC-2) without significant distortion or resampling artifacts.
- **Path to Real Inference**:
  - Recommended deployment architecture for LoFTR is a dedicated server-side Python / PyTorch microservice with GPU acceleration (CUDA / TensorRT) or an optimized distilled Edge variant (e.g. EfficientLoFTR / Mobile-LoFTR) exported with dynamic axes.
