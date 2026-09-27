<p align="center">
  <img src="https://img.shields.io/badge/SIH_2026-SIH26166-blue?style=for-the-badge" alt="SIH 2026"/>
  <img src="https://img.shields.io/badge/ISRO-Chandrayaan--2-orange?style=for-the-badge" alt="ISRO"/>
  <img src="https://img.shields.io/badge/TypeScript-React-3178C6?style=for-the-badge&logo=typescript" alt="TypeScript"/>
  <img src="https://img.shields.io/badge/ONNX_Runtime-Web-green?style=for-the-badge" alt="ONNX"/>
</p>

# 🌙 LunaMatch — Multi-Modal Lunar Image Correspondence Engine

> **SIH 2026 Problem Statement SIH26166**  
> Multi-Modal, Sun Angle & Scale Invariant Lunar Image Correspondence Engine for ISRO Chandrayaan-2 Optical Payloads (OHRC • TMC-2 • IIRS)

LunaMatch is a **fully in-browser** image registration pipeline that aligns lunar satellite imagery from different sensors, viewing angles, and illumination conditions — running **real neural network inference** directly in your web browser via WebAssembly.

---

## 🎯 Key Features

| Feature | Description |
|---|---|
| **3 Real Matching Algorithms** | RIFT (analytical), SuperPoint+LightGlue (ONNX), EfficientLoFTR (ONNX) |
| **Multi-Expert Fusion** | Combines all 3 matchers with RANSAC outlier rejection |
| **TPS Registration** | Thin Plate Spline non-rigid warping for geometric alignment |
| **Illumination Invariance** | Phase congruency-based normalization handles extreme shadow reversal |
| **Real Satellite Data** | Tested on actual Chandrayaan-2 TMC-2 stereo pairs from ISSDC |
| **100% Browser-Based** | All ML inference runs in-browser via `onnxruntime-web` WASM — no server needed |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    LunaMatch Pipeline                    │
├──────────┬──────────────┬──────────────┬────────────────┤
│  RIFT    │  SuperPoint  │  LoFTR       │   Fusion &     │
│  (Phase  │  + LightGlue │  (Dense      │   RANSAC       │
│  Congru- │  (Sparse ONNX│  Transformer │   Outlier      │
│  ency)   │  Inference)  │  ONNX)       │   Rejection    │
├──────────┴──────────────┴──────────────┴────────────────┤
│           Illumination Invariant Normalizer              │
├─────────────────────────────────────────────────────────┤
│      Adaptive Transform Estimator (Rigid → TPS)         │
├─────────────────────────────────────────────────────────┤
│              TPS Warp & Image Registration               │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- **Node.js** ≥ 18
- ~200 MB disk space for ONNX models

### Installation

```bash
# Clone the repository
git clone https://github.com/chakshu-bot/Lunamatch.git
cd Lunamatch

# Install dependencies
npm install

# Download ONNX models (required for neural matchers)
mkdir -p public/models

# SuperPoint + LightGlue (~30 MB total)
curl -L -o public/models/superpoint.onnx \
  https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v0.1.0/superpoint.onnx

curl -L -o public/models/superpoint_lightglue.onnx \
  https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v0.1.3/superpoint_lightglue.onnx

# EfficientLoFTR (~122 MB)
curl -L -o public/models/eloftr_outdoor_opt.onnx \
  https://huggingface.co/SpatialHub/efficient-loftr-onnx/resolve/main/eloftr_outdoor_opt.onnx

# Start development server
npm run dev
```

Open **http://localhost:5173** in Chrome.

---

## 🧪 Testing

```bash
# Type check
npx tsc --noEmit

# Run unit tests (15/15 should pass)
npx tsx -e 'import { runAllLunaMatchUnitTests } from "./src/tests/unit_tests"; runAllLunaMatchUnitTests().then(res => { let p=0; res.forEach(r => { if(r.passed){p++}else{console.error("FAIL:", r.partName, r.testName)} }); console.log("Passed:", p, "/", res.length) })'

# Production build
npx vite build
```

---

## 📡 Real Satellite Data

LunaMatch includes **geometry-aligned stereo pairs** from real Chandrayaan-2 TMC-2 data:

| Preset | Latitude | Region |
|---|---|---|
| CH2 North Overlap | ≈ -34° | Geometry-aligned Aft/Nadir |
| CH2 Center Overlap | ≈ -45° | Bright escarpment region |
| CH2 South Overlap | ≈ -55° | Cratered highlands |

These were extracted from ISSDC PDS4 archives using the geometry CSV files to ensure the Aft (-25°) and Nadir (0°) crops show the **exact same terrain**.

### Downloading Your Own Data

1. Visit [ISSDC Data Products Search](https://chmapbrowse.issdc.gov.in/)
2. Select **TMC 2** or **OHRC** instrument
3. Select **Calibrated** product type
4. Enter your area of interest coordinates
5. Download NCA (Aft) and NCN (Nadir) pairs with matching timestamps

---

## 🔬 The Three Matching Algorithms

### 1. RIFT — Rotation Invariant Feature Transform
- **Type**: Analytical (no neural network)
- **Method**: Phase Congruency + Maximum Index Map (MIM)
- **Strength**: Illumination-invariant, works on extreme shadow reversal
- **Paper**: Li et al., "RIFT: Multi-Modal Image Matching Based on Radiation-Variation Insensitive Feature Transform" (IEEE TGRS 2020)

### 2. SuperPoint + LightGlue — Sparse Neural Matching
- **Type**: Real ONNX inference via `onnxruntime-web`
- **Method**: SuperPoint keypoint detection → LightGlue transformer matching
- **Strength**: Fast, accurate on textured regions with moderate viewpoint change
- **Papers**: DeTone et al. (CVPRW 2018), Lindenberger et al. (ICCV 2023)

### 3. EfficientLoFTR — Dense Transformer Matching
- **Type**: Real ONNX inference via `onnxruntime-web`
- **Method**: Dense FPN + cross-attention transformer with coarse-to-fine matching
- **Strength**: Dense correspondences, handles large viewpoint/illumination changes
- **OOM Mitigation**: Strict 480×480 downsampling caps attention matrix at ~50MB
- **Paper**: Sun et al. (CVPR 2021)

---

## 📁 Project Structure

```
src/
├── matching/
│   ├── rift_matcher.ts         # RIFT analytical matcher
│   ├── lightglue_matcher.ts    # SuperPoint + LightGlue ONNX
│   ├── loftr_matcher.ts        # EfficientLoFTR ONNX
│   └── fusion.ts               # Multi-expert fusion + RANSAC
├── registration/
│   ├── adaptive_transform.ts   # Rigid → Affine → TPS model selection
│   └── warp.ts                 # TPS image warping
├── illumination/
│   └── invariance.ts           # Phase congruency normalization
├── pipeline/
│   └── lunamatch.ts            # Main orchestration pipeline
├── generator/
│   └── synthetic.ts            # Synthetic lunar dataset generator
├── components/                 # React UI components
├── tests/
│   └── unit_tests.ts           # 15 unit tests
└── types/
    └── index.ts                # TypeScript type definitions

public/
├── models/                     # ONNX model files (gitignored)
│   ├── superpoint.onnx
│   ├── superpoint_lightglue.onnx
│   └── eloftr_outdoor_opt.onnx
└── test_images/                # Real Chandrayaan-2 test crops
    ├── ch2_tmc_nca_overlap_*.png  # TMC-2 Aft (-25°) views
    └── ch2_tmc_ncn_overlap_*.png  # TMC-2 Nadir (0°) views

scripts/
├── export_loftr_onnx.py        # LoFTR ONNX export (requires PyTorch)
└── prepare_ch2_images.py       # Chandrayaan-2 image preparation
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 19 + TypeScript |
| Build | Vite 6 |
| ML Runtime | `onnxruntime-web` (WASM backend) |
| Styling | Tailwind CSS |
| Charts | Lucide React icons |

---

## 📊 Performance

| Matcher | Typical Latency | Match Count | Memory |
|---|---|---|---|
| RIFT | ~500 ms | 10-50 | ~20 MB |
| LightGlue | ~2-3 s (first run 10s) | 20-100 | ~60 MB |
| LoFTR | ~5-10 s (first run 30s) | 100-500+ | ~220 MB |
| Fusion | ~10-15 s | Combined | ~280 MB |

*First run includes model download + IndexedDB caching. Subsequent runs use cached models.*

---

## 📜 License

Apache License 2.0

---

## 🙏 Acknowledgments

- **ISRO/ISSDC** — Chandrayaan-2 TMC-2 data via [ISSDC PDS4 Archive](https://chmapbrowse.issdc.gov.in/)
- **Kornia** — LoFTR pretrained weights (Apache 2.0)
- **fabio-sim** — [LightGlue-ONNX](https://github.com/fabio-sim/LightGlue-ONNX) exports
- **SpatialHub** — [EfficientLoFTR ONNX](https://huggingface.co/SpatialHub/efficient-loftr-onnx)

---

<p align="center">
  Built with ❤️ for ISRO Chandrayaan-2 Mission • SIH 2026
</p>
