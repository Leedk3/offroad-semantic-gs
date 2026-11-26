# Offroad Semantic GS

Source-linked, depth-guided semantic Gaussian mapping with explicit render support.

This repository contains the mapping core associated with **Depth-Guided Semantic Gaussian Mapping with Explicit Observation Support for Off-Road Terrain Understanding**. It accepts prepared, aligned observations and exposes the geometry, source selection and readout operations separately.

```
Aligned RGB + axial depth + terrain probabilities + camera poses
              |
       Immutable source archive
              |
    Requested sources / current poses
              |
     Planar Gaussian construction
              |
    Optional native CUDA rendering
              |
      Depth + terrain + support
```

The core includes frozen four-coefficient depth correction, camera-local geometry caching, source-image footprints, ordered source selection, and joint depth/semantic/support readout. It keeps the same requested Gaussian fields across eager/selective and cached/recomputed construction.

The input interface starts **after** depth estimation, terrain prediction and pose estimation. Raw-RGB model inference, training, visual tracking, ROS2 navigation, datasets and checkpoints are outside this source release. A CPU projection is not substituted for native Gaussian rendering.

## Quick start: CPU construction

Use Python 3.10 or newer in a separate environment. For a CPU-only installation:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install '.[test]'
python -m pytest -q -m 'not native'
offroad-gs demo --output outputs/demo
```

The procedural demo creates three source observations, exports a two-source Gaussian map, and checks equality across all four construction modes. It also writes a small **readout-arithmetic fixture**, which is not a render of that map. No model or dataset download is performed.

The CPU core was checked locally with Python 3.13, NumPy 2.4, SciPy 1.17 and PyTorch 2.10. The optional native renderer's research environment used Python 3.11, PyTorch 2.8 and CUDA 12.8. These are distinct environments; cross-version bitwise equality is not asserted. The CI recipe uses Python 3.11 and CPU PyTorch 2.8.

## Prepared observation format

Each source is an NPZ loaded with `allow_pickle=False`:

| Field | Shape / type | Meaning |
|---|---|---|
| `source_id` | integer scalar | Stable, unique observation identity |
| `timestamp_ns` | integer scalar | Acquisition time; strictly increasing |
| `rgb` | `uint8[H,W,3]` | Rectified RGB |
| `depth` | floating `[H,W]` | Axial camera depth, not Euclidean ray length |
| `valid` | `bool[H,W]` | Depth validity |
| `K` | floating `[3,3]` | Intrinsics for this exact image grid |
| `probabilities` | floating `[H,W,C]` | Aligned terrain probabilities; sum to one on admitted pixels |

Use a common class order across sources. Camera axes are x right, y down, z forward. Poses map camera coordinates into one common map frame. Prepare rectification and probability-grid alignment before passing arrays to this package; coarse feature grids are not accepted as image-aligned predictions.

The manifest contains source paths relative to itself, current source poses in the same order, and one query camera:

```json
{
  "sources": ["source_000.npz", "source_001.npz"],
  "source_poses": [
    [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]],
    [[1,0,0,0.1], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
  ],
  "window": 5,
  "calibration": [0.0, 0.0, 0.0, 0.0],
  "class_names": ["soil", "vegetation", "rock"],
  "query": {
    "K": [[70.0, 0.0, 31.0], [0.0, 65.0, 23.0], [0.0, 0.0, 1.0]],
    "camera_to_world": [[1,0,0,0.1], [0,1,0,0], [0,0,1,0], [0,0,0,1]],
    "image_shape": [48, 64],
    "timestamp_ns": 2000000000
  }
}
```

`offroad-gs demo` produces a complete runnable manifest and its source files. All source timestamps must be no later than the query. Query intrinsics for the native renderer must have zero skew.

```bash
offroad-gs build --manifest outputs/demo/manifest.json --output outputs/map.npz
offroad-gs build --manifest outputs/demo/manifest.json --output outputs/eager.npz --eager --no-cache
```

Raw valid depths in `[1,25]` are admitted at stride 4 before correction. The frozen correction is `d * exp(a*log(d) + b*x_norm + c*y_norm + e)`, with pixel-center normalized coordinates. Coefficients default to identity; no fitted calibration is bundled. A corrected sampled-depth range retention below 95% is an explicit error. Units are inherited from the input; this package does not establish metric scale.

`map.npz` contains `xyz`, WXYZ `rotation`, two-axis `scales`, RGB, terrain probabilities, opacity, and `(source_id,y,x)` identities. The source footprint targets a linearized source-image covariance of `4I` while retaining the surface normal. Singular footprints are rejected. World geometry is recomputed under current poses; only immutable camera-local geometry is cached.

## Optional native Gaussian rendering

The native path uses **planar 2D Gaussian/surfel rasterization**, with the pinned revision and patches in `native_runtime.json`. Stock 3DGS or an unpatched rasterizer is not interchangeable with this path.

The upstream renderer and CUDA patch excerpts have a **separate research/noncommercial license**. Read `licenses/Gaussian-Splatting.txt` and `THIRD_PARTY_NOTICES.txt` before installing. This dependency is not covered by the project's MIT license. A CUDA development environment with `nvcc`, a C++ compiler, Git and CUDA-enabled PyTorch is required.

```bash
bash scripts/install_native.sh --accept-research-license
offroad-gs render --map outputs/map.npz --output outputs/render.npz
```

The installer clones a fixed renderer revision, pins GLM, applies two source patches, verifies source hashes and records the resulting extension hash. It does not overwrite an existing checkout. The adapter checks that marker before rendering. No CUDA binary is distributed. Building the external CUDA extension and rendering through the clean installation remain separate from the CPU quick-start checks.

The render packet contains RGB, alpha, conditional depth, semantic mass, normalized terrain probabilities, class indices and a support mask. Depth is normalized by alpha **once**. A supported pixel has alpha greater than 0.5 and finite positive conditional depth. Unsupported depth/probabilities are zero and the unknown class index is `C`. Support means a rendered prediction is available; it is not correctness confidence, free space, traversability, or a safety certificate.

## Python API

```python
from semantic_gs import Observation, SourceArchive, finalize_render

archive = SourceArchive(calibration=(0, 0, 0, 0))
# archive.add(Observation(...)) in acquisition order
# field = archive.construct(current_poses, source_ids=[...], use_cache=True)
# raw = semantic_gs.native.render_native(field, K, pose, (height, width))
# prediction = finalize_render(raw)
```

The archive copies observations. Its calibration is fixed. Requested sources must be an ordered subsequence of arrived observations; eager construction needs poses for the whole arrived prefix. Pose revisions are supplied at query time.

## Experiment records and scope

`benchmarks/` retains small numeric tables from the controlled construction and fusion studies. They contain no dataset imagery or sensor recordings. `benchmarks/protocol.json` defines the conditions and limits of these records. They are archived measurements, not benchmark results rerun by the synthetic demo.

The construction study reports component work under a shared-GPU workload and excludes neural inference, tracking and rendering. The fusion record includes the stronger single-frame control and does not establish an accuracy advantage for Gaussian accumulation. These records do not establish independent geographic generalization or navigation safety. This code release is not yet a complete raw-data reproduction of every manuscript experiment.

## Container

The CPU Docker recipe uses a smaller environment than the original research stack:

```bash
docker build -t offroad-semantic-gs .
docker run --rm -v "$PWD/outputs:/outputs" offroad-semantic-gs demo --output /outputs/demo
```

The Docker recipe does not install the optional CUDA renderer or ROS2. Its image build is separate from the local CPU tests.

## License and attribution

Original project code is MIT-licensed, with explicit exceptions for the CUDA patch excerpts and optional upstream renderer. See `LICENSE`, `THIRD_PARTY_NOTICES.txt`, and `licenses/`. Optional DA3/DINOv3 predictions must be obtained under their own code/model terms; no weights or automatic model downloads are included.

Use `CITATION.cff` to cite this software. The associated manuscript title is given above; no acceptance status or DOI is claimed.
