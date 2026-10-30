# Source and dependency notices

This project includes and modifies source from
[nigelhartman/splat_analyzer](https://github.com/nigelhartman/splat_analyzer)
at commit `a3cd884`. Its MIT license and copyright notice are preserved verbatim
in `LICENSE.md`.

Changes to that source include automatic label discovery, mask-based 3D instance
association, alternative names and parent/child relationships, output provenance,
clearance-first cameras, removal of the per-category instance cap, regression
checks, standalone setup scripts, and a generic local result viewer.

Fetched dependencies and model weights are not vendored into this source folder:

| Dependency | Source / version | License reference |
| --- | --- | --- |
| RAM++ inference source | [recognize-anything](https://github.com/xinyu1205/recognize-anything), `7cb804a8609e9f4b1a50b7f31436d2df40bb9481` | Apache 2.0; copy in `licenses/recognize-anything-LICENSE.txt` |
| Apple Metal renderer | [iffyloop/gsplat-mps](https://github.com/iffyloop/gsplat-mps), `10372c263f8e54427183de8a5855a62f94c2f206` | AGPL v3; copy in `licenses/gsplat-mps-LICENSE.txt` |
| CUDA renderer | [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat), `1.5.3` | Upstream project's license |
| RAM++ weights | [recognize-anything-plus-model](https://huggingface.co/xinyu1205/recognize-anything-plus-model), `84d4aee3a0265c4e0df1f714f0572011d1bf2ec3` | Model repository's license and model card |
| SAM weights | [facebook/sam-vit-base](https://huggingface.co/facebook/sam-vit-base), `d0d9250bd6a38ebbb901aa29e8d37787e366f7be` | Model repository's license and model card |
| Semantic text matching | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | Model repository’s Apache 2.0 license |
| Structured JSON generation | [lm-format-enforcer](https://github.com/noamgat/lm-format-enforcer), `0.11.3` | Upstream MIT license |
| Streaming camera/splat uploads | [python-multipart](https://github.com/Kludex/python-multipart), `0.0.32` | Upstream Apache 2.0 license |
| Qwen2.5-VL scene-understanding weights | [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), `66285546d2b821cf421d4f5eb2576359d3770cd3` | Model repository’s license and model card |
| OWLv2 weights | [google/owlv2-base-patch16-ensemble](https://huggingface.co/google/owlv2-base-patch16-ensemble), `cfd3195ba4ea9592eec887ded089f4c08eff231d` | Model repository's license and model card |
| Three.js viewer runtime | [Three.js](https://github.com/mrdoob/three.js), `0.179.0` | Upstream project's license |
| Spark viewer runtime | [Spark](https://github.com/sparkjsdev/spark), `2.0.0` | Upstream project's license |

Python dependency versions are recorded in the requirements files. Dependencies
and model weights retain their respective licenses; the top-level license does
not replace them.
