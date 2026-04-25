# CLAUDE.md — image_vision

## Project identity

`image_vision` (formerly `image_detection`) is a thin, opinionated wrapper around the Elixir ML ecosystem (Bumblebee, Ortex, Nx) that sits next to the [`image`](https://hex.pm/packages/image) library. It exposes three vision tasks — **classification**, **segmentation**, and **object detection** — through a small, friendly API.

## Audience

The target user is a working Elixir developer who:

* Wants to know "is this a cat?" or "how many people are in this photo?" or "give me the alpha mask for the foreground object".

* Is **not** an ML researcher. They should not need to know what a featurizer is, what a serving is, what shape a tensor must be, or which model variant to pick.

* Already uses [`image`](https://hex.pm/packages/image) (`Vix.Vips.Image`) for their image handling and expects this library to feel like a natural extension of it.

## Design priorities (in order)

1. **Developer ease of use.** The default call path is one function with one argument: `Image.Classification.labels(image)`, `Image.Segmentation.segment(image)`, `Image.Detection.detect(image)`. The user should never *have to* pick a model, configure a backend, manage a server, or download weights manually. Sensible defaults handle every step.

2. **Strong, opinionated defaults.** Each task has exactly one default model that is good enough for ~90% of use cases. Defaults are chosen for: permissive licensing (Apache 2.0 / MIT only), reasonable size (<500 MB), broad applicability, and proven quality. Power users can override every default through options, but they should not have to.

3. **Great documentation.** Every public function has the standard `### Arguments / ### Options / ### Returns / ### Examples` template (see `~/.claude/CLAUDE.md`). Module docs explain the *task* in plain language before the API. Doctest examples use real images from `test/support/images/`. Guides exist for the three common workflows (classify, segment, detect) and walk through what's happening at each step.

4. **Tight interop with `image`.** All public functions accept `t:Vix.Vips.Image.t/0` directly and return masks/overlays as `Vix.Vips.Image.t` where appropriate. Tensor conversion happens internally via `Image.to_nx/2` and `Image.from_nx/1`. The user should never need to touch Nx directly to use this library.

5. **Small install footprint.** Avoid eVision (huge — pulls in OpenCV). Use Ortex (~10–20 MB ORT shared library) for ONNX inference. ML deps are `optional: true` so `mix.exs` users opt into the runtime they want.

## What this library is *not*

* Not a research toolkit. If a user wants to swap encoder backbones or fine-tune a head, they should drop down to Bumblebee / Ortex directly.

* Not a generation library. Image generation (Stable Diffusion, etc.) lives elsewhere.

* Not a model zoo. We pick one default per task and document one or two upgrade paths. We do not enumerate every checkpoint on HuggingFace.

## Default models (April 2026)

| Task | Default | License | Runtime |
|---|---|---|---|
| Classification | `facebook/convnext-tiny-224` | Apache 2.0 | Bumblebee |
| Embedding | `facebook/dinov2-base` | Apache 2.0 | Bumblebee |
| Promptable segmentation | `facebook/sam2.1-hiera-tiny` (ONNX) | Apache 2.0 | Ortex |
| Semantic / instance segmentation | `facebook/mask2former-swin-tiny-coco-instance` (ONNX) | MIT | Ortex |
| Object detection | `PekingU/rtdetr_v2_r18vd` (ONNX) | Apache 2.0 | Ortex |

All defaults are permissive-licensed. We never default to AGPL/GPL (e.g. YOLOv8/11) or non-commercial (e.g. SegFormer's NVIDIA SSL, ConvNeXt-V2 Meta weights, DINOv3) models, even when they're more accurate.

## Model weight management

* ONNX weights for Ortex-backed tasks are auto-downloaded on first use from the configured HuggingFace repo, and cached on disk.

* Cache directory is configurable: `config :image_vision, :cache_dir, "/path/to/cache"`. Default falls back to `:filename.basedir(:user_cache, "image_vision")` (XDG-compliant per-user cache).

* Bumblebee-backed tasks defer to Bumblebee's own HF cache (controlled by `BUMBLEBEE_CACHE_DIR` and friends).

* Users who want fully offline operation can pre-download into the cache directory.

## Dependency posture

* `:image` is a hard dependency.

* `:bumblebee`, `:nx`, `:nx_image` are `optional: true` — required only for classification/embedding.

* `:ortex` is `optional: true` — required only for segmentation/detection.

* `:req` is required — used for ONNX weight downloads. Small, ~no transitive cost.

* No `:axon_onnx` (dormant, doesn't handle modern transformer vision graphs).

* No `:evision` (huge OpenCV footprint).

## Code style notes specific to this project

* Module structure follows the existing pattern: each task is its own top-level `Image.*` module (`Image.Classification`, `Image.Segmentation`, `Image.Detection`) so the user sees them as natural neighbours of `Image.Text`, `Image.Shape`, etc. from the `image` library.

* Compile-time gating (`if ImageVision.bumblebee_configured?(), do: defmodule ...`) lets the package compile cleanly when optional deps are absent.

* Use `Nx.select/3` (not `Nx.map/2`) for elementwise tensor operations — `Nx.map` does per-element backend transfers and is unusably slow.

* Always normalise via `NxImage.normalize/3` for ImageNet preprocessing. Don't hand-roll.

* When a serving is required (Bumblebee path), prefer the supervised-process pattern already in `lib/classification.ex`: child spec via `classifier/1`, optional autostart from app config.
