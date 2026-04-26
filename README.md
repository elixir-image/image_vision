# ImageVision

`ImageVision` is a simple, opinionated image vision library for Elixir. It sits alongside the [`image`](https://hex.pm/packages/image) library and answers common questions about an image — **what's in it**, **where are the objects**, **which pixels belong to which object**, **what's the foreground**, **describe it in words**, **does it match these labels** — with strong defaults and no ML expertise required.

## Quick start

```elixir
# Classification — what is in this image?
iex> puppy = Image.open!("puppy.jpg")
iex> Image.Classification.labels(puppy)
["Blenheim spaniel"]

# Detection — where are the objects and what are they?
iex> street = Image.open!("street.jpg")
iex> detections = Image.Detection.detect(street)
iex> hd(detections)
%{label: "person", score: 0.94, box: {120, 45, 60, 180}}

# Draw bounding boxes on the image
iex> Image.Detection.draw_bbox_with_labels(detections, street)

# Segmentation — which pixels belong to which object?
iex> segments = Image.Segmentation.segment_panoptic(street)
iex> Enum.map(segments, & &1.label)
["person", "car", "road", "sky"]

# Colour-coded overlay of all segments
iex> Image.Segmentation.compose_overlay(street, segments)

# Promptable segmentation — isolate the object at a specific point
iex> %{mask: mask} = Image.Segmentation.segment(puppy, prompt: {:point, 320, 240})
iex> {:ok, cutout} = Image.Segmentation.apply_mask(puppy, mask)

# Embedding — 768-dim feature vector for similarity search
iex> Image.Classification.embed(puppy)
#Nx.Tensor<f32[768]>

# Background removal — class-agnostic foreground cutout
iex> {:ok, cutout} = Image.Background.remove(puppy)

# Image captioning — natural-language description
iex> Image.Captioning.caption(puppy)
"a small brown and white puppy sitting on a wooden floor"

# Zero-shot classification — your labels, no retraining required
iex> Image.ZeroShot.classify(puppy, ["a dog", "a cat", "a horse"])
[%{label: "a dog", score: 0.998}, %{label: "a cat", score: 0.002}, ...]
```

## Installation

Add `:image_vision` to `mix.exs` along with whichever optional ML backends you need:

```elixir
def deps do
  [
    {:image_vision, "~> 0.2"},

    # Required for Image.Classification and Image.Classification.embed/2
    {:bumblebee, "~> 0.6"},
    {:nx, "~> 0.10"},
    {:exla, "~> 0.10"},     # or {:torchx, "~> 0.10"} for Torch backend

    # Required for Image.Detection and Image.Segmentation
    {:ortex, "~> 0.1"}
  ]
end
```

All ML deps are optional — omit any you do not use. The library compiles cleanly without them.

## Prerequisites

For the vast majority of users on Linux x86_64, macOS (Intel and Apple Silicon), and Windows x86_64, **no native toolchain is required**. The libraries used here ship precompiled native binaries for those platforms and `mix deps.get` is all you need.

If your platform isn't covered by precompiled binaries — uncommon Linux distros, ARM Linux, glibc mismatches — you'll need:

* **A Rust toolchain** for `:ortex` (the ONNX runtime wrapper used by detection and segmentation) and for `:tokenizers` (pulled in transitively by `:bumblebee`). Install via [rustup](https://rustup.rs):

  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```

* **A C compiler** for `:vix` (the libvips wrapper used by `:image`). On Linux install `build-essential` (Debian/Ubuntu) or `gcc` (Fedora/RHEL); on macOS install Xcode Command Line Tools (`xcode-select --install`).

* **`libvips`** if you need advanced libvips features beyond what the precompiled NIF includes. On macOS: `brew install vips`. On Linux: your distro's `libvips-dev` / `vips-devel` package. Most users don't need this.

If you see Cargo or `cc` errors during `mix deps.compile`, you've likely landed on a platform without precompiled coverage — install the toolchain above and re-run.

### Disk space and first-call latency

Model weights are downloaded on first call and cached on disk. Across all default models the total is approximately:

| Task | Default model | Size |
|---|---|---|
| Classification | `facebook/convnext-tiny-224` | ~110 MB |
| Embedding | `facebook/dinov2-base` | ~340 MB |
| Detection | `onnx-community/rtdetr_r50vd` | ~175 MB |
| Segmentation (SAM 2) | `SharpAI/sam2-hiera-tiny-onnx` | ~150 MB |
| Segmentation (panoptic) | `Xenova/detr-resnet-50-panoptic` | ~175 MB |
| Background removal | `onnx-community/BiRefNet_lite-ONNX` | ~210 MB |
| Captioning | `Salesforce/blip-image-captioning-base` | ~990 MB |
| Zero-shot classification | `openai/clip-vit-base-patch32` | ~605 MB |

The first call to each task therefore appears to "hang" while weights download — that's expected, not a bug.

To pre-download all default models before first use (recommended for production deployments and CI):

```bash
mix image_vision.download_models
```

Pass `--classify`, `--detect`, `--segment`, `--background`, `--caption`, or `--zero-shot` to limit scope.

### Livebook Desktop

Livebook Desktop launches as a GUI application and **does not inherit your shell's `PATH`**. Tools installed via `rustup`, `mise`, `asdf`, or Homebrew aren't visible to it by default — even if `cargo` works fine in your terminal.

If you hit "cargo: command not found" or similar during `Mix.install` inside Livebook Desktop, create `~/.livebookdesktop.sh` and add the relevant directories to `PATH`. A reasonable starting point:

```bash
# ~/.livebookdesktop.sh

# Rust (rustup)
export PATH="$HOME/.cargo/bin:$PATH"

# Homebrew (Apple Silicon)
export PATH="/opt/homebrew/bin:$PATH"

# mise — uncomment if you use it
# eval "$(mise activate bash)"

# asdf — uncomment if you use it
# . "$HOME/.asdf/asdf.sh"
```

Restart Livebook Desktop after creating this file. See the [Livebook Desktop documentation](https://github.com/livebook-dev/livebook/blob/main/README.md#livebook-desktop) for details.

## Default models

All models are permissively licensed. Weights are downloaded automatically on first call and cached on disk — no manual setup required.

| Task | Model | License | Size |
|---|---|---|---|
| Classification | [`facebook/convnext-tiny-224`](https://huggingface.co/facebook/convnext-tiny-224) | Apache 2.0 | ~110 MB |
| Embedding | [`facebook/dinov2-base`](https://huggingface.co/facebook/dinov2-base) | Apache 2.0 | ~340 MB |
| Object detection | [`onnx-community/rtdetr_r50vd`](https://huggingface.co/onnx-community/rtdetr_r50vd) | Apache 2.0 | ~175 MB |
| Promptable segmentation | [`SharpAI/sam2-hiera-tiny-onnx`](https://huggingface.co/SharpAI/sam2-hiera-tiny-onnx) | Apache 2.0 | ~150 MB |
| Panoptic segmentation | [`Xenova/detr-resnet-50-panoptic`](https://huggingface.co/Xenova/detr-resnet-50-panoptic) | Apache 2.0 | ~175 MB |
| Background removal | [`onnx-community/BiRefNet_lite-ONNX`](https://huggingface.co/onnx-community/BiRefNet_lite-ONNX) | MIT | ~210 MB |
| Captioning | [`Salesforce/blip-image-captioning-base`](https://huggingface.co/Salesforce/blip-image-captioning-base) | BSD-3-Clause | ~990 MB |
| Zero-shot classification | [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32) | MIT | ~605 MB |

## Configuration

### Model cache

ONNX model weights are cached in a per-user directory by default (`~/.cache/image_vision` on Linux, `~/Library/Caches/image_vision` on macOS). Override in `config/runtime.exs`:

```elixir
config :image_vision, :cache_dir, "/var/lib/my_app/models"
```

### Classification serving

`Image.Classification` runs a supervised [Bumblebee](https://hex.pm/packages/bumblebee) serving. It does not autostart by default. Start it in your application's supervision tree:

```elixir
# application.ex
def start(_type, _args) do
  children = [
    Image.Classification.classifier(),
    Image.Classification.embedder()   # omit if you do not need embeddings
  ]

  Supervisor.start_link(children, strategy: :one_for_one)
end
```

Or enable autostart via config so `ImageVision.Supervisor` handles it:

```elixir
# config/runtime.exs
config :image_vision, :classifier, autostart: true
```

To use a different model:

```elixir
config :image_vision, :classifier,
  model: {:hf, "facebook/convnext-large-224-22k-1k"},
  featurizer: {:hf, "facebook/convnext-large-224-22k-1k"},
  autostart: true
```

## Guides

- [Classification](https://github.com/elixir-image/image_vision/blob/v0.2.0/guides/classification.md) — classifying images and computing embeddings
- [Detection](https://github.com/elixir-image/image_vision/blob/v0.2.0/guides/detection.md) — bounding-box object detection
- [Segmentation](https://github.com/elixir-image/image_vision/blob/v0.2.0/guides/segmentation.md) — promptable and panoptic segmentation
- [Background removal](https://github.com/elixir-image/image_vision/blob/v0.2.0/guides/background.md) — class-agnostic foreground cutout
- [Captioning](https://github.com/elixir-image/image_vision/blob/v0.2.0/guides/captioning.md) — natural-language image descriptions
- [Zero-shot classification](https://github.com/elixir-image/image_vision/blob/v0.2.0/guides/zero_shot.md) — classify against arbitrary labels via CLIP
