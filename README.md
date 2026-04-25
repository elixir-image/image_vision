# ImageVision

`ImageVision` is a simple, opinionated image vision library for Elixir. It sits alongside the [`image`](https://hex.pm/packages/image) library and answers three questions about any image — **what's in it**, **where are the objects**, and **which pixels belong to which object** — with strong defaults and no ML expertise required.

## Tasks

| Task | Module | What it does |
|---|---|---|
| Classification | [`Image.Classification`](https://hexdocs.pm/image_vision/Image.Classification.html) | Labels like `"sports car"` or `"Blenheim spaniel"` |
| Embedding | [`Image.Classification`](https://hexdocs.pm/image_vision/Image.Classification.html) | 768-dim feature vector for similarity search |
| Segmentation | [`Image.Segmentation`](https://hexdocs.pm/image_vision/Image.Segmentation.html) | Pixel masks — promptable ("cut out this object") or class-labeled ("every region in the image") |
| Detection | [`Image.Detection`](https://hexdocs.pm/image_vision/Image.Detection.html) | Bounding boxes with class labels |

## Installation

```elixir
def deps do
  [
    {:image_vision, "~> 0.2"},
    # Nx backend for classification (pick one)
    {:exla, "~> 0.9"},
    # Required for classification
    {:bumblebee, "~> 0.6"},
    # Required for segmentation and detection
    {:ortex, "~> 0.1"}
  ]
end
```

## Quick examples

```elixir
# Classification — what's in this image?
iex> puppy = Image.open!("puppy.jpg")
iex> Image.Classification.labels(puppy)
["Blenheim spaniel"]

# Embedding — feature vector for similarity search
iex> Image.Classification.embed(puppy)
#Nx.Tensor<f32[768]>

# Promptable segmentation — mask the object at the centre
iex> %{mask: mask} = Image.Segmentation.segment(puppy)
iex> cutout = Image.Segmentation.apply_mask!(puppy, mask)

# Promptable segmentation — mask the object at a specific point
iex> %{mask: mask} = Image.Segmentation.segment(puppy, prompt: {:point, 320, 240})

# Class-labeled segmentation — every region
iex> street = Image.open!("street.jpg")
iex> segments = Image.Segmentation.segment_panoptic(street)
iex> Enum.map(segments, & &1.label)
["person", "car", "road", "sky"]

# Overlay coloured segments on the original image
iex> overlay = Image.Segmentation.compose_overlay(street, segments)

# Object detection — bounding boxes with labels
iex> detections = Image.Detection.detect(street)
iex> hd(detections)
%{label: "person", score: 0.94, box: {120, 45, 60, 180}}

# Draw bounding boxes on the image
iex> annotated = Image.Detection.draw_bbox_with_labels(detections, street)
```

## Default models

All defaults are permissively licensed (Apache 2.0 / MIT). Models are downloaded automatically on first call and cached to disk — no manual setup needed.

| Task | Model | License | Size |
|---|---|---|---|
| Classification | `facebook/convnext-tiny-224` | Apache 2.0 | ~110 MB |
| Embedding | `facebook/dinov2-base` | Apache 2.0 | ~340 MB |
| Promptable segmentation | `SharpAI/sam2-hiera-tiny-onnx` (SAM 2) | Apache 2.0 | ~150 MB |
| Class-labeled segmentation | `Xenova/detr-resnet-50-panoptic` | Apache 2.0 | ~175 MB |
| Object detection | `onnx-community/rtdetr_r50vd` (RT-DETR) | Apache 2.0 | ~175 MB |

## Configuration

### Model cache

ONNX models are cached in a per-user directory by default. Override in `config/runtime.exs`:

```elixir
config :image_vision, :cache_dir, "/var/lib/my_app/models"
```

The default is `~/Library/Caches/image_vision` on macOS and `~/.cache/image_vision` on Linux.

### Classification serving

`Image.Classification` runs a supervised Bumblebee serving. It autostarts by default:

```elixir
# config/runtime.exs
config :image_vision, :classifier,
  model: {:hf, "facebook/convnext-large-224-22k-1k"},  # larger model
  featurizer: {:hf, "facebook/convnext-large-224-22k-1k"},
  autostart: true
```

Set `autostart: false` to manage the serving in your own supervision tree:

```elixir
# application.ex
children = [
  Image.Classification.classifier(),
  Image.Classification.embedder()
]
```

## Guides

- [Classification](https://github.com/elixir-image/image_vision/blob/v0.2.0/guides/classification.md) — classifying images and computing embeddings
- [Segmentation](https://github.com/elixir-image/image_vision/blob/v0.2.0/guides/segmentation.md) — promptable and class-labeled segmentation
- [Detection](https://github.com/elixir-image/image_vision/blob/v0.2.0/guides/detection.md) — bounding-box object detection

## Design

`ImageVision` is for developers who want answers, not ML configuration. The defaults are chosen for permissive licensing, broad applicability, and reasonable size — not raw benchmark ranking. Power users can override every default via options; most users never need to.

See [CLAUDE.md](https://github.com/elixir-image/image_vision/blob/v0.2.0/CLAUDE.md) for the full design rationale.
