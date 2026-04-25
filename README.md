# ImageVision

`ImageVision` is a simple, opinionated image vision library for Elixir. It sits alongside the [`image`](https://hex.pm/packages/image) library and answers three questions about any image — **what's in it**, **where are the objects**, and **which pixels belong to which object** — with strong defaults and no ML expertise required.

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
```

## Installation

Add `:image_vision` to `mix.exs` along with whichever optional ML backends you need:

```elixir
def deps do
  [
    {:image_vision, "~> 0.2"},

    # Required for Image.Classification and Image.Classification.embed/2
    {:bumblebee, "~> 0.6"},
    {:nx, "~> 0.11"},
    {:exla, "~> 0.9"},      # or {:torchx, "~> 0.9"} for Torch backend

    # Required for Image.Detection and Image.Segmentation
    {:ortex, "~> 0.1"}
  ]
end
```

All ML deps are optional — omit any you do not use. The library compiles cleanly without them.

## Default models

All models are permissively licensed. Weights are downloaded automatically on first call and cached on disk — no manual setup required.

| Task | Model | License | Size |
|---|---|---|---|
| Classification | [`facebook/convnext-tiny-224`](https://huggingface.co/facebook/convnext-tiny-224) | Apache 2.0 | ~110 MB |
| Embedding | [`facebook/dinov2-base`](https://huggingface.co/facebook/dinov2-base) | Apache 2.0 | ~340 MB |
| Object detection | [`onnx-community/rtdetr_r50vd`](https://huggingface.co/onnx-community/rtdetr_r50vd) | Apache 2.0 | ~175 MB |
| Promptable segmentation | [`SharpAI/sam2-hiera-tiny-onnx`](https://huggingface.co/SharpAI/sam2-hiera-tiny-onnx) | Apache 2.0 | ~150 MB |
| Panoptic segmentation | [`Xenova/detr-resnet-50-panoptic`](https://huggingface.co/Xenova/detr-resnet-50-panoptic) | Apache 2.0 | ~175 MB |

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
