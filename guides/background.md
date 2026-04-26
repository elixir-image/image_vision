# Background Removal

`Image.Background` answers "what is the foreground here?" by separating subject from background. The model is class-agnostic — it doesn't care whether the foreground is a person, a product, an animal, or a piece of furniture; it decides what's the salient subject and isolates it.

## Removing the background

The simplest call returns the input image with the background made transparent (alpha channel applied):

```elixir
iex> photo = Image.open!("portrait.jpg")
iex> {:ok, cutout} = Image.Background.remove(photo)
iex> Image.write!(cutout, "portrait-cutout.png")
```

The result has four bands (RGB + alpha). Save as PNG or any other format that supports transparency.

## Getting just the mask

If you want the foreground mask itself — for compositing onto a different background, layering, or further processing — call `mask/2`:

```elixir
iex> photo = Image.open!("portrait.jpg")
iex> mask = Image.Background.mask(photo)
iex> Image.write!(mask, "mask.png")
```

The mask is a single-band greyscale image at the same dimensions as the input. Pixel intensity reflects model confidence: pure white (255) is "definitely foreground", pure black (0) is "definitely background", values in between reflect uncertainty at boundaries.

## When to use this vs. segmentation

`image_vision` has three different ways to produce masks; pick by what you actually want:

- **`Image.Background.remove/2`** — class-agnostic, no input beyond the image. Best for "isolate the subject of this photo". Works without prompts. Single foreground/background distinction.

- **`Image.Segmentation.segment/2` (SAM 2)** — promptable. You click a point or draw a box and SAM masks *that specific object*. Best when an image has multiple distinct objects and you want one of them, or when the salient-object heuristic of background removal disagrees with what you actually want.

- **`Image.Segmentation.segment_panoptic/2`** — labels every region in the image with a class. Best when you want to enumerate everything in a scene, not just the foreground.

## Using a different model

`remove/2` and `mask/2` accept `:repo` and `:model_file` to swap in any compatible BiRefNet ONNX export:

```elixir
# Full BiRefNet (~890 MB, higher quality, slower)
iex> Image.Background.remove(image, repo: "onnx-community/BiRefNet-ONNX")
```

Both functions also share the `ImageVision.ModelCache` cache root with the segmentation and detection models, so configure once via `config :image_vision, :cache_dir, ...`.

## Pre-downloading

To populate the cache before first use:

```bash
mix image_vision.download_models --background
```

## Default model

[BiRefNet lite](https://huggingface.co/onnx-community/BiRefNet_lite-ONNX) is MIT licensed and ~210 MB. It's a distilled variant of the full BiRefNet that trades a small amount of accuracy for a much smaller model and faster inference. State-of-the-art for salient-object detection / dichotomous image segmentation as of mid-2024.

## Dependencies

Background removal requires `:ortex`. Add to `mix.exs`:

```elixir
{:ortex, "~> 0.1"}
```
