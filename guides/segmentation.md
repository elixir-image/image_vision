# Image Segmentation

`Image.Segmentation` produces pixel-level masks: which pixels belong to a given object or region.

Two functions cover different use cases:

- `segment/2` — **promptable**: click a point or draw a box to cut out a specific object.
- `segment_panoptic/2` — **class-labeled**: every region in the image gets a label (`person`, `car`, `sky`…).

## Promptable segmentation (SAM 2)

### Segment the centre object

With no prompt, `segment/2` segments whatever is at the centre of the image:

```elixir
iex> image = Image.open!("product_photo.jpg")
iex> %{mask: mask, score: score} = Image.Segmentation.segment(image)
iex> score
0.94
```

### Segment by point

```elixir
iex> %{mask: mask} = Image.Segmentation.segment(image, prompt: {:point, 320, 240})
```

### Segment by bounding box

```elixir
iex> %{mask: mask} = Image.Segmentation.segment(image, prompt: {:box, 100, 50, 200, 300})
```

The box is `{x, y, width, height}` in pixel coordinates of the original image.

### Multiple prompts

Pass a list of `{:point, x, y}` tuples to guide the model toward a specific object when a single point is ambiguous:

```elixir
iex> %{mask: mask} = Image.Segmentation.segment(image,
...>   prompt: [{:point, 320, 240}, {:point, 340, 260}])
```

### Getting all candidate masks

SAM produces three mask candidates for every prompt. Retrieve them all with `multimask: true`:

```elixir
iex> masks = Image.Segmentation.segment(image, multimask: true)
iex> length(masks)
3
iex> hd(masks).score
0.97
```

## Class-labeled segmentation (DETR-panoptic)

`segment_panoptic/2` returns one segment per detected region, each with a class label and a binary mask:

```elixir
iex> street = Image.open!("street.jpg")
iex> segments = Image.Segmentation.segment_panoptic(street)
iex> Enum.map(segments, & {&1.label, Float.round(&1.score, 2)})
[{"person", 0.97}, {"car", 0.93}, {"road", 0.88}, {"sky", 0.85}]
```

Uses 250 COCO panoptic categories covering everyday objects and background regions.

## Composing results with the original image

### Cut out a segmented object

`apply_mask/2` makes the mask the alpha channel — white pixels become opaque, black pixels transparent:

```elixir
iex> %{mask: mask} = Image.Segmentation.segment(image)
iex> {:ok, cutout} = Image.Segmentation.apply_mask(image, mask)
iex> Image.save!(cutout, "cutout.png")
```

### Colour-coded overlay

`compose_overlay/3` draws a colour-coded overlay of all segments:

```elixir
iex> overlay = Image.Segmentation.compose_overlay(street, segments)
iex> Image.save!(overlay, "segmented.jpg")
```

Adjust transparency with `:alpha` (default `0.5`):

```elixir
iex> overlay = Image.Segmentation.compose_overlay(street, segments, alpha: 0.3)
```

## Using a different model

Both `segment/2` and `segment_panoptic/2` accept options to swap models. They are passed per call rather than via app config — neither function uses a long-running serving, so there is no autostart cost to overriding on a single call.

### Promptable (SAM 2)

```elixir
# Use a larger SAM 2 variant for better quality on small or thin objects
iex> Image.Segmentation.segment(image,
...>   prompt: {:point, 320, 240},
...>   repo: "SharpAI/sam2-hiera-small-onnx")
```

`segment/2` accepts:

- `:repo` — any HuggingFace repo containing a SAM 2 ONNX export with separate encoder and decoder files
- `:encoder_file` — encoder filename within the repo (default `"encoder.onnx"`)
- `:decoder_file` — decoder filename within the repo (default `"decoder.onnx"`)

The protocol matches `SharpAI/sam2-hiera-tiny-onnx` (separate encoder/decoder, the standard SAM 2 ONNX export shape). Repos that bundle both into a single file or use a different I/O layout will not work without changes to the wrapper.

### Class-labeled (DETR-panoptic)

```elixir
# Quantized variant — much smaller, some accuracy cost
iex> Image.Segmentation.segment_panoptic(image, model_file: "onnx/model_quantized.onnx")

# A different ONNX-exported DETR-panoptic repo
iex> Image.Segmentation.segment_panoptic(image, repo: "your-org/detr-panoptic-onnx")
```

`segment_panoptic/2` accepts:

- `:repo` — any HuggingFace repo with a DETR-panoptic ONNX export and a `config.json` providing `id2label`
- `:model_file` — ONNX filename within the repo (default `"onnx/model.onnx"`)

Labels are read from the repo's `config.json`. Where that config has placeholder `LABEL_n` entries, the wrapper falls back to the canonical [COCO panoptic taxonomy](https://github.com/cocodataset/panopticapi/blob/master/panoptic_coco_categories.json), so common stuff classes (`sky-other-merged`, `mountain-merged`, `grass-merged`, …) resolve correctly even on repos with incomplete configs.

### Pre-downloading

To populate the cache before first use:

```bash
mix image_vision.download_models --segment
```

This fetches the configured defaults. For non-default repos, the cache populates on first call to `segment/2` or `segment_panoptic/2`.

## Dependencies

Segmentation requires `:ortex`. Add to `mix.exs`:

```elixir
{:ortex, "~> 0.1"}
```

Model weights (~150 MB for SAM 2, ~175 MB for DETR) are downloaded on first call and cached. Configure the cache directory with:

```elixir
config :image_vision, :cache_dir, "/path/to/cache"
```
