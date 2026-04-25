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

## Dependencies

Segmentation requires `:ortex`. Add to `mix.exs`:

```elixir
{:ortex, "~> 0.1"}
```

Model weights (~150 MB for SAM 2, ~175 MB for DETR) are downloaded on first call and cached. Configure the cache directory with:

```elixir
config :image_vision, :cache_dir, "/path/to/cache"
```
