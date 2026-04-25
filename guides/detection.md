# Object Detection

`Image.Detection` answers "where are the objects in this image and what are they?". It returns a list of bounding boxes with class labels and confidence scores.

## Basic detection

```elixir
iex> street = Image.open!("street.jpg")
iex> detections = Image.Detection.detect(street)
iex> hd(detections)
%{label: "person", score: 0.97, box: {120, 45, 60, 180}}
```

Each detection is a map with:
- `:label` — one of 80 COCO class names (`person`, `bicycle`, `car`, `dog`, …)
- `:score` — confidence score in `[0.0, 1.0]`
- `:box` — `{x, y, width, height}` in pixel coordinates of the original image

Results are sorted by descending confidence.

## Filtering by confidence

The default minimum score is `0.5`. Raise it to get only high-confidence detections:

```elixir
iex> Image.Detection.detect(street, min_score: 0.8)
[%{label: "person", score: 0.94, box: {120, 45, 60, 180}}, ...]
```

## Drawing bounding boxes

`draw_bbox_with_labels/2` annotates the image:

```elixir
iex> detections = Image.Detection.detect(street)
iex> annotated = Image.Detection.draw_bbox_with_labels(detections, street)
iex> Image.save!(annotated, "annotated.jpg")
```

It accepts the same image that `detect/2` was called on. Pipeline:

```elixir
iex> street
...> |> Image.Detection.detect()
...> |> Image.Detection.draw_bbox_with_labels(street)
...> |> Image.save!("annotated.jpg")
```

## Available classes

The default RT-DETR model is trained on COCO 80 classes:

```elixir
iex> Image.Detection.classes()
["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", ...]
```

## Using a different model

`detect/2` accepts `:repo` and `:filename` to swap in any RT-DETR-family ONNX model from HuggingFace:

```elixir
# Smaller R18 backbone (~80 MB) — faster, slightly less accurate
iex> Image.Detection.detect(image, repo: "onnx-community/rtdetr_r18vd")

# Quantized variant of the default (~45 MB INT8) — much smaller download, some accuracy cost
iex> Image.Detection.detect(image, filename: "onnx/model_quantized.onnx")
```

For one-off use, pass options per call. To make a non-default model the project default, you can wrap the call:

```elixir
defp detect(image), do: Image.Detection.detect(image, repo: "onnx-community/rtdetr_r18vd")
```

To pre-download a model into the cache:

```bash
mix image_vision.download_models --detect
```

(The download task fetches the configured default; if you've changed the repo for a single call site, the cache will populate on first use.)

### Caveat: COCO 80 labels are hardcoded

`detect/2` maps class indices to label strings using a baked-in COCO 80 list ([detection.ex](https://github.com/elixir-image/image_vision/blob/main/lib/detection.ex)). RT-DETR models trained on a different label set (e.g. Open Images, custom domains) will produce indices the wrapper can't translate — labels will be wrong even though boxes and scores are correct. For non-COCO models, use the underlying `Ortex.run/2` directly with the model's own `id2label`.

## Default model

RT-DETR (`onnx-community/rtdetr_r50vd`) is a real-time transformer-based detector that outperforms YOLOv8 on COCO while being **Apache 2.0 licensed** (YOLOv8/11 are AGPL). It is NMS-free — no Non-Maximum Suppression post-processing is needed.

Model weights are downloaded on first call and cached. Configure the cache directory with:

```elixir
config :image_vision, :cache_dir, "/path/to/cache"
```

## Dependencies

Detection requires `:ortex`. Add to `mix.exs`:

```elixir
{:ortex, "~> 0.1"}
```
