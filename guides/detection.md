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

## Choosing a lighter model

The default `onnx/model.onnx` (~175 MB FP32) gives the best accuracy. Use the quantized variant (~45 MB INT8) for a much smaller download at some accuracy cost:

```elixir
iex> Image.Detection.detect(image, filename: "onnx/model_quantized.onnx")
```

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
