# Face Detection

`Image.FaceDetection` answers "where are the faces in this image?". It returns a list of bounding boxes, confidence scores, and five facial landmarks (right eye, left eye, nose tip, right mouth corner, left mouth corner) per detected face.

## Basic detection

```elixir
iex> image = Image.open!("group.jpg")
iex> faces = Image.FaceDetection.detect(image)
iex> hd(faces)
%{
  box: {412, 88, 96, 124},
  score: 0.94,
  landmarks: [{438.2, 130.1}, {478.7, 129.6}, {458.0, 152.3}, {442.1, 178.5}, {475.0, 178.2}]
}
```

Each detection is a map with:
- `:box` — `{x, y, width, height}` in pixel coordinates of the original image
- `:score` — confidence score in `[0.0, 1.0]`
- `:landmarks` — a list of five `{x, y}` tuples: right eye, left eye, nose tip, right mouth corner, left mouth corner — in that order

Results are sorted by descending confidence.

## Filtering by confidence

The default minimum score is `0.6`. Raise it for stricter detections:

```elixir
iex> Image.FaceDetection.detect(image, min_score: 0.8)
```

`:nms_iou` (default `0.3`) controls how aggressively overlapping boxes are collapsed by non-maximum suppression. Lower values keep fewer overlapping faces.

## Boxes only

When landmarks aren't needed, `boxes/2` skips them:

```elixir
iex> Image.FaceDetection.boxes(image)
[{412, 88, 96, 124}, {612, 102, 84, 110}]
```

## Drawing detections

`draw_boxes/3` overlays bounding boxes, the score as a percentage label, and the five landmark dots:

```elixir
iex> faces = Image.FaceDetection.detect(image)
iex> annotated = Image.FaceDetection.draw_boxes(faces, image)
iex> Image.write!(annotated, "annotated.jpg")
```

Pipeline form:

```elixir
iex> image
...> |> Image.FaceDetection.detect()
...> |> Image.FaceDetection.draw_boxes(image)
...> |> Image.write!("annotated.jpg")
```

Drawing options include `:color`, `:stroke_width`, `:landmark_radius`, `:font_size`, and `:show_landmarks?` (set to `false` to skip the dots).

## Face-aware crop

`crop_largest/2` is a convenience for the common "crop to the most prominent face" case (the wire-in point for face-aware crop bias used by `gravity: :face` in `image_plug`, ImageKit `z-`, and Cloudflare `face-zoom`):

```elixir
iex> {:ok, portrait} = Image.FaceDetection.crop_largest(image, padding: 0.2)
```

The largest face is chosen by bounding-box area. `:padding` is a fraction of each face dimension — `0.0` is a tight crop, `0.5` adds 50% on each side, `1.0` doubles the box. The expanded crop is clipped to the image bounds.

When no face meets the score threshold, `crop_largest/2` returns `{:error, :no_face_detected}`.

## Default model

[YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (`opencv/face_detection_yunet`) — the OpenCV team's production face detector. Roughly **340 KB on disk**, MIT licensed, real-time on CPU. The 2023-March export produces decoded boxes, keypoints, and scores directly.

Model weights are downloaded on first call and cached. Configure the cache directory with:

```elixir
config :image_vision, :cache_dir, "/path/to/cache"
```

## Using a different model

`detect/2` accepts `:repo` and `:model_file` to swap in a different YuNet ONNX export:

```elixir
iex> Image.FaceDetection.detect(image,
...>   repo: "opencv/face_detection_yunet",
...>   model_file: "face_detection_yunet_2023mar.onnx"
...> )
```

### Caveat: post-processor is YuNet 2023-March specific

The output decoder assumes YuNet's 2023-March 12-tensor convention (`cls_*`, `obj_*`, `bbox_*`, `kps_*` at strides 8/16/32, fixed 640×640 input). `SCRFD`, `BlazeFace`, and other face-detector exports produce different output shapes and need a different post-processor — they will not work as a drop-in replacement.

## Dependencies

Face detection requires `:ortex`. Add to `mix.exs`:

```elixir
{:ortex, "~> 0.1"}
```
