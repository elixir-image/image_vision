# TODO

## ~~Add `Image.FaceDetection`~~ — shipped in v0.3.0

`Image.FaceDetection` ships with the `detect/2`, `boxes/2`,
`crop_largest/2`, and `draw_boxes/3` API, backed by YuNet
2023-March (MIT, ~340 KB). The `crop_largest/2` helper is the
wire-in point for the face-aware crop bias still pending in
`image_plug`'s interpreter — `gravity: :face`, ImageKit `z-`,
Cloudflare `face-zoom`, and Cloudinary `e_pixelate_faces` can
all drive their face-aware behaviour through it once the
`image_plug` interpreter passes detection results back into
the resize / pixelate ops.

The original recommendation notes (kept for context):

The default model should follow the same conventions as
`Image.Background`, `Image.Detection`, and
`Image.Segmentation`:

* ONNX export, loaded via Ortex, conditional on
  `ImageVision.ortex_configured?()`.
* Hosted on HuggingFace under a stable namespace.
* Permissive licence (MIT / Apache 2.0 — *not* AGPL).
* Sensible-defaults API: `detect/2` returns
  `[%{box: %{x:, y:, w:, h:}, score:, landmarks: [{x, y}, …]}]`.

### Recommended primary model: YuNet (OpenCV)

* HuggingFace: [`opencv/face_detection_yunet`](https://huggingface.co/opencv/face_detection_yunet)
  (or any community ONNX export of the upstream
  [opencv_zoo YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)).
* Size: ~340 KB — **the smallest production-quality face
  detector available**. Comparable to BlazeFace but with
  better accuracy.
* Licence: MIT.
* Output: bounding boxes + 5 facial landmarks (left eye,
  right eye, nose tip, mouth corners) + confidence.
* Input: 320×320 (configurable). Real-time on CPU.
* Maintained by the OpenCV team — long-term stability is high.

This is the best "sensible default" for a defaults-first
library. The 340 KB size means it's reasonable to ship in a
Docker image without bloating the layer.

### Recommended alternative for higher accuracy: SCRFD

* HuggingFace: [`onnx-community/scrfd_2.5g_bnkps`](https://huggingface.co/onnx-community/scrfd_2.5g_bnkps)
  (or the larger `scrfd_10g_bnkps` for production servers).
* Size: 3 MB (2.5g variant) to ~17 MB (10g).
* Licence: MIT (InsightFace).
* Output: same shape as YuNet — boxes + 5 landmarks + score.
* Slightly better mAP on WIDER FACE than YuNet, especially
  on small / occluded faces.
* Same on-disk model format / preprocessing path; an
  `:image_vision` user can swap via `:repo` and `:model_file`
  options without code changes.

### Recommended ultra-light alternative: BlazeFace (MediaPipe)

* HuggingFace: [`onnx-community/blazeface`](https://huggingface.co/Xenova/blazeface) /
  community exports of the
  [MediaPipe BlazeFace](https://github.com/google/mediapipe).
* Size: ~250 KB.
* Licence: Apache 2.0.
* Output: boxes + 6 landmarks.
* Good for selfies / portrait photography (front-facing
  variant) but worse on group photos or occluded faces than
  YuNet / SCRFD.

### Sketch of `Image.FaceDetection` API

```elixir
{:ok, image} = Image.open("./crowd.jpg")

[%{box: box, score: score, landmarks: marks} | _] =
  Image.FaceDetection.detect(image)

# Convenience: just the bounding boxes, ranked.
boxes = Image.FaceDetection.boxes(image)

# Crop to the largest detected face with N% padding.
{:ok, portrait} = Image.FaceDetection.crop_largest(image, padding: 0.2)
```

The `crop_largest/2` helper is the wire-in point for
`gravity: :face`, ImageKit `z-<n>`, and Cloudflare
`face-zoom` over in `image_plug`.

## Other ideas (lower priority)

* **Pose estimation** (MediaPipe Pose / RT-Pose) — for
  pose-aware cropping. Niche, but useful for sports / fashion
  imagery.

* **OCR-aware detection** — wrap `image_ocr` to expose
  `Image.FaceDetection`-style "where are the text regions"
  results. Sibling concern; could land in `image_ocr` itself.

* **Aesthetic-quality scoring** — model that rates an image's
  composition / sharpness / exposure. Useful for picking the
  best frame from a video or the best variant from a batch.
  Several open-source variants exist (NIMA, MUSIQ); none yet
  packaged as a clean ONNX export with a permissive licence.
