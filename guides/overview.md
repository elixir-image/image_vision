# Computer-vision tasks at a glance

`image_vision` ships seven public modules, each wrapping a different ONNX or Bumblebee-served model. This guide is a map: it explains which module solves which kind of problem, shows the call shape for each, and points you at the per-module guides for the depth.

Four task families cover everything in the library:

* **Classification** — "what is this image about?". Returns a list of labels.
* **Detection** — "where are the things?". Returns bounding boxes plus labels.
* **Segmentation** — "which pixels belong to which thing?". Returns pixel masks.
* **Description** — "what's happening in this image, in words?". Returns a natural-language sentence.

Most computer-vision tasks reduce to one of these four. Some applications — face-aware crops, background removal, content moderation — stitch two or three together.

## The decision table

| You want… | Use | Returns |
| --- | --- | --- |
| One-line label for what's in the image | `Image.Classification.labels/2` | `["Blenheim spaniel"]` |
| Embedding vector for similarity / search | `Image.Classification.embed/2` | `Nx.Tensor` (1000-dim by default) |
| Classify against custom labels (no retraining) | `Image.ZeroShot.classify/3` | `[%{label: "cat", score: 0.92}, …]` |
| Bounding boxes around every detected object | `Image.Detection.detect/2` | `[%{box: {x, y, w, h}, label: "dog", score: 0.87}, …]` |
| Bounding boxes around every face (plus landmarks) | `Image.FaceDetection.detect/2` | `[%{box: …, score: …, landmarks: [{x, y}, …]}, …]` |
| Cut out a specific object from a click or box | `Image.Segmentation.segment/2` | `Vix.Vips.Image` mask |
| Class-labeled regions for the whole scene | `Image.Segmentation.segment_panoptic/2` | `[%{label: "sky", mask: …}, …]` |
| Extract just the foreground subject | `Image.Background.remove/2` | `Vix.Vips.Image` with alpha |
| Natural-language caption for the image | `Image.Captioning.caption/2` | `"a dog sitting on a park bench"` |

The same `t:Vix.Vips.Image.t/0` is the input for everything; the modules differ only in what they extract from it.

## Classification — "what is this image about?"

`Image.Classification` runs an ImageNet-trained backbone (ConvNeXt by default) and returns the top labels. It's the cheapest task in the library — fast inference, small model, no per-pixel work — so it's a good first reach for content tagging, search indexing, or anything that needs a coarse "what kind of image is this?" signal.

```elixir
puppy = Image.open!("puppy.jpg")

Image.Classification.labels(puppy)
#=> ["Blenheim spaniel"]

Image.Classification.labels(puppy, min_score: 0.1)
#=> ["Blenheim spaniel", "cocker spaniel", "papillon"]
```

For similarity search — "find me other images that look like this" — use `embed/2` instead. It returns the model's penultimate-layer activations as an `Nx.Tensor`, which you can dot-product against pre-computed embeddings of your library:

```elixir
query_vec = Image.Classification.embed(puppy)
# Compare against pre-stored vectors via cosine similarity, etc.
```

When ImageNet's 1000 categories don't cover your labels — "branded vs non-branded", "indoor vs outdoor", "edited vs unedited" — reach for `Image.ZeroShot` instead. It uses CLIP to classify against arbitrary labels you supply at request time:

```elixir
Image.ZeroShot.classify(puppy, ["a dog", "a cat", "a bird"])
#=> [%{label: "a dog", score: 0.97}, %{label: "a cat", score: 0.02}, …]
```

CLIP is heavier than ConvNeXt (a vision transformer plus a text transformer) but the labels are free-form, which makes it the right tool for any taxonomy you didn't train against.

See [`classification.md`](classification.md) and [`zero_shot.md`](zero_shot.md) for the full API surface.

## Detection — "where are the things?"

`Image.Detection` and `Image.FaceDetection` both return bounding boxes; they differ in what they look for and how rich the per-box metadata is.

`Image.Detection` uses RT-DETR (real-time detection transformer) trained on COCO, so it knows ~80 object categories — people, vehicles, animals, household items, food. Box plus label plus confidence:

```elixir
park = Image.open!("park.jpg")

Image.Detection.detect(park)
#=> [
#=>   %{box: {120, 80, 240, 380}, label: "person",  score: 0.95},
#=>   %{box: {410, 220, 180, 160}, label: "dog",    score: 0.91},
#=>   %{box: {620, 30,  300, 200}, label: "kite",   score: 0.82}
#=> ]
```

`Image.FaceDetection` uses YuNet, a much smaller and faster model specialised for faces. The trade-off: only one class (faces), but you also get five-point landmarks (eyes, nose, mouth corners) per box:

```elixir
portrait = Image.open!("portrait.jpg")

Image.FaceDetection.detect(portrait)
#=> [
#=>   %{
#=>     box: {142, 84, 218, 240},
#=>     score: 0.98,
#=>     landmarks: [{195, 162}, {275, 160}, {235, 200}, {200, 240}, {270, 240}]
#=>   }
#=> ]
```

Both modules ship a `draw/2` helper that overlays the boxes on the image — useful for debugging and for the "show the user what the model saw" UX.

`Image.FaceDetection` is the basis for `image_plug`'s `gravity: :face` crop and `Ops.PixelateFaces`. See [`detection.md`](detection.md) for the general object case and [`image_plug`'s face-aware guide](https://hexdocs.pm/image_plug/face_aware.html) for the face-aware crop integration.

## Segmentation — "which pixels belong to which thing?"

`Image.Segmentation` is the heaviest task by far — it produces per-pixel masks rather than just boxes. Two flavours, with very different shapes:

* **Promptable** (`segment/2`) — you give it a hint ("here's where the object is") and it returns one mask. Backed by SAM 2. Good for "the user clicked here, cut out what they pointed at".

* **Panoptic** (`segment_panoptic/2`) — no prompt, returns one mask per detected region with a class label. Good for scene parsing — sky, road, building, person, car all separated automatically.

Promptable, with a click point:

```elixir
photo = Image.open!("scene.jpg")
mask  = Image.Segmentation.segment(photo, point: {340, 220})

# Apply the mask: blank out everything outside the segmented object.
{:ok, isolated} = Vix.Vips.Operation.bandjoin([photo, mask])
```

Promptable, with a bounding box:

```elixir
mask = Image.Segmentation.segment(photo, box: {120, 80, 280, 320})
```

Panoptic — every region gets a class:

```elixir
Image.Segmentation.segment_panoptic(photo)
#=> [
#=>   %{label: "sky",      mask: %Vix.Vips.Image{...}},
#=>   %{label: "person",   mask: %Vix.Vips.Image{...}},
#=>   %{label: "building", mask: %Vix.Vips.Image{...}}
#=> ]
```

`Image.Background` is segmentation's narrowest and most ergonomic case: foreground vs background, no prompt needed. It's what you reach for when the only thing you actually want is "give me the subject with a transparent background":

```elixir
{:ok, cutout} = Image.Background.remove(photo)
# `cutout` is the original image with everything outside the detected
# foreground subject made transparent.
```

`Image.Background.remove/2` is a class-agnostic foreground extractor — it works on any subject (person, product, pet, plant) without needing to be told what to look for. For arbitrary user-pointed segmentation (the "click anywhere" UX), use `Image.Segmentation.segment/2` instead.

See [`segmentation.md`](segmentation.md) and [`background.md`](background.md) for the full call shapes.

## Description — "what's happening, in words?"

`Image.Captioning` produces a natural-language sentence describing the image, using BLIP (Bootstrapping Language-Image Pre-training):

```elixir
photo = Image.open!("park.jpg")

Image.Captioning.caption(photo)
#=> "a man walking his dog in a park"
```

Captions are useful for accessibility (`alt` text generation), search (full-text indexing on caption strings), and content workflows (auto-tagging that would have to enumerate labels otherwise). The model is generative, so the same image can produce slightly different captions across runs unless you fix the seed; for stable indexing pin the generation parameters via `:generation_config`.

BLIP is the heaviest model in the library — a vision encoder plus a language decoder, a few hundred MB on disk and several seconds per inference on CPU. For high-throughput captioning you want a GPU-backed Nx backend (EXLA on CUDA or Metal); CPU works for low-volume use.

See [`captioning.md`](captioning.md) for prompt templates, batch processing, and the generation-parameter knobs.

## Composing tasks

A few common combinations:

* **Face-aware crop** — `FaceDetection.detect/2` → take the highest-confidence box → crop to that region with padding. This is what `image_plug`'s `gravity: :face` does end-to-end.

* **Pixelate faces** — `FaceDetection.detect/2` → for each box, apply pixelation only inside the region. This is `Image.Plug.Pipeline.Ops.PixelateFaces`.

* **Caption + tag** — `Captioning.caption/2` for a sentence + `Classification.labels/2` for indexable labels. Caption is human-readable, labels are queryable; storing both gives you free-text search plus faceted filtering.

* **Background swap** — `Background.remove/2` to extract the subject → composite onto a different background image with `Image.compose/3`. The "studio shot" effect from arbitrary phone photos.

* **Object-aware blur** — `Detection.detect/2` → for each detection over a confidence threshold, blur outside the box. Useful for privacy redaction at scale.

* **Visual search** — `Classification.embed/2` on every image at index time → store the vectors in a vector database → at query time, `embed` the query image and dot-product. This is the standard "more like this" search.

* **Custom-taxonomy moderation** — `ZeroShot.classify/3` with your moderation labels (`["safe", "violent", "explicit", …]`). No training data, no model fine-tuning — just label strings.

The library is designed to be composable: the modules don't share state, every function takes a `Vix.Vips.Image`, and every output is either another `Vix.Vips.Image` (so you can pipe it into the next step) or a plain Elixir term (so you can serialise / store / route it).

## Optional dependencies and model loading

Different modules need different runtimes:

| Module | Runtime | Notes |
| --- | --- | --- |
| `Image.Classification`, `Image.Captioning`, `Image.ZeroShot` | Bumblebee + Nx + EXLA | Hugging Face model loading; GPU-friendly via EXLA backend. |
| `Image.Detection`, `Image.Segmentation`, `Image.Background` | Ortex + Nx | ONNX Runtime; faster cold-start than Bumblebee, no GPU acceleration on macOS. |
| `Image.FaceDetection` | Ortex + Nx | YuNet ONNX model, ~340 KB on disk. |

Add `:image_vision` to your deps plus the runtime stack you need. The minimum to use everything is:

```elixir
def deps do
  [
    {:image_vision, "~> 0.3"},
    {:bumblebee,    "~> 0.6"},
    {:ortex,        "~> 0.1"},
    {:nx,           "~> 0.10"},
    {:exla,         "~> 0.10"}
  ]
end
```

If you only need Ortex-backed tasks (Detection / Segmentation / Background / FaceDetection), you can drop Bumblebee. If you only need Bumblebee-backed tasks (Classification / Captioning / ZeroShot), you can drop Ortex.

### Model cache

All modules download their weights from HuggingFace on first use and cache them on disk. The default cache directory is OS-dependent:

* macOS: `~/Library/Caches/image_vision/`
* Linux: `~/.cache/image_vision/`
* configurable via `config :image_vision, :cache_dir, "/path/to/cache"`

In a containerised deployment, mount that directory as a volume so the model weights survive container restarts. See `image_playground`'s Dockerfile for an example.

### CPU vs GPU

Every model runs on CPU out of the box. For higher throughput, configure EXLA with a CUDA or Metal backend; both `bumblebee` and `nx_image` honour the configured backend automatically. On Apple Silicon the Metal backend is the realistic option; on Linux + NVIDIA, CUDA is the standard. See [Nx's installation guide](https://hexdocs.pm/exla/EXLA.html) for backend setup.

## Related

* [`classification.md`](classification.md) — full Classification API including `embed/2` and the model-config knobs.
* [`zero_shot.md`](zero_shot.md) — CLIP-based custom-label classification.
* [`detection.md`](detection.md) — RT-DETR object detection plus `draw/2` overlay.
* [`segmentation.md`](segmentation.md) — promptable (SAM 2) and panoptic segmentation.
* [`background.md`](background.md) — class-agnostic foreground extraction.
* [`captioning.md`](captioning.md) — BLIP-based natural-language captioning.
* [`image_plug`'s face-aware guide](https://hexdocs.pm/image_plug/face_aware.html) — `Image.FaceDetection` integrated into the URL-driven transform pipeline.
