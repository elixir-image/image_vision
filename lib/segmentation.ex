if ImageVision.ortex_configured?() do
  defmodule Image.Segmentation do
    @moduledoc """
    Image segmentation — which pixels belong to which object?

    Two complementary entry points cover different use cases:

    * `segment/2` — **promptable segmentation** via SAM 2. Click a
      point or draw a box and get back a precise mask for that object.
      Great for "cut out this foreground" or "mask this product".

    * `segment_panoptic/2` — **class-labeled segmentation** via
      DETR-panoptic. Every region in the image gets a class label
      (`person`, `car`, `sky`, `road`…). Great for "what's in this
      image and where?"

    ## Quick start

        # Mask the object at the centre of the image
        iex> image = Image.open!("photo.jpg")
        iex> %{mask: mask, score: _} = Image.Segmentation.segment(image)

        # Mask the object at a specific point
        iex> %{mask: mask} = Image.Segmentation.segment(image, prompt: {:point, 320, 240})

        # Class-label every region
        iex> segments = Image.Segmentation.segment_panoptic(image)
        iex> Enum.map(segments, & &1.label)
        ["person", "car", "road", "sky"]

    ## Composing results with an image

        # Make the masked object the only visible content (alpha mask)
        iex> cutout = Image.Segmentation.apply_mask(image, mask)

        # Colour-coded overlay of all segments
        iex> overlay = Image.Segmentation.compose_overlay(image, segments)

    ## Default models

    * **Promptable** — `SharpAI/sam2-hiera-tiny-onnx` (SAM 2 Tiny,
      Apache 2.0, encoder ~128 MB + decoder ~20 MB). Downloaded on
      first call via `ImageVision.ModelCache`.

    * **Class-labeled** — `Xenova/detr-resnet-50-panoptic` (DETR
      ResNet-50, Apache 2.0, ~172 MB). 250 COCO panoptic classes
      covering everyday things and stuff.

    Both can be overridden via options — see `segment/2` and
    `segment_panoptic/2` for details.

    ## Optional dependency

    This module is only available when
    [Ortex](https://hex.pm/packages/ortex) is configured in your
    application's `mix.exs`.

    """

    alias Vix.Vips.Image, as: Vimage

    # --- SAM 2 defaults -------------------------------------------------

    @sam_repo "SharpAI/sam2-hiera-tiny-onnx"
    @sam_encoder_file "encoder.onnx"
    @sam_decoder_file "decoder.onnx"
    @sam_input_size 1024

    # --- DETR-panoptic defaults -----------------------------------------

    @detr_repo "Xenova/detr-resnet-50-panoptic"
    @detr_model_file "onnx/model.onnx"
    @detr_config_file "config.json"
    # Resize short side to 800, keeping aspect (standard DETR pre-processing).
    @detr_short_side 800
    @detr_no_object_class 250

    # Canonical COCO panoptic id → label map (133 categories, IDs 1-200
    # with gaps), drawn from the official panoptic_coco_categories.json:
    # https://github.com/cocodataset/panopticapi/blob/master/panoptic_coco_categories.json
    #
    # Used as a fallback for IDs where the model's config.json carries
    # a placeholder `"LABEL_n"` instead of the real class name. Both the
    # original `facebook/detr-resnet-50-panoptic` config and the Xenova
    # ONNX repackage drop the names for IDs 183-200 (the COCO panoptic
    # "merged" stuff classes — sky-other-merged, mountain-merged, etc.),
    # even though the model predicts them confidently.
    @coco_panoptic_id2label %{
      1 => "person",
      2 => "bicycle",
      3 => "car",
      4 => "motorcycle",
      5 => "airplane",
      6 => "bus",
      7 => "train",
      8 => "truck",
      9 => "boat",
      10 => "traffic light",
      11 => "fire hydrant",
      13 => "stop sign",
      14 => "parking meter",
      15 => "bench",
      16 => "bird",
      17 => "cat",
      18 => "dog",
      19 => "horse",
      20 => "sheep",
      21 => "cow",
      22 => "elephant",
      23 => "bear",
      24 => "zebra",
      25 => "giraffe",
      27 => "backpack",
      28 => "umbrella",
      31 => "handbag",
      32 => "tie",
      33 => "suitcase",
      34 => "frisbee",
      35 => "skis",
      36 => "snowboard",
      37 => "sports ball",
      38 => "kite",
      39 => "baseball bat",
      40 => "baseball glove",
      41 => "skateboard",
      42 => "surfboard",
      43 => "tennis racket",
      44 => "bottle",
      46 => "wine glass",
      47 => "cup",
      48 => "fork",
      49 => "knife",
      50 => "spoon",
      51 => "bowl",
      52 => "banana",
      53 => "apple",
      54 => "sandwich",
      55 => "orange",
      56 => "broccoli",
      57 => "carrot",
      58 => "hot dog",
      59 => "pizza",
      60 => "donut",
      61 => "cake",
      62 => "chair",
      63 => "couch",
      64 => "potted plant",
      65 => "bed",
      67 => "dining table",
      70 => "toilet",
      72 => "tv",
      73 => "laptop",
      74 => "mouse",
      75 => "remote",
      76 => "keyboard",
      77 => "cell phone",
      78 => "microwave",
      79 => "oven",
      80 => "toaster",
      81 => "sink",
      82 => "refrigerator",
      84 => "book",
      85 => "clock",
      86 => "vase",
      87 => "scissors",
      88 => "teddy bear",
      89 => "hair drier",
      90 => "toothbrush",
      92 => "banner",
      93 => "blanket",
      95 => "bridge",
      100 => "cardboard",
      107 => "counter",
      109 => "curtain",
      112 => "door-stuff",
      118 => "floor-wood",
      119 => "flower",
      122 => "fruit",
      125 => "gravel",
      128 => "house",
      130 => "light",
      133 => "mirror-stuff",
      138 => "net",
      141 => "pillow",
      144 => "platform",
      145 => "playingfield",
      147 => "railroad",
      148 => "river",
      149 => "road",
      151 => "roof",
      154 => "sand",
      155 => "sea",
      156 => "shelf",
      159 => "snow",
      161 => "stairs",
      166 => "tent",
      168 => "towel",
      171 => "wall-brick",
      175 => "wall-stone",
      176 => "wall-tile",
      177 => "wall-wood",
      178 => "water-other",
      180 => "window-blind",
      181 => "window-other",
      184 => "tree-merged",
      185 => "fence-merged",
      186 => "ceiling-merged",
      187 => "sky-other-merged",
      188 => "cabinet-merged",
      189 => "table-merged",
      190 => "floor-other-merged",
      191 => "pavement-merged",
      192 => "mountain-merged",
      193 => "grass-merged",
      194 => "dirt-merged",
      195 => "paper-merged",
      196 => "food-other-merged",
      197 => "building-other-merged",
      198 => "rock-merged",
      199 => "wall-other-merged",
      200 => "rug-merged"
    }

    @default_min_score 0.5

    # ImageNet normalisation (shared by both models).
    # Kept as plain lists — Nx.tensor/1 cannot be called at compile time
    # when the Nx backend is not yet started.
    @imagenet_mean [0.485, 0.456, 0.406]
    @imagenet_std [0.229, 0.224, 0.225]

    @typedoc """
    A segmented region returned by `segment_panoptic/2`.

    * `:label` — COCO panoptic class name, e.g. `"person"` or `"road"`.
    * `:score` — confidence score in `[0.0, 1.0]`.
    * `:mask` — single-band `t:Vix.Vips.Image.t/0`; white (255) pixels
      belong to this segment, black (0) do not.

    """
    @type segment :: %{
            label: String.t(),
            score: float(),
            mask: Vimage.t()
          }

    @typedoc """
    A mask returned by `segment/2`.

    * `:score` — SAM IoU prediction score.
    * `:mask` — single-band `t:Vix.Vips.Image.t/0` in original image
      dimensions; white pixels are the segmented object.

    """
    @type mask_result :: %{score: float(), mask: Vimage.t()}

    # --- Public API: segment/2 (SAM 2 promptable) -----------------------

    @doc """
    Segments an object in an image using SAM 2.

    Accepts an optional point or box prompt to select which object to
    segment. With no prompt, the centre of the image is used.

    ### Arguments

    * `image` is any `t:Vix.Vips.Image.t/0`.

    * `options` is a keyword list of options.

    ### Options

    * `:prompt` selects what to segment:
      * `:auto` — segment the object at the image centre (default).
      * `{:point, x, y}` — segment the object at pixel `(x, y)`.
      * `{:box, x, y, w, h}` — segment the object inside the box.
      * A list of `{:point, x, y}` tuples for multi-point prompting.

    * `:multimask` — when `true`, returns all three SAM candidate masks
      as a list sorted by descending score. When `false` (default),
      returns only the best mask as a single `t:mask_result/0`.

    * `:min_score` — minimum IoU score to return when `:multimask` is
      `true`. The default is `0.0`.

    * `:repo` — HuggingFace repo for the SAM 2 ONNX models. Default
      is `"SharpAI/sam2-hiera-tiny-onnx"`.

    * `:encoder_file` — encoder ONNX filename within the repo. Default
      is `"encoder.onnx"`.

    * `:decoder_file` — decoder ONNX filename within the repo. Default
      is `"decoder.onnx"`.

    ### Returns

    * A `t:mask_result/0` map when `:multimask` is `false`.

    * A list of `t:mask_result/0` maps sorted by descending `:score`
      when `:multimask` is `true`.

    ### Examples

        iex> image = Image.open!("./test/support/images/puppy.webp")
        iex> %{score: score, mask: mask} = Image.Segmentation.segment(image)
        iex> score > 0.0
        true
        iex> match?(%Vix.Vips.Image{}, mask)
        true

    """
    @spec segment(Vimage.t(), Keyword.t()) :: mask_result() | [mask_result()]
    def segment(%Vimage{} = image, options \\ []) do
      prompt = Keyword.get(options, :prompt, :auto)
      multimask = Keyword.get(options, :multimask, false)
      min_score = Keyword.get(options, :min_score, 0.0)
      repo = Keyword.get(options, :repo, @sam_repo)
      encoder_file = Keyword.get(options, :encoder_file, @sam_encoder_file)
      decoder_file = Keyword.get(options, :decoder_file, @sam_decoder_file)

      encoder = load_model(repo, encoder_file)
      decoder = load_model(repo, decoder_file)

      {preprocessed, resized_w, resized_h} = sam_preprocess(image)
      {image_embed, high_res_feats_0, high_res_feats_1} = sam_encode(encoder, preprocessed)

      orig_w = Image.width(image)
      orig_h = Image.height(image)
      input_scale = max(orig_w, orig_h) / @sam_input_size

      {point_coords, point_labels} =
        encode_sam_prompt(prompt, orig_w, orig_h, input_scale)

      {masks_raw, iou_preds} =
        sam_decode(
          decoder,
          image_embed,
          high_res_feats_0,
          high_res_feats_1,
          point_coords,
          point_labels
        )

      # masks_raw: [1, 3, 256, 256]; upscale each to original image size
      if multimask do
        Enum.map(0..2, fn i ->
          score = masks_raw |> then(fn _ -> Nx.to_number(iou_preds[0][i]) end)
          mask_tensor = masks_raw[0][i]
          mask = sam_mask_to_image(mask_tensor, orig_w, orig_h, resized_w, resized_h)
          %{score: score, mask: mask}
        end)
        |> Enum.filter(fn %{score: s} -> s >= min_score end)
        |> Enum.sort_by(& &1.score, :desc)
      else
        best_idx = iou_preds[0] |> Nx.argmax() |> Nx.to_number()
        score = Nx.to_number(iou_preds[0][best_idx])
        mask_tensor = masks_raw[0][best_idx]
        mask = sam_mask_to_image(mask_tensor, orig_w, orig_h, resized_w, resized_h)
        %{score: score, mask: mask}
      end
    end

    # --- Public API: segment_panoptic/2 (DETR-panoptic) ----------------

    @doc """
    Segments and labels every region in an image using DETR-panoptic.

    Returns one segment per detected object or region, each with a
    class label, confidence score, and a binary mask. Covers 250 COCO
    panoptic categories including everyday objects (`person`, `car`,
    `dog`) and background regions (`sky`, `road`, `grass`).

    ### Arguments

    * `image` is any `t:Vix.Vips.Image.t/0`.

    * `options` is a keyword list of options.

    ### Options

    * `:min_score` — minimum confidence score to include a segment.
      The default is `0.5`.

    * `:repo` — HuggingFace repo for the ONNX model. Default is
      `"Xenova/detr-resnet-50-panoptic"`.

    * `:model_file` — ONNX filename within the repo. Default is
      `"onnx/model.onnx"`. Use `"onnx/model_quantized.onnx"` (~44 MB)
      for a much smaller model with some accuracy loss.

    ### Returns

    * A list of `t:segment/0` maps sorted by descending `:score`.
      May be empty if no segment meets `:min_score`.

    ### Examples

        iex> image = Image.open!("./test/support/images/puppy.webp")
        iex> segments = Image.Segmentation.segment_panoptic(image)
        iex> is_list(segments)
        true
        iex> Enum.all?(segments, &match?(%{label: _, score: _, mask: _}, &1))
        true

    """
    @spec segment_panoptic(Vimage.t(), Keyword.t()) :: [segment()]
    def segment_panoptic(%Vimage{} = image, options \\ []) do
      min_score = Keyword.get(options, :min_score, @default_min_score)
      repo = Keyword.get(options, :repo, @detr_repo)
      model_file = Keyword.get(options, :model_file, @detr_model_file)

      model = load_model(repo, model_file)
      id2label = load_detr_labels(repo)

      orig_w = Image.width(image)
      orig_h = Image.height(image)

      {batch, _input_h, _input_w} = detr_preprocess(image)

      {logits, _pred_boxes, pred_masks} = Ortex.run(model, batch)
      logits = Nx.backend_transfer(logits, Nx.BinaryBackend)
      pred_masks = Nx.backend_transfer(pred_masks, Nx.BinaryBackend)

      detr_postprocess(logits, pred_masks, id2label,
        orig_w: orig_w,
        orig_h: orig_h,
        min_score: min_score
      )
    end

    # --- Public API: helpers --------------------------------------------

    @doc """
    Applies a mask as the alpha channel of an image.

    White pixels in the mask become fully opaque; black pixels become
    fully transparent. The result is an RGBA image suitable for
    compositing or exporting with transparency.

    ### Arguments

    * `image` is any `t:Vix.Vips.Image.t/0`.

    * `mask` is a single-band `t:Vix.Vips.Image.t/0` of the same
      dimensions, such as the `:mask` field of `t:mask_result/0` or
      `t:segment/0`.

    ### Returns

    * `{:ok, image}` — an RGBA `t:Vix.Vips.Image.t/0`, or

    * `{:error, reason}`.

    ### Examples

        iex> image = Image.open!("./test/support/images/puppy.webp")
        iex> %{mask: mask} = Image.Segmentation.segment(image)
        iex> {:ok, cutout} = Image.Segmentation.apply_mask(image, mask)
        iex> Image.bands(cutout)
        4

    """
    @spec apply_mask(Vimage.t(), Vimage.t()) ::
            {:ok, Vimage.t()} | {:error, Image.error()}
    def apply_mask(%Vimage{} = image, %Vimage{} = mask) do
      with {:ok, flat} <- Image.flatten(image),
           {:ok, srgb} <- Image.to_colorspace(flat, :srgb) do
        Image.add_alpha(srgb, mask)
      end
    end

    @doc """
    Overlays colour-coded segment masks on an image.

    Each segment gets a distinct colour. Useful for visualising the
    output of `segment_panoptic/2`.

    ### Arguments

    * `image` is any `t:Vix.Vips.Image.t/0`.

    * `segments` is the list returned by `segment_panoptic/2`, or any
      list of maps with `:mask` and `:label` keys.

    * `options` is a keyword list of options.

    ### Options

    * `:alpha` — opacity of the overlay as a float in `[0.0, 1.0]`.
      The default is `0.5`.

    ### Returns

    * The annotated `t:Vix.Vips.Image.t/0`.

    ### Examples

        iex> image = Image.open!("./test/support/images/puppy.webp")
        iex> segments = Image.Segmentation.segment_panoptic(image)
        iex> overlay = Image.Segmentation.compose_overlay(image, segments)
        iex> match?(%Vix.Vips.Image{}, overlay)
        true

    """
    @spec compose_overlay(Vimage.t(), [segment() | mask_result()], Keyword.t()) ::
            Vimage.t()
    @dialyzer {:nowarn_function, {:compose_overlay, 3}}
    def compose_overlay(%Vimage{} = image, segments, options \\ []) do
      alpha = Keyword.get(options, :alpha, 0.5)
      opacity = round(alpha * 255)

      overlay_colors = [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 165, 0],
        [128, 0, 128],
        [0, 255, 255],
        [255, 20, 147],
        [0, 128, 0]
      ]

      segments
      |> Enum.with_index()
      |> Enum.reduce(image, fn {segment, index}, acc ->
        mask = Map.fetch!(segment, :mask)
        [r, g, b] = Enum.at(overlay_colors, rem(index, length(overlay_colors)))

        case build_color_overlay(mask, r, g, b, opacity, acc) do
          {:ok, composite} -> composite
          {:error, _} -> acc
        end
      end)
    end

    # --- Private: model loading -----------------------------------------

    defp load_model(repo, filename) do
      key = {__MODULE__, repo, filename}

      case :persistent_term.get(key, nil) do
        nil ->
          path = ImageVision.ModelCache.fetch!(repo, filename)
          model = Ortex.load(path)
          :persistent_term.put(key, model)
          model

        model ->
          model
      end
    end

    # --- Private: SAM 2 pre/encode/decode -------------------------------

    # Resize so longest edge = @sam_input_size, embed in square canvas
    # at top-left. Returns {tensor [1,3,1024,1024], resized_w, resized_h}.
    defp sam_preprocess(%Vimage{} = image) do
      flat = image |> Image.flatten!() |> Image.to_colorspace!(:srgb)
      resized = Image.thumbnail!(flat, @sam_input_size)
      resized_w = Image.width(resized)
      resized_h = Image.height(resized)
      padded = Image.embed!(resized, @sam_input_size, @sam_input_size, x: 0, y: 0)

      tensor =
        padded
        |> Image.to_nx!(backend: Nx.BinaryBackend)
        |> Nx.as_type(:f32)
        |> Nx.divide(255.0)
        |> NxImage.normalize(Nx.tensor(@imagenet_mean), Nx.tensor(@imagenet_std))
        |> Nx.transpose(axes: [2, 0, 1])
        |> Nx.new_axis(0)

      {tensor, resized_w, resized_h}
    end

    defp sam_encode(encoder, tensor) do
      # Model outputs in order: high_res_feats_0, high_res_feats_1, image_embed
      {high_res_feats_0, high_res_feats_1, image_embed} = Ortex.run(encoder, tensor)
      {image_embed, high_res_feats_0, high_res_feats_1}
    end

    # Converts a user prompt into SAM point_coords + point_labels tensors.
    # Coords are in the 1024×1024 input space.
    defp encode_sam_prompt(:auto, orig_w, orig_h, input_scale) do
      cx = orig_w / 2
      cy = orig_h / 2
      encode_sam_prompt({:point, cx, cy}, orig_w, orig_h, input_scale)
    end

    defp encode_sam_prompt({:point, x, y}, _orig_w, _orig_h, scale) do
      coords = Nx.tensor([[[x / scale, y / scale]]], type: :f32)
      labels = Nx.tensor([[1]], type: :f32)
      {coords, labels}
    end

    defp encode_sam_prompt({:box, x, y, w, h}, _orig_w, _orig_h, scale) do
      # Box uses two corner points with labels 2 (top-left) and 3 (bottom-right)
      coords =
        Nx.tensor(
          [[[x / scale, y / scale], [(x + w) / scale, (y + h) / scale]]],
          type: :f32
        )

      labels = Nx.tensor([[2, 3]], type: :f32)
      {coords, labels}
    end

    defp encode_sam_prompt(points, _orig_w, _orig_h, scale) when is_list(points) do
      coords_list = Enum.map(points, fn {:point, x, y} -> [x / scale, y / scale] end)
      labels_list = List.duplicate(1, length(points))
      coords = Nx.tensor([coords_list], type: :f32)
      labels = Nx.tensor([labels_list], type: :f32)
      {coords, labels}
    end

    defp sam_decode(
           decoder,
           image_embed,
           high_res_feats_0,
           high_res_feats_1,
           point_coords,
           point_labels
         ) do
      mask_input = Nx.broadcast(Nx.tensor(0, type: :f32), {1, 1, 256, 256})
      has_mask = Nx.tensor([0], type: :f32)

      {masks, iou_predictions} =
        Ortex.run(decoder, {
          image_embed,
          high_res_feats_0,
          high_res_feats_1,
          point_coords,
          point_labels,
          mask_input,
          has_mask
        })

      {
        Nx.backend_transfer(masks, Nx.BinaryBackend),
        Nx.backend_transfer(iou_predictions, Nx.BinaryBackend)
      }
    end

    # Converts a SAM output mask tensor {256, 256} (logits) to a
    # single-band Vimage at original image dimensions.
    defp sam_mask_to_image(mask_tensor, orig_w, orig_h, resized_w, resized_h) do
      # The 256×256 mask covers the full 1024×1024 padded input space.
      # Crop to the valid region, which is resized_w×resized_h in input
      # space, scaled to mask space by the factor 256/1024 = 1/4.
      valid_h = round(resized_h * 256 / @sam_input_size)
      valid_w = round(resized_w * 256 / @sam_input_size)

      binary =
        mask_tensor
        |> Nx.slice([0, 0], [valid_h, valid_w])
        |> Nx.greater(0)
        |> Nx.multiply(255)
        |> Nx.as_type(:u8)
        |> Nx.new_axis(2)
        |> Nx.rename([:height, :width, :bands])

      binary
      |> Image.from_nx!()
      |> Image.resize!(orig_w / valid_w, vertical_scale: orig_h / valid_h)
    end

    # --- Private: DETR-panoptic pre/post --------------------------------

    # Resize so short side = @detr_short_side, preserving aspect ratio.
    # Returns {batch_tensor, input_h, input_w}.
    defp detr_preprocess(%Vimage{} = image) do
      flat = image |> Image.flatten!() |> Image.to_colorspace!(:srgb)

      orig_w = Image.width(flat)
      orig_h = Image.height(flat)

      scale = @detr_short_side / min(orig_w, orig_h)
      target_w = round(orig_w * scale)
      target_h = round(orig_h * scale)

      resized = Image.thumbnail!(flat, target_w, height: target_h)
      input_w = Image.width(resized)
      input_h = Image.height(resized)

      tensor =
        resized
        |> Image.to_nx!(backend: Nx.BinaryBackend)
        |> Nx.as_type(:f32)
        |> Nx.divide(255.0)
        |> NxImage.normalize(Nx.tensor(@imagenet_mean), Nx.tensor(@imagenet_std))
        |> Nx.transpose(axes: [2, 0, 1])
        |> Nx.new_axis(0)

      pixel_mask = Nx.broadcast(Nx.tensor(1, type: :s64), {1, 64, 64})
      {{tensor, pixel_mask}, input_h, input_w}
    end

    # Resolves a class index to a human-readable label, preferring the
    # repo's id2label and falling back to the canonical COCO panoptic
    # map when the repo carries a `LABEL_n` placeholder.
    defp lookup_panoptic_label(id2label, class_idx) do
      case Map.get(id2label, to_string(class_idx)) do
        nil -> Map.get(@coco_panoptic_id2label, class_idx, "class_#{class_idx}")
        "LABEL_" <> _ -> Map.get(@coco_panoptic_id2label, class_idx, "class_#{class_idx}")
        name -> name
      end
    end

    # Loads id2label from config.json; cached in :persistent_term.
    defp load_detr_labels(repo) do
      key = {__MODULE__, :labels, repo}

      case :persistent_term.get(key, nil) do
        nil ->
          path = ImageVision.ModelCache.fetch!(repo, @detr_config_file)
          {:ok, raw} = File.read(path)
          config = :json.decode(raw)
          labels = Map.get(config, "id2label", %{})
          :persistent_term.put(key, labels)
          labels

        labels ->
          labels
      end
    end

    # Converts raw DETR-panoptic outputs into a list of segments.
    defp detr_postprocess(logits, pred_masks, id2label, options) do
      min_score = Keyword.fetch!(options, :min_score)
      orig_w = Keyword.fetch!(options, :orig_w)
      orig_h = Keyword.fetch!(options, :orig_h)

      # logits: [1, 100, 251] → numerically-stable softmax, pick top class per query
      query_logits = logits[0]

      scores_per_query =
        query_logits
        |> then(fn x ->
          shifted = Nx.subtract(x, Nx.reduce_max(x, axes: [-1], keep_axes: true))
          exp_x = Nx.exp(shifted)
          Nx.divide(exp_x, Nx.sum(exp_x, axes: [-1], keep_axes: true))
        end)

      best_class = Nx.argmax(scores_per_query, axis: -1)
      best_score = Nx.reduce_max(scores_per_query, axes: [-1])

      class_list = Nx.to_flat_list(best_class)
      score_list = Nx.to_flat_list(best_score)

      # pred_masks: [1, queries, H/4, W/4]
      {1, _num_queries, mask_h, mask_w} = Nx.shape(pred_masks)

      Enum.zip(class_list, score_list)
      |> Enum.with_index()
      |> Enum.flat_map(fn {{class_idx, score}, query_idx} ->
        if class_idx != @detr_no_object_class and score >= min_score do
          label = lookup_panoptic_label(id2label, class_idx)
          mask_tensor = pred_masks[0][query_idx]

          mask =
            mask_tensor
            |> Nx.sigmoid()
            |> Nx.greater(0.5)
            |> Nx.multiply(255)
            |> Nx.as_type(:u8)
            |> Nx.new_axis(2)
            |> Nx.rename([:height, :width, :bands])
            |> Image.from_nx!()
            |> Image.resize!(orig_w / mask_w, vertical_scale: orig_h / mask_h)

          [%{label: label, score: score, mask: mask}]
        else
          []
        end
      end)
      |> Enum.sort_by(& &1.score, :desc)
    end

    # --- Private: overlay helper ----------------------------------------

    # Builds a coloured translucent overlay for one mask and composites
    # it onto the base image.
    defp build_color_overlay(%Vimage{} = mask, r, g, b, opacity, %Vimage{} = base) do
      w = Image.width(base)
      h = Image.height(base)

      # Scale mask values by the desired opacity fraction (linear: out = in * a + b).
      with {:ok, scaled_alpha} <-
             Vix.Vips.Operation.linear(mask, [opacity / 255.0], [0.0]),
           {:ok, color_layer} <- Image.new(w, h, color: [r, g, b]),
           {:ok, overlay} <- Image.add_alpha(color_layer, scaled_alpha) do
        Image.compose(base, overlay)
      end
    end
  end
end
