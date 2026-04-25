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
      input_scale = @sam_input_size / max(orig_w, orig_h)

      {point_coords, point_labels} =
        encode_sam_prompt(prompt, orig_w, orig_h, input_scale)

      {masks_raw, iou_preds} =
        sam_decode(decoder, image_embed, high_res_feats_0, high_res_feats_1,
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

      {batch, input_h, input_w} = detr_preprocess(image)

      {logits, _pred_boxes, pred_masks} = Ortex.run(model, batch)

      detr_postprocess(logits, pred_masks, id2label,
        input_h: input_h,
        input_w: input_w,
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
        |> Image.to_nx!()
        |> Nx.as_type(:f32)
        |> Nx.divide(255.0)
        |> NxImage.normalize(Nx.tensor(@imagenet_mean), Nx.tensor(@imagenet_std))
        |> Nx.transpose(axes: [2, 0, 1])
        |> Nx.new_axis(0)

      {tensor, resized_w, resized_h}
    end

    defp sam_encode(encoder, tensor) do
      {image_embed, high_res_feats_0, high_res_feats_1} = Ortex.run(encoder, tensor)
      {image_embed, high_res_feats_0, high_res_feats_1}
    end

    # Converts a user prompt into SAM point_coords + point_labels tensors.
    # Coords are in the 1024×1024 input space.
    defp encode_sam_prompt(:auto, orig_w, orig_h, _input_scale) do
      cx = orig_w / 2
      cy = orig_h / 2
      encode_sam_prompt({:point, cx, cy}, orig_w, orig_h, orig_w / @sam_input_size)
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

    defp sam_decode(decoder, image_embed, high_res_feats_0, high_res_feats_1,
           point_coords,
           point_labels
         ) do
      mask_input = Nx.broadcast(Nx.tensor(0, type: :f32), {1, 1, 256, 256})
      has_mask = Nx.tensor([0], type: :f32)

      {masks, iou_predictions, _low_res} =
        Ortex.run(decoder, {
          image_embed,
          high_res_feats_0,
          high_res_feats_1,
          point_coords,
          point_labels,
          mask_input,
          has_mask
        })

      {masks, iou_predictions}
    end

    # Converts a SAM output mask tensor [256, 256] (logits) to a
    # single-band Vimage at original image dimensions.
    defp sam_mask_to_image(mask_tensor, orig_w, orig_h, resized_w, resized_h) do
      # Up to the padded 1024×1024 input, then crop to the valid region,
      # then resize to original dimensions.
      upscaled =
        mask_tensor
        |> Nx.reshape({1024, 1024, 1})
        |> Nx.slice([0, 0, 0], [resized_h, resized_w, 1])

      # Threshold at 0 (logits > 0 → object)
      binary =
        upscaled
        |> Nx.greater(0)
        |> Nx.multiply(255)
        |> Nx.as_type(:u8)

      binary
      |> Image.from_nx!()
      |> Image.resize!(orig_w / resized_w, vertical_scale: orig_h / resized_h)
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
        |> Image.to_nx!()
        |> Nx.as_type(:f32)
        |> Nx.divide(255.0)
        |> NxImage.normalize(Nx.tensor(@imagenet_mean), Nx.tensor(@imagenet_std))
        |> Nx.transpose(axes: [2, 0, 1])
        |> Nx.new_axis(0)

      {tensor, input_h, input_w}
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
    defp detr_postprocess(logits, pred_masks, id2label, opts) do
      min_score = Keyword.fetch!(opts, :min_score)
      input_h = Keyword.fetch!(opts, :input_h)
      input_w = Keyword.fetch!(opts, :input_w)
      orig_w = Keyword.fetch!(opts, :orig_w)
      orig_h = Keyword.fetch!(opts, :orig_h)

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

      # pred_masks: [1, 100, H/4, W/4]
      mask_h = div(input_h, 4)
      mask_w = div(input_w, 4)

      Enum.zip(class_list, score_list)
      |> Enum.with_index()
      |> Enum.flat_map(fn {{class_idx, score}, query_idx} ->
        if class_idx != @detr_no_object_class and score >= min_score do
          label = Map.get(id2label, to_string(class_idx), "class_#{class_idx}")
          mask_tensor = pred_masks[0][query_idx]

          mask =
            mask_tensor
            |> Nx.reshape({mask_h, mask_w, 1})
            |> Nx.sigmoid()
            |> Nx.greater(0.5)
            |> Nx.multiply(255)
            |> Nx.as_type(:u8)
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
