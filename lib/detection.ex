if ImageVision.ortex_configured?() do
  defmodule Image.Detection do
    @moduledoc """
    Object detection — where are the objects in this image?

    Pass a `t:Vix.Vips.Image.t/0` to `detect/2` and get back a list
    of detected objects with their class labels, confidence scores,
    and bounding boxes.

    ## Quick start

        iex> car = Image.open!("./test/support/images/lamborghini-forsennato-concept.jpg")
        iex> [%{label: _, score: _, box: _} | _] = Image.Detection.detect(car)

    ## Default model

    The default is [RT-DETR](https://huggingface.co/onnx-community/rtdetr_r50vd) —
    a real-time, transformer-based detector that beats YOLOv8 on
    COCO and is **Apache 2.0 licensed** (unlike YOLOv8/11 which are
    AGPL). The ONNX export is hosted at
    `onnx-community/rtdetr_r50vd` and is downloaded on first call
    via `ImageVision.ModelCache`.

    * Model: `onnx-community/rtdetr_r50vd` / `onnx/model.onnx` (~175 MB).

    * Classes: 80 standard COCO classes (`person`, `bicycle`, `car`, …).

    * Output: per-query class scores (sigmoid) and `cxcywh` bounding
      boxes. RT-DETR is NMS-free by design — no Non-Maximum
      Suppression post-processing is required.

    ## Drawing detections

    Use `draw_bbox_with_labels/2` to overlay detections on the
    original image:

        image
        |> Image.Detection.detect()
        |> Image.Detection.draw_bbox_with_labels(image)

    ## Optional dependency

    This module is only available when [Ortex](https://hex.pm/packages/ortex)
    is configured in your application's `mix.exs`.

    """

    alias Vix.Vips.Image, as: Vimage

    @default_repo "onnx-community/rtdetr_r50vd"
    @default_filename "onnx/model.onnx"

    @input_size 640
    @num_classes 80
    @default_min_score 0.5

    # Standard 80-class COCO labels in the order produced by the
    # HuggingFace RT-DETR `id2label` map. Baked at compile time so
    # the module works without `priv/` lookups.
    @coco_classes ~w(
      person bicycle car motorcycle airplane bus train truck boat
      traffic_light fire_hydrant stop_sign parking_meter bench bird
      cat dog horse sheep cow elephant bear zebra giraffe backpack
      umbrella handbag tie suitcase frisbee skis snowboard sports_ball
      kite baseball_bat baseball_glove skateboard surfboard tennis_racket
      bottle wine_glass cup fork knife spoon bowl banana apple sandwich
      orange broccoli carrot hot_dog pizza donut cake chair couch
      potted_plant bed dining_table toilet tv laptop mouse remote
      keyboard cell_phone microwave oven toaster sink refrigerator
      book clock vase scissors teddy_bear hair_drier toothbrush
    )
                  |> Enum.map(&String.replace(&1, "_", " "))

    @typedoc """
    A single detected object.

    * `:label` is one of the 80 COCO class names, e.g. `"person"`.

    * `:score` is the confidence score, a float in `[0.0, 1.0]`.

    * `:box` is `{x, y, width, height}` in pixel coordinates of the
      original image. `(x, y)` is the top-left corner.

    """
    @type detection :: %{
            label: String.t(),
            score: float(),
            box: {non_neg_integer(), non_neg_integer(), pos_integer(), pos_integer()}
          }

    @doc """
    Detects objects in an image and returns a list of detections
    sorted by descending confidence.

    ### Arguments

    * `image` is any `t:Vix.Vips.Image.t/0`.

    * `options` is a keyword list of options.

    ### Options

    * `:min_score` is the minimum confidence score, a float in
      `[0.0, 1.0]`, that a detection must meet to be returned. The
      default is `0.5`.

    * `:repo` is the HuggingFace repository for the model. The
      default is `"onnx-community/rtdetr_r50vd"`.

    * `:filename` is the ONNX file path within the repository. The
      default is `"onnx/model.onnx"`. Use `"onnx/model_quantized.onnx"`
      (~45 MB INT8) for a much smaller model with some accuracy loss.

    ### Returns

    * A list of `t:detection/0` maps, sorted by descending `:score`.

    ### Examples

        iex> car = Image.open!("./test/support/images/lamborghini-forsennato-concept.jpg")
        iex> [%{label: _, score: _, box: _} | _] =
        ...>   Image.Detection.detect(car, min_score: 0.5)

    """
    @spec detect(image :: Vimage.t(), options :: Keyword.t()) :: [detection()]
    def detect(%Vimage{} = image, options \\ []) do
      min_score = Keyword.get(options, :min_score, @default_min_score)
      repo = Keyword.get(options, :repo, @default_repo)
      filename = Keyword.get(options, :filename, @default_filename)

      model = load_model(repo, filename)

      {batch, scale_x, scale_y} = preprocess(image)

      {logits, pred_boxes} = Ortex.run(model, batch)

      postprocess(logits, pred_boxes, scale_x, scale_y,
        original_width: Image.width(image),
        original_height: Image.height(image),
        min_score: min_score
      )
    end

    @doc """
    Draws bounding boxes with class labels onto an image.

    Builds an SVG overlay — one box and label per detection — and
    composites it onto the image. Each distinct class label gets a
    consistent colour so multiple detections of the same class are
    easy to identify at a glance.

    ### Arguments

    * `detections` is the list returned from `detect/2`.

    * `image` is the image upon which detection was run.

    * `options` is a keyword list of options. Currently unused;
      accepted for forward compatibility.

    ### Returns

    * The annotated `t:Vix.Vips.Image.t/0`.

    ### Examples

        iex> car = Image.open!("./test/support/images/lamborghini-forsennato-concept.jpg")
        iex> annotated =
        ...>   car
        ...>   |> Image.Detection.detect()
        ...>   |> Image.Detection.draw_bbox_with_labels(car)
        iex> match?(%Vix.Vips.Image{}, annotated)
        true

    """
    @spec draw_bbox_with_labels([detection()], Vimage.t(), Keyword.t()) :: Vimage.t()
    def draw_bbox_with_labels(detections, %Vimage{} = image, _options \\ []) do
      width = Image.width(image)
      height = Image.height(image)

      palette = ~w(
        #e6194b #3cb44b #4363d8 #f58231 #911eb4
        #42d4f4 #f032e6 #bfef45 #469990 #9a6324
      )

      label_colors =
        detections
        |> Enum.map(& &1.label)
        |> Enum.uniq()
        |> Enum.zip(Stream.cycle(palette))
        |> Map.new()

      boxes_svg =
        Enum.map(detections, fn %{label: label, score: score, box: {x, y, w, h}} ->
          color = Map.fetch!(label_colors, label)
          text = "#{label} #{Float.round(score * 100, 1)}%"
          label_y = max(0, y - 20)
          label_w = String.length(text) * 8 + 8

          """
          <rect x="#{x}" y="#{y}" width="#{w}" height="#{h}"
                fill="none" stroke="#{color}" stroke-width="2"/>
          <rect x="#{x}" y="#{label_y}" width="#{label_w}" height="20" fill="#{color}"/>
          <text x="#{x + 4}" y="#{label_y + 14}"
                font-family="sans-serif" font-size="13" font-weight="bold" fill="white">#{text}</text>
          """
        end)
        |> Enum.join()

      svg = """
      <svg xmlns="http://www.w3.org/2000/svg" width="#{width}" height="#{height}">
        #{boxes_svg}
      </svg>
      """

      {:ok, overlay} = Image.open(svg, access: :sequential)
      Image.compose!(image, overlay)
    end

    @doc """
    Returns the list of class labels the default model can detect.

    ### Returns

    * A list of 80 COCO class names as binaries, in the order used
      by RT-DETR.

    """
    @spec classes() :: [String.t()]
    def classes do
      @coco_classes
    end

    # --- Internal: model loading + preprocessing + post-processing ---

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

    # Prepares an image for RT-DETR: resize so the longest edge is
    # @input_size, pad with black to @input_size × @input_size in the
    # top-left, normalize with ImageNet mean/std, transpose to CHW,
    # add a batch axis. Returns the tensor plus the per-axis scale
    # factor needed to map predictions back to the original image.

    defp preprocess(%Vimage{} = image) do
      flattened =
        image
        |> Image.flatten!()
        |> Image.to_colorspace!(:srgb)

      resized = Image.thumbnail!(flattened, @input_size)
      resized_width = Image.width(resized)
      resized_height = Image.height(resized)

      padded = Image.embed!(resized, @input_size, @input_size, x: 0, y: 0)

      tensor =
        padded
        |> Image.to_nx!(backend: Nx.BinaryBackend)
        |> Nx.as_type(:f32)
        |> Nx.divide(255.0)
        |> Nx.transpose(axes: [2, 0, 1])
        |> Nx.new_axis(0)

      scale_x = Image.width(image) / resized_width
      scale_y = Image.height(image) / resized_height

      {tensor, scale_x, scale_y}
    end

    # Converts the raw RT-DETR outputs to a list of detections in
    # original-image pixel coordinates. RT-DETR emits @num_queries
    # query slots; each has @num_classes per-class logits (apply
    # sigmoid for scores) and a `cxcywh` bounding box normalized to
    # the model input. We pick the top class per query, threshold
    # by `:min_score`, scale boxes back to the original image, and
    # return them sorted by descending score.

    defp postprocess(logits, pred_boxes, scale_x, scale_y, options) do
      original_width = Keyword.fetch!(options, :original_width)
      original_height = Keyword.fetch!(options, :original_height)
      min_score = Keyword.fetch!(options, :min_score)

      scores =
        logits[0]
        |> Nx.backend_transfer(Nx.BinaryBackend)
        |> Nx.sigmoid()

      boxes = pred_boxes[0] |> Nx.backend_transfer(Nx.BinaryBackend)

      best_class = Nx.argmax(scores, axis: 1)
      best_score = Nx.reduce_max(scores, axes: [1])

      best_class_list = Nx.to_flat_list(best_class)
      best_score_list = Nx.to_flat_list(best_score)
      boxes_list = Nx.to_list(boxes)

      Enum.zip([best_class_list, best_score_list, boxes_list])
      |> Enum.with_index()
      |> Enum.flat_map(fn {{class_idx, score, [cx, cy, w, h]}, _q} ->
        if score >= min_score and class_idx >= 0 and class_idx < @num_classes do
          box =
            cxcywh_to_xywh(cx, cy, w, h,
              scale_x: scale_x,
              scale_y: scale_y,
              max_width: original_width,
              max_height: original_height
            )

          case box do
            nil ->
              []

            valid_box ->
              [%{label: Enum.at(@coco_classes, class_idx), score: score, box: valid_box}]
          end
        else
          []
        end
      end)
      |> Enum.sort_by(& &1.score, :desc)
    end

    # Converts a normalized `cxcywh` box (each coord in [0, 1] over
    # the model's @input_size × @input_size input) into `{x, y, w, h}`
    # pixel coordinates on the original image. Clips to image bounds
    # and returns nil if the result has zero area.

    defp cxcywh_to_xywh(cx, cy, w, h, options) do
      scale_x = Keyword.fetch!(options, :scale_x)
      scale_y = Keyword.fetch!(options, :scale_y)
      max_width = Keyword.fetch!(options, :max_width)
      max_height = Keyword.fetch!(options, :max_height)

      input_size = @input_size

      cx_px = cx * input_size * scale_x
      cy_px = cy * input_size * scale_y
      w_px = w * input_size * scale_x
      h_px = h * input_size * scale_y

      x1 = max(0, round(cx_px - w_px / 2))
      y1 = max(0, round(cy_px - h_px / 2))
      x2 = min(max_width, round(cx_px + w_px / 2))
      y2 = min(max_height, round(cy_px + h_px / 2))

      width = x2 - x1
      height = y2 - y1

      if width > 0 and height > 0 do
        {x1, y1, width, height}
      else
        nil
      end
    end
  end
end
