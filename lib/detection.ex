if ImageVision.ortex_configured?() do
  defmodule Image.Detection do
    @moduledoc """
    Object detection — where are the objects in this image?

    Pass a `t:Vix.Vips.Image.t/0` to `detect/2` and get back a list
    of detected objects with their class labels, confidence scores,
    and bounding boxes.

    ## Quick start

        iex> street = Image.open!("./test/support/images/street.jpg")
        iex> [%{label: _, score: _, box: _} | _] = Image.Detection.detect(street)

    ## Default model

    The default is [RT-DETR](https://huggingface.co/PekingU/rtdetr_r50vd) —
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

        iex> street = Image.open!("./test/support/images/street.jpg")
        iex> [%{label: _, score: _, box: _} | _] =
        ...>   Image.Detection.detect(street, min_score: 0.5)

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

    ### Arguments

    * `detections` is the list returned from `detect/2`.

    * `image` is the image upon which detection was run.

    ### Options

    * `:stroke_color` is the box outline color. The default is `:red`.

    * `:stroke_width` is the box outline thickness in pixels. The
      default is `4`.

    * `:font_size` is the label font size. The default is `20`.

    ### Returns

    * The annotated `t:Vix.Vips.Image.t/0`.

    ### Examples

        iex> street = Image.open!("./test/support/images/street.jpg")
        iex> annotated =
        ...>   street
        ...>   |> Image.Detection.detect()
        ...>   |> Image.Detection.draw_bbox_with_labels(street)
        iex> match?(%Vix.Vips.Image{}, annotated)
        true

    """
    @spec draw_bbox_with_labels([detection()], Vimage.t(), Keyword.t()) :: Vimage.t()
    def draw_bbox_with_labels(detections, %Vimage{} = image, options \\ []) do
      stroke_color = Keyword.get(options, :stroke_color, :red)
      stroke_width = Keyword.get(options, :stroke_width, 4)
      font_size = Keyword.get(options, :font_size, 20)

      {width, height, _bands} = Image.shape(image)

      Enum.reduce(detections, image, fn %{label: label, box: {x, y, w, h}}, acc ->
        {:ok, box_image} =
          Image.Shape.rect(w, h, stroke_color: stroke_color, stroke_width: stroke_width)

        {:ok, text_image} =
          Image.Text.text(label, text_fill_color: stroke_color, font_size: font_size)

        acc
        |> Image.compose!(box_image, x: x, y: y)
        |> Image.compose!(text_image,
          x: min(max(x, 0), width) + 5,
          y: min(max(y, 0), height) + 5
        )
      end)
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

      mean = Nx.tensor([0.485, 0.456, 0.406])
      std = Nx.tensor([0.229, 0.224, 0.225])

      tensor =
        padded
        |> Image.to_nx!()
        |> Nx.as_type(:f32)
        |> Nx.divide(255.0)
        |> NxImage.normalize(mean, std)
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

    defp postprocess(logits, pred_boxes, scale_x, scale_y, opts) do
      original_width = Keyword.fetch!(opts, :original_width)
      original_height = Keyword.fetch!(opts, :original_height)
      min_score = Keyword.fetch!(opts, :min_score)

      scores = Nx.sigmoid(logits[0])
      boxes = pred_boxes[0]

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

    defp cxcywh_to_xywh(cx, cy, w, h, opts) do
      scale_x = Keyword.fetch!(opts, :scale_x)
      scale_y = Keyword.fetch!(opts, :scale_y)
      max_width = Keyword.fetch!(opts, :max_width)
      max_height = Keyword.fetch!(opts, :max_height)

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
