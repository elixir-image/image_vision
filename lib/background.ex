if ImageVision.ortex_configured?() do
  defmodule Image.Background do
    @moduledoc """
    Foreground/background separation — "remove the background from
    this image".

    Two entry points cover different use cases:

    * `remove/2` — returns the input image with the foreground mask
      applied as an alpha channel. The background becomes
      transparent. Useful when you want a cutout you can drop onto
      another background or save as a transparent PNG.

    * `mask/2` — returns the foreground mask itself as a single-band
      `t:Vix.Vips.Image.t/0` (white = foreground, black = background).
      Useful when you want to do your own compositing.

    This is class-agnostic salient-object detection: the model
    decides what is "the subject" of the image and separates it from
    the rest. For a class-labeled segmentation use
    `Image.Segmentation.segment_panoptic/2` instead; for promptable
    per-object segmentation use `Image.Segmentation.segment/2`.

    ## Quick start

        iex> image = Image.open!("./test/support/images/puppy.webp")
        iex> {:ok, cutout} = Image.Background.remove(image)

    ## Default model

    [BiRefNet lite](https://huggingface.co/onnx-community/BiRefNet_lite-ONNX)
    — MIT licensed, ~210 MB. Current state-of-the-art for salient
    object detection / dichotomous image segmentation, distilled into
    a smaller variant suitable for general use.

    Override via `:repo` and `:model_file` to use the full BiRefNet
    or another compatible export.

    ## Optional dependency

    This module is only available when [Ortex](https://hex.pm/packages/ortex)
    is configured in your application's `mix.exs`.

    """

    alias Vix.Vips.Image, as: Vimage

    @default_repo "onnx-community/BiRefNet_lite-ONNX"
    @default_model_file "onnx/model.onnx"

    # BiRefNet expects 1024×1024 RGB input.
    @input_size 1024

    # ImageNet normalisation (BiRefNet uses standard ImageNet stats).
    @imagenet_mean [0.485, 0.456, 0.406]
    @imagenet_std [0.229, 0.224, 0.225]

    @doc """
    Removes the background from an image.

    Computes a foreground mask and returns the input image with that
    mask applied as the alpha channel. Background pixels become
    transparent.

    ### Arguments

    * `image` is any `t:Vix.Vips.Image.t/0`.

    * `options` is a keyword list of options.

    ### Options

    * `:repo` is the HuggingFace repository for the BiRefNet ONNX
      export. The default is `"onnx-community/BiRefNet_lite-ONNX"`.

    * `:model_file` is the ONNX filename within the repo. The default
      is `"onnx/model.onnx"`.

    ### Returns

    * `{:ok, cutout}` where `cutout` is a `t:Vix.Vips.Image.t/0`
      with the foreground only and a soft alpha channel reflecting
      the model's confidence at the boundaries, or

    * `{:error, reason}`.

    ### Examples

        iex> image = Image.open!("./test/support/images/puppy.webp")
        iex> {:ok, cutout} = Image.Background.remove(image)
        iex> Image.has_alpha?(cutout)
        true

    """
    @spec remove(Vimage.t(), Keyword.t()) :: {:ok, Vimage.t()} | {:error, Image.error()}
    def remove(%Vimage{} = image, options \\ []) do
      mask_image = mask(image, options)

      with {:ok, flat} <- Image.flatten(image),
           {:ok, srgb} <- Image.to_colorspace(flat, :srgb) do
        Image.add_alpha(srgb, mask_image)
      end
    end

    @doc """
    Computes a foreground mask for an image.

    Returns a single-band greyscale image at the original dimensions
    where pixel intensity reflects the model's confidence that the
    pixel belongs to the foreground.

    ### Arguments

    * `image` is any `t:Vix.Vips.Image.t/0`.

    * `options` is a keyword list of options.

    ### Options

    * `:repo` is the HuggingFace repository for the BiRefNet ONNX
      export. The default is `"onnx-community/BiRefNet_lite-ONNX"`.

    * `:model_file` is the ONNX filename within the repo. The default
      is `"onnx/model.onnx"`.

    ### Returns

    * A single-band `t:Vix.Vips.Image.t/0` at the input image's
      width and height. Pixel values are in `[0, 255]` where higher
      values are more confidently foreground.

    ### Examples

        iex> image = Image.open!("./test/support/images/puppy.webp")
        iex> mask = Image.Background.mask(image)
        iex> {Image.width(mask), Image.height(mask)} == {Image.width(image), Image.height(image)}
        true

    """
    @spec mask(Vimage.t(), Keyword.t()) :: Vimage.t()
    def mask(%Vimage{} = image, options \\ []) do
      repo = Keyword.get(options, :repo, @default_repo)
      model_file = Keyword.get(options, :model_file, @default_model_file)

      model = load_model(repo, model_file)

      orig_w = Image.width(image)
      orig_h = Image.height(image)

      tensor = preprocess(image)
      {logits} = Ortex.run(model, tensor)

      logits
      |> Nx.backend_transfer(Nx.BinaryBackend)
      |> postprocess(orig_w, orig_h)
    end

    # --- Private --------------------------------------------------------

    defp load_model(repo, model_file) do
      key = {__MODULE__, repo, model_file}

      case :persistent_term.get(key, nil) do
        nil ->
          path = ImageVision.ModelCache.fetch!(repo, model_file)
          model = Ortex.load(path)
          :persistent_term.put(key, model)
          model

        model ->
          model
      end
    end

    # Resize to @input_size × @input_size (ignoring aspect — BiRefNet is
    # trained this way), normalise with ImageNet stats, transpose to
    # NCHW, add batch dim. Returns an f32 tensor of shape [1, 3, 1024, 1024].
    defp preprocess(%Vimage{} = image) do
      flat = image |> Image.flatten!() |> Image.to_colorspace!(:srgb)

      {:ok, resized} =
        Vix.Vips.Operation.thumbnail_image(flat, @input_size,
          height: @input_size,
          size: :VIPS_SIZE_FORCE
        )

      resized
      |> Image.to_nx!(backend: Nx.BinaryBackend)
      |> Nx.as_type(:f32)
      |> Nx.divide(255.0)
      |> NxImage.normalize(Nx.tensor(@imagenet_mean), Nx.tensor(@imagenet_std))
      |> Nx.transpose(axes: [2, 0, 1])
      |> Nx.new_axis(0)
    end

    # Squeeze [1, 1, H, W] → [H, W], apply sigmoid, scale to [0, 255] u8,
    # turn into a Vimage with named axes (so vips reads the row stride
    # correctly), and resize back to the original image dimensions.
    defp postprocess(logits, orig_w, orig_h) do
      logits
      |> Nx.squeeze(axes: [0, 1])
      |> Nx.sigmoid()
      |> Nx.multiply(255)
      |> Nx.as_type(:u8)
      |> Nx.new_axis(2)
      |> Nx.rename([:height, :width, :bands])
      |> Image.from_nx!()
      |> Image.resize!(orig_w / @input_size, vertical_scale: orig_h / @input_size)
    end
  end
end
