defmodule Image.BackgroundTest do
  use ExUnit.Case, async: false

  @moduletag :ml
  @moduletag :ortex

  @image_path Path.join(__DIR__, "support/images/puppy.webp")

  describe "mask/2" do
    test "returns a single-band mask at original image dimensions" do
      image = Image.open!(@image_path)
      mask = Image.Background.mask(image)

      assert match?(%Vix.Vips.Image{}, mask)
      assert Image.bands(mask) == 1
      assert Image.width(mask) == Image.width(image)
      assert Image.height(mask) == Image.height(image)
    end

    test "mask covers a meaningful but not overwhelming fraction of pixels" do
      image = Image.open!(@image_path)
      mask = Image.Background.mask(image)

      total_pixels = Image.width(mask) * Image.height(mask)
      {:ok, mask_t} = Image.to_nx(mask, backend: Nx.BinaryBackend)

      # Count pixels above the half-confidence threshold (128).
      foreground_pixels =
        mask_t
        |> Nx.greater(127)
        |> Nx.as_type(:u32)
        |> Nx.sum()
        |> Nx.to_number()

      fraction = foreground_pixels / total_pixels

      assert fraction > 0.02,
             "mask appears empty: only #{Float.round(fraction * 100, 2)}% above threshold"

      assert fraction < 0.95,
             "mask covers nearly the entire image: #{Float.round(fraction * 100, 2)}% above threshold"
    end
  end

  describe "remove/2" do
    test "returns an image with an alpha channel at original dimensions" do
      image = Image.open!(@image_path)

      {:ok, cutout} = Image.Background.remove(image)

      assert match?(%Vix.Vips.Image{}, cutout)
      assert Image.has_alpha?(cutout)
      assert Image.width(cutout) == Image.width(image)
      assert Image.height(cutout) == Image.height(image)
    end

    test "alpha channel reflects foreground confidence (not all opaque, not all transparent)" do
      image = Image.open!(@image_path)
      {:ok, cutout} = Image.Background.remove(image)

      {_bands, alpha} = Image.split_alpha(cutout)
      {:ok, alpha_t} = Image.to_nx(alpha, backend: Nx.BinaryBackend)

      total = Image.width(cutout) * Image.height(cutout)
      transparent = alpha_t |> Nx.less(64) |> Nx.as_type(:u32) |> Nx.sum() |> Nx.to_number()
      opaque = alpha_t |> Nx.greater(192) |> Nx.as_type(:u32) |> Nx.sum() |> Nx.to_number()

      assert transparent / total > 0.05,
             "alpha is too uniformly opaque — background not being removed"

      assert opaque / total > 0.02,
             "alpha is too uniformly transparent — foreground not preserved"
    end
  end
end
