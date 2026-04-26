defmodule Image.CaptioningTest do
  use ExUnit.Case, async: false

  @moduletag :ml
  @moduletag :captioning

  @images Path.join(__DIR__, "support/images")

  # Start the captioner serving once for the whole suite. Loading
  # BLIP-base takes ~10s and downloads ~990 MB on first run.
  setup_all do
    Application.ensure_all_started(:exla)
    spec = Image.Captioning.captioner()
    start_supervised!(spec)
    :ok
  end

  describe "caption/2" do
    test "returns a non-empty string for a real image" do
      image = Image.open!(Path.join(@images, "puppy.webp"))
      caption = Image.Captioning.caption(image)

      assert is_binary(caption)
      assert byte_size(caption) > 0
    end

    test "captions of different images produce different text" do
      puppy = Image.open!(Path.join(@images, "puppy.webp"))
      car = Image.open!(Path.join(@images, "lamborghini-forsennato-concept.jpg"))

      puppy_caption = Image.Captioning.caption(puppy)
      car_caption = Image.Captioning.caption(car)

      assert puppy_caption != car_caption
    end

    test "puppy image caption mentions a dog-related concept" do
      image = Image.open!(Path.join(@images, "puppy.webp"))
      caption = Image.Captioning.caption(image) |> String.downcase()

      assert Enum.any?(~w(dog puppy spaniel canine), &String.contains?(caption, &1)),
             "expected dog-related word in caption: #{inspect(caption)}"
    end
  end
end
