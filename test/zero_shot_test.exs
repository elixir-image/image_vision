defmodule Image.ZeroShotTest do
  use ExUnit.Case, async: false

  @moduletag :ml
  @moduletag :zero_shot

  @images Path.join(__DIR__, "support/images")

  setup_all do
    Application.ensure_all_started(:exla)
    :ok
  end

  describe "classify/3" do
    test "puppy ranks highest against 'a dog'" do
      image = Image.open!(Path.join(@images, "puppy.webp"))
      results = Image.ZeroShot.classify(image, ["a dog", "a cat", "a car"])

      assert is_list(results)
      assert length(results) == 3
      assert [%{label: "a dog"} | _] = results
    end

    test "cat ranks highest against 'a cat'" do
      image = Image.open!(Path.join(@images, "cat.png"))
      results = Image.ZeroShot.classify(image, ["a dog", "a cat", "a car"])

      assert [%{label: "a cat"} | _] = results
    end

    test "car ranks highest against 'a car'" do
      image = Image.open!(Path.join(@images, "lamborghini-forsennato-concept.jpg"))
      results = Image.ZeroShot.classify(image, ["a dog", "a cat", "a car"])

      assert [%{label: "a car"} | _] = results
    end

    test "scores are floats summing to ~1.0" do
      image = Image.open!(Path.join(@images, "puppy.webp"))
      results = Image.ZeroShot.classify(image, ["a dog", "a cat", "a horse"])

      total = results |> Enum.map(& &1.score) |> Enum.sum()
      assert_in_delta total, 1.0, 1.0e-3

      Enum.each(results, fn %{score: s} ->
        assert is_float(s)
        assert s >= 0.0 and s <= 1.0
      end)
    end

    test "results are sorted by descending score" do
      image = Image.open!(Path.join(@images, "puppy.webp"))
      results = Image.ZeroShot.classify(image, ["a dog", "a cat", "a horse"])

      scores = Enum.map(results, & &1.score)
      assert scores == Enum.sort(scores, :desc)
    end

    test "honours a custom prompt template" do
      image = Image.open!(Path.join(@images, "puppy.webp"))

      results =
        Image.ZeroShot.classify(image, ["dog", "cat"],
          template: "a high-resolution photograph of a {label}"
        )

      assert [%{label: "dog"} | _] = results
    end

    test "template: nil uses labels verbatim" do
      image = Image.open!(Path.join(@images, "puppy.webp"))
      results = Image.ZeroShot.classify(image, ["a photo of a dog", "a photo of a car"], template: nil)
      assert [%{label: "a photo of a dog"} | _] = results
    end
  end

  describe "label/3" do
    test "returns the single best label as a string" do
      image = Image.open!(Path.join(@images, "puppy.webp"))
      assert Image.ZeroShot.label(image, ["a dog", "a cat"]) == "a dog"
    end
  end

  describe "similarity/3" do
    test "two views of the same subject score higher than two unrelated images" do
      puppy = Image.open!(Path.join(@images, "puppy.webp"))
      cat = Image.open!(Path.join(@images, "cat.png"))
      car = Image.open!(Path.join(@images, "lamborghini-forsennato-concept.jpg"))

      # Both animals (similar feature space) should score higher than animal vs vehicle.
      animals = Image.ZeroShot.similarity(puppy, cat)
      cross = Image.ZeroShot.similarity(puppy, car)

      assert animals > cross
    end

    test "similarity is in [-1.0, 1.0]" do
      a = Image.open!(Path.join(@images, "puppy.webp"))
      b = Image.open!(Path.join(@images, "cat.png"))

      score = Image.ZeroShot.similarity(a, b)
      assert is_float(score)
      assert score >= -1.0 and score <= 1.0
    end
  end
end
