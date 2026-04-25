defmodule Image.ClassificationTest do
  use ExUnit.Case, async: false

  @moduletag :ml
  @moduletag :classification

  @images Path.join(__DIR__, "support/images")

  # Start the classifier serving once for the whole suite. Loading
  # ConvNeXt-tiny-224 takes several seconds, so we do it here rather
  # than per-test.
  setup_all do
    Application.ensure_all_started(:exla)
    spec = Image.Classification.classifier()
    start_supervised!(spec)
    :ok
  end

  describe "classify/2" do
    test "returns a predictions map with label and score keys" do
      image = Image.open!(Path.join(@images, "puppy.webp"))

      assert %{predictions: [%{label: label, score: score} | _rest]} =
               Image.Classification.classify(image)

      assert is_binary(label)
      assert is_float(score)
      assert score >= 0.0 and score <= 1.0
    end
  end

  describe "labels/2" do
    test "classifies a Cavalier King Charles Spaniel as a Blenheim spaniel" do
      image = Image.open!(Path.join(@images, "puppy.webp"))
      labels = Image.Classification.labels(image, min_score: 0.1)
      assert "Blenheim spaniel" in labels
    end

    test "classifies a Lamborghini as a sports car" do
      image = Image.open!(Path.join(@images, "lamborghini-forsennato-concept.jpg"))
      labels = Image.Classification.labels(image)
      assert "sports car" in labels
    end

    test "classifies a tabby cat as a cat" do
      image = Image.open!(Path.join(@images, "cat.png"))
      labels = Image.Classification.labels(image)
      assert Enum.any?(labels, &String.contains?(&1, "cat"))
    end

    test "returns an empty list when min_score is 1.0" do
      image = Image.open!(Path.join(@images, "puppy.webp"))
      assert Image.Classification.labels(image, min_score: 1.0) == []
    end

    test "returns more labels with a lower min_score threshold" do
      image = Image.open!(Path.join(@images, "puppy.webp"))
      high = Image.Classification.labels(image, min_score: 0.8)
      low = Image.Classification.labels(image, min_score: 0.1)
      assert length(low) >= length(high)
    end
  end
end
