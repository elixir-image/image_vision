defmodule Image.SegmentationTest do
  use ExUnit.Case, async: false

  @moduletag :ml

  @corpus_dir Path.join(__DIR__, "support/images/segmentation")
  @corpus Path.join(@corpus_dir, "corpus.json")
          |> File.read!()
          |> :json.decode()

  # SAM 2 and DETR-panoptic both load on first call via :persistent_term;
  # no supervised process is needed. Models are heavy so we load lazily.

  describe "segment/2 — SAM 2 promptable segmentation" do
    test "returns a mask and score for every corpus image" do
      for entry <- @corpus do
        image = open_image(entry)
        [cx, cy] = entry["prompt_point"]

        result = Image.Segmentation.segment(image, prompt: {:point, cx, cy})

        assert %{mask: %Vix.Vips.Image{} = mask, score: score} = result,
               "expected {mask, score} for #{entry["file"]}"

        assert is_float(score), "score should be float for #{entry["file"]}"
        assert score >= 0.0 and score <= 1.0

        assert Vix.Vips.Image.width(mask) == entry["width"],
               "mask width mismatch for #{entry["file"]}"

        assert Vix.Vips.Image.height(mask) == entry["height"],
               "mask height mismatch for #{entry["file"]}"
      end
    end

    test "mask covers the prompted foreground region" do
      for entry <- @corpus do
        image = open_image(entry)
        [cx, cy] = entry["prompt_point"]

        %{mask: mask} = Image.Segmentation.segment(image, prompt: {:point, cx, cy})

        # The ground-truth foreground covers at least 1% of the image.
        # Our output mask should also be non-trivial (>0.5% of pixels).
        total_pixels = entry["width"] * entry["height"]
        {:ok, stats} = Vix.Vips.Operation.stats(mask)
        # stats tensor shape: [bands+1, 6]; band 0 is "all bands" summary.
        # Column 4 is the sum of pixel values.
        mask_sum = Nx.to_number(stats[[0, 4]])
        # Each foreground pixel = 255; compute fraction of image covered.
        foreground_fraction = mask_sum / (total_pixels * 255)

        assert foreground_fraction > 0.005,
               "mask for #{entry["file"]} covers only #{Float.round(foreground_fraction * 100, 2)}% — expected > 0.5%"
      end
    end

    test "box prompt produces a non-empty mask" do
      entry = hd(@corpus)
      image = open_image(entry)
      [x, y, w, h] = entry["prompt_box"]

      %{mask: mask} = Image.Segmentation.segment(image, prompt: {:box, x, y, w, h})

      total_pixels = entry["width"] * entry["height"]
      {:ok, stats} = Vix.Vips.Operation.stats(mask)
      mask_sum = Nx.to_number(stats[[0, 4]])
      foreground_fraction = mask_sum / (total_pixels * 255)

      assert foreground_fraction > 0.005,
             "box-prompted mask is nearly empty (#{Float.round(foreground_fraction * 100, 2)}%)"
    end

    test "IoU with ground-truth mask exceeds 0.3 for point prompt" do
      for entry <- @corpus do
        image = open_image(entry)
        [cx, cy] = entry["prompt_point"]

        %{mask: pred_mask} = Image.Segmentation.segment(image, prompt: {:point, cx, cy})

        gt_path = Path.join(@corpus_dir, entry["mask_file"])
        {:ok, gt_mask} = Image.open(gt_path)

        iou = mask_iou(pred_mask, gt_mask)

        assert iou > 0.3,
               "IoU for #{entry["file"]} is #{Float.round(iou, 3)} — expected > 0.3"
      end
    end
  end

  describe "segment_panoptic/2 — DETR panoptic segmentation" do
    test "returns a non-empty list of segments for every corpus image" do
      for entry <- @corpus do
        image = open_image(entry)
        segments = Image.Segmentation.segment_panoptic(image)

        assert is_list(segments) and segments != [],
               "expected non-empty segments for #{entry["file"]}"
      end
    end

    test "each segment has a string label and a mask with correct dimensions" do
      entry = hd(@corpus)
      image = open_image(entry)
      segments = Image.Segmentation.segment_panoptic(image)

      for %{label: label, mask: mask} <- segments do
        assert is_binary(label), "label should be a string, got #{inspect(label)}"

        assert Vix.Vips.Image.width(mask) == entry["width"]
        assert Vix.Vips.Image.height(mask) == entry["height"]
      end
    end

    test "detects 'cat' or 'dog' as a segment label for every corpus image" do
      for entry <- @corpus do
        image = open_image(entry)
        expected = entry["coco_class"]
        segments = Image.Segmentation.segment_panoptic(image)
        labels = Enum.map(segments, & &1.label)

        assert Enum.any?(labels, &(&1 == expected)),
               "expected label '#{expected}' in #{inspect(labels)} for #{entry["file"]}"
      end
    end

    test "min_score option filters out low-confidence segments" do
      entry = hd(@corpus)
      image = open_image(entry)

      all_segments = Image.Segmentation.segment_panoptic(image, min_score: 0.1)
      high_segments = Image.Segmentation.segment_panoptic(image, min_score: 0.9)

      assert length(all_segments) >= length(high_segments),
             "higher min_score should return fewer or equal segments"
    end
  end

  # ---------------------------------------------------------------------------
  # Helpers
  # ---------------------------------------------------------------------------

  defp open_image(entry) do
    entry["file"]
    |> then(&Path.join(@corpus_dir, &1))
    |> Image.open!()
  end

  # Compute Intersection-over-Union between two single-band masks.
  # Both masks are expected to be 8-bit greyscale (0 or 255).
  defp mask_iou(%Vix.Vips.Image{} = pred, %Vix.Vips.Image{} = gt) do
    {:ok, pred_t} = Image.to_nx(pred)
    {:ok, gt_t} = Image.to_nx(gt)

    pred_bin = Nx.greater(pred_t, 127)
    gt_bin = Nx.greater(gt_t, 127)

    intersection = Nx.logical_and(pred_bin, gt_bin) |> Nx.sum() |> Nx.to_number()
    union = Nx.logical_or(pred_bin, gt_bin) |> Nx.sum() |> Nx.to_number()

    if union == 0, do: 0.0, else: intersection / union
  end
end
