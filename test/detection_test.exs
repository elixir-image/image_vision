defmodule Image.DetectionTest do
  use ExUnit.Case, async: false

  @moduletag :ml

  @corpus_dir Path.join(__DIR__, "support/images/segmentation")
  @corpus Path.join(@corpus_dir, "corpus.json")
          |> File.read!()
          |> :json.decode()

  # RT-DETR loads lazily on first call via :persistent_term — no supervised
  # process required, but the first test in the suite will pay the load cost.

  describe "detect/2" do
    test "returns a list of detections for every corpus image" do
      for entry <- @corpus do
        image = open_image(entry)
        detections = Image.Detection.detect(image)

        assert is_list(detections),
               "expected a list for #{entry["file"]}, got #{inspect(detections)}"
      end
    end

    test "each detection has label, score, and box keys" do
      entry = hd(@corpus)
      image = open_image(entry)
      detections = Image.Detection.detect(image)

      assert detections != [], "expected at least one detection in #{entry["file"]}"

      for %{label: label, score: score, box: {x, y, w, h}} <- detections do
        assert is_binary(label)
        assert is_float(score) and score >= 0.0 and score <= 1.0
        assert is_number(x) and is_number(y) and is_number(w) and is_number(h)
        assert w > 0 and h > 0
      end
    end

    test "detects 'cat' or 'dog' in every corpus image" do
      for entry <- @corpus do
        image = open_image(entry)
        expected = entry["coco_class"]
        detections = Image.Detection.detect(image)
        labels = Enum.map(detections, & &1.label)

        assert Enum.any?(labels, &(&1 == expected)),
               "expected '#{expected}' in #{inspect(labels)} for #{entry["file"]}"
      end
    end

    test "detected bounding box overlaps ground-truth box with IoU > 0.3" do
      for entry <- @corpus do
        image = open_image(entry)
        expected = entry["coco_class"]
        [gt_x, gt_y, gt_w, gt_h] = entry["prompt_box"]

        detections = Image.Detection.detect(image)

        best_iou =
          detections
          |> Enum.filter(&(&1.label == expected))
          |> Enum.map(fn %{box: {x, y, w, h}} ->
            box_iou({x, y, w, h}, {gt_x, gt_y, gt_w, gt_h})
          end)
          |> Enum.max(fn -> 0.0 end)

        assert best_iou > 0.3,
               "best IoU for #{entry["file"]} is #{Float.round(best_iou, 3)} — expected > 0.3"
      end
    end

    test "min_score option filters detections" do
      entry = hd(@corpus)
      image = open_image(entry)

      all_detections = Image.Detection.detect(image, min_score: 0.1)
      high_detections = Image.Detection.detect(image, min_score: 0.9)

      assert length(all_detections) >= length(high_detections),
             "higher min_score should return fewer or equal detections"
    end

    test "all returned scores meet the requested min_score" do
      entry = hd(@corpus)
      image = open_image(entry)
      min_score = 0.5

      detections = Image.Detection.detect(image, min_score: min_score)

      for %{score: score} <- detections do
        assert score >= min_score,
               "detection score #{score} is below min_score #{min_score}"
      end
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

  # Intersection-over-Union for two boxes in {x, y, w, h} format.
  defp box_iou({x1, y1, w1, h1}, {x2, y2, w2, h2}) do
    ax1 = x1
    ay1 = y1
    ax2 = x1 + w1
    ay2 = y1 + h1

    bx1 = x2
    by1 = y2
    bx2 = x2 + w2
    by2 = y2 + h2

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    union = w1 * h1 + w2 * h2 - intersection

    if union == 0, do: 0.0, else: intersection / union
  end
end
